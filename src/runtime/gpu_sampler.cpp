#include "runtime/gpu_sampler.h"

#include <cstring>

#include "core/tensor_ggml.hpp"
#include "core/util.h"

namespace sd {

    bool GpuSampler::supported(ggml_backend_t backend) {
        ggml_context* ctx = ggml_init({ggml_tensor_overhead() * 6, nullptr, true});
        ggml_tensor* x    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_tensor* p4   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_tensor* p6   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
        const bool ok     = ggml_backend_supports_op(backend, ggml_denoise(ctx, x, x, nullptr, nullptr, nullptr, nullptr, p4)) &&
                            ggml_backend_supports_op(backend, ggml_sample_step(ctx, x, x, p6, GGML_SAMPLE_STEP_EULER_ANCESTRAL, 0));
        ggml_free(ctx);
        return ok;
    }

    GpuSampler::GpuSampler(ggml_backend_t backend, const Tensor<float>& x, bool with_mask, int steps)
        : backend_(backend), shape_(x.shape()), slot_count_(8 * static_cast<int64_t>(steps) + 32) {
        ggml_init_params params = {ggml_tensor_overhead() * static_cast<size_t>(16 + slot_count_), nullptr, true};
        ctx_                    = ggml_init(params);
        auto latent             = [&](const char* name) {
            ggml_tensor* t = make_ggml_tensor(ctx_, x, false);
            ggml_set_name(t, name);
            return t;
        };
        x_                = latent("gpu_sampler.x");
        pred_[COND]       = latent("gpu_sampler.pred_cond");
        pred_[UNCOND]     = latent("gpu_sampler.pred_uncond");
        pred_[IMG_UNCOND] = latent("gpu_sampler.pred_img_uncond");
        if (with_mask) {
            mask_ = latent("gpu_sampler.mask");
            init_ = latent("gpu_sampler.init");
        }
        const size_t slot_bytes              = static_cast<size_t>(kSlotFloats * slot_count_) * sizeof(float);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_));
        if (host_buft != nullptr) {
            slot_buffer_ = ggml_backend_buft_alloc_buffer(host_buft, slot_bytes);
        }
        if (slot_buffer_ == nullptr) {
            slot_table_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, kSlotFloats * slot_count_);
            ggml_set_name(slot_table_, "gpu_sampler.slots");
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (slot_buffer_ == nullptr && buffer_ != nullptr) {
            slot_buffer_owned_ = false;
            slot_buffer_       = buffer_;
        }
        if (buffer_ == nullptr) {
            LOG_ERROR("gpu sampler: failed to allocate the latent store on %s", ggml_backend_name(backend_));
        }
    }

    GpuSampler::~GpuSampler() {
        ggml_backend_synchronize(backend_);
        ggml_gallocr_free(step_allocr_);
        ggml_free(step_ctx_);
        for (auto& entry : constants_) {
            ggml_backend_buffer_free(entry.second.buffer);
            ggml_free(entry.second.ctx);
        }
        if (slot_buffer_owned_) {
            ggml_backend_buffer_free(slot_buffer_);
        }
        ggml_backend_buffer_free(buffer_);
        ggml_free(ctx_);
    }

    void GpuSampler::upload_x(const Tensor<float>& x) {
        ggml_backend_tensor_set(x_, x.data(), 0, ggml_nbytes(x_));
    }

    void GpuSampler::upload_mask(const Tensor<float>& mask, const Tensor<float>& init) {
        ggml_backend_tensor_set(mask_, mask.data(), 0, ggml_nbytes(mask_));
        ggml_backend_tensor_set(init_, init.data(), 0, ggml_nbytes(init_));
    }

    void GpuSampler::set_c_in(float c_in) {
        c_in_ = scalar({1}, &c_in);
    }

    Tensor<float> GpuSampler::download(const ggml_tensor* tensor) const {
        Tensor<float> out(shape_);
        ggml_backend_synchronize(backend_);
        ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
        return out;
    }

    ggml_tensor* GpuSampler::find_constant(const std::string& key) {
        auto it = constants_.find(key);
        return it == constants_.end() ? nullptr : it->second.tensor;
    }

    ggml_tensor* GpuSampler::constant(const std::string& key, ggml_type type, const std::vector<int64_t>& ne, const void* data) {
        auto it = constants_.find(key);
        if (it != constants_.end()) {
            return it->second.tensor;
        }
        Constant entry;
        entry.ctx    = ggml_init({ggml_tensor_overhead(), nullptr, true});
        entry.tensor = ggml_new_tensor(entry.ctx, type, static_cast<int>(ne.size()), ne.data());
        ggml_set_name(entry.tensor, key.c_str());
        entry.buffer = ggml_backend_alloc_ctx_tensors(entry.ctx, backend_);
        if (entry.buffer == nullptr) {
            LOG_WARN("gpu sampler: failed to allocate run constant '%s' (%.2f MB), using a per-call input",
                     key.c_str(), ggml_nbytes(entry.tensor) / (1024.0 * 1024.0));
            ggml_free(entry.ctx);
            constants_[key] = Constant{};
            return nullptr;
        }
        ggml_backend_tensor_set(entry.tensor, data, 0, ggml_nbytes(entry.tensor));
        constants_[key] = entry;
        return entry.tensor;
    }

    ggml_tensor* GpuSampler::scalar(const std::vector<int64_t>& ne, const float* data) {
        GGML_ASSERT(slots_used_ < slot_count_);
        std::array<int64_t, GGML_MAX_DIMS> dims = {1, 1, 1, 1};
        int64_t n                               = 1;
        for (size_t i = 0; i < ne.size(); ++i) {
            dims[i] = ne[i];
            n *= ne[i];
        }
        GGML_ASSERT(n <= kSlotFloats);
        const int64_t slot  = slots_used_++;
        char* base          = slot_table_ != nullptr ? static_cast<char*>(slot_table_->data)
                                                     : static_cast<char*>(ggml_backend_buffer_get_base(slot_buffer_));
        ggml_tensor* tensor = ggml_new_tensor(ctx_, GGML_TYPE_F32, static_cast<int>(ne.size()), dims.data());
        GGML_ASSERT(ggml_backend_tensor_alloc(slot_buffer_, tensor, base + static_cast<size_t>(slot * kSlotFloats) * sizeof(float)) == GGML_STATUS_SUCCESS);
        if (ggml_backend_buffer_is_host(slot_buffer_)) {
            std::memcpy(tensor->data, data, static_cast<size_t>(n) * sizeof(float));
        } else {
            ggml_backend_tensor_set(tensor, data, 0, static_cast<size_t>(n) * sizeof(float));
        }
        return tensor;
    }

    size_t GpuSampler::resident_bytes(ggml_backend_t backend) const {
        if (backend != backend_) {
            return 0;
        }
        size_t bytes = buffer_ != nullptr ? ggml_backend_buffer_get_size(buffer_) : 0;
        for (const auto& entry : constants_) {
            bytes += entry.second.buffer != nullptr ? ggml_backend_buffer_get_size(entry.second.buffer) : 0;
        }
        if (step_allocr_ != nullptr) {
            bytes += ggml_gallocr_get_buffer_size(step_allocr_, 0);
        }
        return bytes;
    }

    bool GpuSampler::step(const std::array<float, 4>& p_denoise,
                          bool has_uncond,
                          bool has_img_uncond,
                          const std::array<float, 6>& p_step,
                          ggml_sample_step_mode mode,
                          uint32_t seed) {
        ggml_tensor* p_denoise_slot = scalar({4}, p_denoise.data());
        ggml_tensor* p_step_slot    = scalar({6}, p_step.data());
        ggml_free(step_ctx_);
        step_ctx_             = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true});
        ggml_tensor* denoised = ggml_denoise(step_ctx_, x_, pred_[COND],
                                             has_uncond ? pred_[UNCOND] : nullptr,
                                             has_img_uncond ? pred_[IMG_UNCOND] : nullptr,
                                             mask_, init_, GGMLRunner::device_scalar(step_ctx_, p_denoise_slot));
        ggml_tensor* next     = ggml_sample_step(step_ctx_, x_, denoised, GGMLRunner::device_scalar(step_ctx_, p_step_slot), mode, seed);
        ggml_cgraph* graph    = ggml_new_graph(step_ctx_);
        ggml_build_forward_expand(graph, ggml_cpy(step_ctx_, next, x_));
        const int variant = (has_uncond ? 1 : 0) + (has_img_uncond ? 2 : 0) + (mode == GGML_SAMPLE_STEP_EULER_ANCESTRAL ? 4 : 0);
        if (variant != step_variant_) {
            if (step_allocr_ != nullptr) {
                ggml_backend_synchronize(backend_);
            }
            ggml_gallocr_free(step_allocr_);
            step_allocr_  = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            step_variant_ = variant;
        }
        if (!ggml_gallocr_alloc_graph(step_allocr_, graph)) {
            LOG_ERROR("gpu sampler: failed to allocate the step graph");
            return false;
        }
        return ggml_backend_graph_compute_async(backend_, graph) == GGML_STATUS_SUCCESS;
    }

}  // namespace sd
