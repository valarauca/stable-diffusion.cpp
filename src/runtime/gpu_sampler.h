#ifndef __SD_RUNTIME_GPU_SAMPLER_H__
#define __SD_RUNTIME_GPU_SAMPLER_H__

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/ggml_runner.h"
#include "core/tensor.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace sd {

    // Keeps the latent, the model predictions and the sampler state on one backend for a whole
    // sampling run. Diffusion runners read x and write predictions through set_device_io, or through
    // download and upload_pred when they only take host tensors, and a
    // small graph of GGML_OP_DENOISE + GGML_OP_SAMPLE_STEP advances x in place each step. Run
    // constants and per-call scalars are served as RunnerDeviceInputs, so nothing waits on the
    // backend until the final download.
    class GpuSampler : public RunnerDeviceInputs {
    public:
        enum Slot {
            COND,
            UNCOND,
            IMG_UNCOND,
            SLOT_COUNT,
        };

        // True when backend runs GGML_OP_DENOISE and GGML_OP_SAMPLE_STEP.
        static bool supported(ggml_backend_t backend);

        GpuSampler(ggml_backend_t backend, const Tensor<float>& x, bool with_mask, int steps);
        ~GpuSampler() override;
        GpuSampler(const GpuSampler&)            = delete;
        GpuSampler& operator=(const GpuSampler&) = delete;

        bool ok() const { return buffer_ != nullptr; }

        ggml_tensor* x() const { return x_; }
        ggml_tensor* c_in() const { return c_in_; }
        ggml_tensor* pred(Slot slot) const { return pred_[slot]; }

        void upload_x(const Tensor<float>& x);
        void upload_mask(const Tensor<float>& mask, const Tensor<float>& init);
        void upload_pred(Slot slot, const Tensor<float>& pred);
        void set_c_in(float c_in);
        Tensor<float> download(const ggml_tensor* tensor) const;

        // p_denoise: {c_skip, c_out, txt_cfg, img_cfg}. p_step: {sigma, sigma_next, sigma_down,
        // sigma_up, alpha, step}. Submits without waiting for the backend.
        bool step(const std::array<float, 4>& p_denoise,
                  bool has_uncond,
                  bool has_img_uncond,
                  const std::array<float, 6>& p_step,
                  ggml_sample_step_mode mode,
                  uint32_t seed);

        ggml_tensor* constant(const std::string& key, ggml_type type, const std::vector<int64_t>& ne, const void* data) override;
        ggml_tensor* find_constant(const std::string& key) override;
        ggml_tensor* scalar(const std::vector<int64_t>& ne, const float* data) override;
        size_t resident_bytes(ggml_backend_t backend) const override;

    private:
        static constexpr int64_t kSlotFloats = 16;

        struct Constant {
            ggml_context* ctx             = nullptr;
            ggml_backend_buffer_t buffer  = nullptr;
            ggml_tensor* tensor           = nullptr;
        };

        ggml_backend_t backend_       = nullptr;
        ggml_context* ctx_            = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
        std::vector<int64_t> shape_;
        ggml_tensor* x_         = nullptr;
        ggml_tensor* c_in_      = nullptr;
        ggml_tensor* mask_      = nullptr;
        ggml_tensor* init_      = nullptr;
        ggml_tensor* slot_table_ = nullptr;
        std::array<ggml_tensor*, SLOT_COUNT> pred_{};
        ggml_backend_buffer_t slot_buffer_ = nullptr;
        bool slot_buffer_owned_            = true;
        int64_t slot_count_ = 0;
        int64_t slots_used_ = 0;
        std::map<std::string, Constant> constants_;
        ggml_context* step_ctx_     = nullptr;
        ggml_gallocr_t step_allocr_ = nullptr;
        int step_variant_           = -1;
    };

}  // namespace sd

#endif  // __SD_RUNTIME_GPU_SAMPLER_H__
