#ifndef __SD_MODEL_DIFFUSION_SEEDVR2_HPP__
#define __SD_MODEL_DIFFUSION_SEEDVR2_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/model.hpp"

// Ref: https://github.com/ByteDance-Seed/SeedVR (NaDiT 7B, mmdit_sr blocks)

namespace SeedVR2 {

    constexpr int SEEDVR2_GRAPH_SIZE = 131072;

    struct SeedVR2Config {
        int64_t hidden_size  = 3072;
        int64_t head_dim     = 128;
        int64_t num_heads    = 24;
        int64_t mlp_ratio    = 4;
        int64_t in_channels  = 33;
        int64_t out_channels = 16;
        int64_t text_dim     = 5120;
        int num_layers       = 36;
        float eps            = 1e-5f;

        static SeedVR2Config detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            SeedVR2Config config;
            config.num_layers = 0;
            for (const auto& [name, storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                const std::string key = name.substr(prefix.size());
                if (starts_with(key, "blocks.")) {
                    auto items = split_string(key, '.');
                    if (items.size() > 1) {
                        config.num_layers = std::max(config.num_layers, atoi(items[1].c_str()) + 1);
                    }
                } else if (key == "vid_in.proj.weight") {
                    config.in_channels = storage.ne[0] / 4;
                    config.hidden_size = storage.ne[1];
                } else if (key == "vid_out.proj.weight") {
                    config.out_channels = storage.ne[1] / 4;
                } else if (key == "txt_in.weight") {
                    config.text_dim = storage.ne[0];
                }
            }
            config.num_heads = config.hidden_size / config.head_dim;
            return config;
        }
    };

    struct WindowSlice {
        int64_t t0, t1, h0, h1, w0, w1;
    };

    // 720p-sized attention windows over a (t, h, w) token grid in the
    // reference order (w outer, t inner). The shifted variant moves every axis
    // that has more than one window by half a window.
    static std::vector<WindowSlice> make_windows(int64_t t, int64_t h, int64_t w, bool shifted) {
        const double scale      = std::sqrt(45.0 * 80.0 / static_cast<double>(h * w));
        const int64_t resized_h = std::max<int64_t>(1, static_cast<int64_t>(std::nearbyint(h * scale)));
        const int64_t resized_w = std::max<int64_t>(1, static_cast<int64_t>(std::nearbyint(w * scale)));
        const int64_t window_h  = (resized_h + 2) / 3;
        const int64_t window_w  = (resized_w + 2) / 3;
        const int64_t window_t  = (std::min<int64_t>(t, 30) + 3) / 4;

        auto axis = [shifted](int64_t size, int64_t window) {
            std::vector<std::pair<int64_t, int64_t>> ranges;
            if (!shifted) {
                for (int64_t i = 0; i * window < size; ++i) {
                    ranges.push_back({i * window, std::min((i + 1) * window, size)});
                }
                return ranges;
            }
            const double shift = window < size ? 0.5 : 0.0;
            int64_t count      = static_cast<int64_t>(std::ceil((size - shift) / window));
            count              = shift > 0 ? count + 1 : 1;
            for (int64_t i = 0; i < count; ++i) {
                const int64_t start = std::max<int64_t>(static_cast<int64_t>((i - shift) * window), 0);
                const int64_t end   = std::min<int64_t>(static_cast<int64_t>((i - shift + 1) * window), size);
                if (end > start) {
                    ranges.push_back({start, end});
                }
            }
            return ranges;
        };

        const auto ts = axis(t, window_t);
        const auto hs = axis(h, window_h);
        const auto ws = axis(w, window_w);
        std::vector<WindowSlice> windows;
        for (const auto& [w0, w1] : ws) {
            for (const auto& [h0, h1] : hs) {
                for (const auto& [t0, t1] : ts) {
                    windows.push_back({t0, t1, h0, h1, w0, w1});
                }
            }
        }
        return windows;
    }

    static float torch_linspace(float start, float end, int64_t steps, int64_t i) {
        if (steps == 1) {
            return start;
        }
        const float step = (end - start) / static_cast<float>(steps - 1);
        return i < steps / 2 ? start + step * i : end - step * (steps - 1 - i);
    }

    // Interleaved rotation of the first 60 head dims, 20 per axis, with
    // positions spread over [-1, 1] across the window. The frequencies are
    // pi * linspace(1, 128, 10) rounded to fp16, as stored in the checkpoint
    // rope buffers that the reference model loads.
    static std::vector<float> window_rope(int64_t frames, int64_t rows, int64_t cols, int64_t head_dim) {
        constexpr int axis_freqs = 10;
        std::array<float, axis_freqs> freqs;
        for (int j = 0; j < axis_freqs; ++j) {
            freqs[j] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(torch_linspace(1.f, 128.f, axis_freqs, j) * static_cast<float>(M_PI)));
        }
        const int64_t pairs = head_dim / 2;
        std::vector<float> pe;
        pe.reserve(static_cast<size_t>(frames * rows * cols * pairs * 4));
        for (int64_t a = 0; a < frames; ++a) {
            for (int64_t b = 0; b < rows; ++b) {
                for (int64_t c = 0; c < cols; ++c) {
                    const float pos[3] = {torch_linspace(-1.f, 1.f, frames, a),
                                          torch_linspace(-1.f, 1.f, rows, b),
                                          torch_linspace(-1.f, 1.f, cols, c)};
                    for (int64_t i = 0; i < pairs; ++i) {
                        const float angle = i < 3 * axis_freqs ? pos[i / axis_freqs] * freqs[i % axis_freqs] : 0.f;
                        const float cos_v = std::cos(angle);
                        const float sin_v = std::sin(angle);
                        pe.insert(pe.end(), {cos_v, -sin_v, sin_v, cos_v});
                    }
                }
            }
        }
        return pe;
    }

    // Windows grouped by shape so each group runs as one batched attention.
    struct WindowGroup {
        int64_t frames = 0;
        int64_t rows   = 0;
        int64_t cols   = 0;
        int64_t count  = 0;
        std::vector<int32_t> tokens;
        std::vector<float> pe;
    };

    struct WindowPlan {
        std::vector<WindowGroup> groups;
        std::vector<int32_t> inverse;
        int64_t windows = 0;
    };

    static WindowPlan plan_windows(int64_t t, int64_t h, int64_t w, bool shifted, int64_t head_dim) {
        WindowPlan plan;
        std::map<std::array<int64_t, 3>, size_t> group_of;
        for (const auto& s : make_windows(t, h, w, shifted)) {
            const std::array<int64_t, 3> shape = {s.t1 - s.t0, s.h1 - s.h0, s.w1 - s.w0};
            auto it                            = group_of.find(shape);
            if (it == group_of.end()) {
                it = group_of.emplace(shape, plan.groups.size()).first;
                plan.groups.push_back({shape[0], shape[1], shape[2]});
            }
            auto& group = plan.groups[it->second];
            for (int64_t a = s.t0; a < s.t1; ++a) {
                for (int64_t b = s.h0; b < s.h1; ++b) {
                    for (int64_t c = s.w0; c < s.w1; ++c) {
                        group.tokens.push_back(static_cast<int32_t>((a * h + b) * w + c));
                    }
                }
            }
            group.count++;
            plan.windows++;
        }
        plan.inverse.resize(static_cast<size_t>(t * h * w));
        int32_t position = 0;
        for (auto& group : plan.groups) {
            group.pe = window_rope(group.frames, group.rows, group.cols, head_dim);
            for (int32_t token : group.tokens) {
                plan.inverse[token] = position++;
            }
        }
        return plan;
    }

    struct WindowInputs {
        const WindowPlan* plan = nullptr;
        std::vector<ggml_tensor*> tokens;
        std::vector<ggml_tensor*> pe;
        ggml_tensor* inverse = nullptr;
    };

    // Shift, scale and gate rows of the shared time embedding, indexed by
    // [0] attn and [1] mlp.
    struct Modulation {
        std::array<ggml_tensor*, 2> shift;
        std::array<ggml_tensor*, 2> scale;
        std::array<ggml_tensor*, 2> gate;
    };

    struct AdaSingle : public GGMLBlock {
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            for (const char* name : {"attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate"}) {
                params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            }
        }

        explicit AdaSingle(int64_t dim)
            : dim(dim) {}

        ggml_tensor* get(const std::string& name) {
            return params[name];
        }
    };

    struct TransformerBlock : public GGMLBlock {
        SeedVR2Config config;

        explicit TransformerBlock(const SeedVR2Config& config)
            : config(config) {
            const int64_t hidden = config.hidden_size;
            for (const std::string stream : {"vid", "txt"}) {
                blocks["attn.proj_qkv." + stream]     = std::make_shared<Linear>(hidden, 3 * hidden, false);
                blocks["attn.proj_out." + stream]     = std::make_shared<Linear>(hidden, hidden, true);
                blocks["attn.norm_q." + stream]       = std::make_shared<RMSNorm>(config.head_dim, config.eps);
                blocks["attn.norm_k." + stream]       = std::make_shared<RMSNorm>(config.head_dim, config.eps);
                blocks["mlp." + stream + ".proj_in"]  = std::make_shared<Linear>(hidden, hidden * config.mlp_ratio, true);
                blocks["mlp." + stream + ".proj_out"] = std::make_shared<Linear>(hidden * config.mlp_ratio, hidden, true);
                blocks["ada." + stream]               = std::make_shared<AdaSingle>(hidden);
            }
        }

        ggml_tensor* modulate(GGMLRunnerContext* ctx, ggml_tensor* x, const Modulation& emb, const std::string& stream, int layer) {
            auto ada        = std::dynamic_pointer_cast<AdaSingle>(blocks["ada." + stream]);
            const char* tag = layer == 0 ? "attn" : "mlp";
            auto scale      = ggml_add(ctx->ggml_ctx, emb.scale[layer], ada->get(std::string(tag) + "_scale"));
            auto shift      = ggml_add(ctx->ggml_ctx, emb.shift[layer], ada->get(std::string(tag) + "_shift"));
            x               = ggml_mul(ctx->ggml_ctx, ggml_rms_norm(ctx->ggml_ctx, x, config.eps), scale);
            return ggml_add(ctx->ggml_ctx, x, shift);
        }

        ggml_tensor* gate(GGMLRunnerContext* ctx, ggml_tensor* x, const Modulation& emb, const std::string& stream, int layer) {
            auto ada = std::dynamic_pointer_cast<AdaSingle>(blocks["ada." + stream]);
            auto g   = ggml_add(ctx->ggml_ctx, emb.gate[layer], ada->get(layer == 0 ? "attn_gate" : "mlp_gate"));
            return ggml_mul(ctx->ggml_ctx, x, g);
        }

        // Returns q and k RMS normed and v, each [head_dim, heads, n].
        std::array<ggml_tensor*, 3> project_qkv(GGMLRunnerContext* ctx, ggml_tensor* x, const std::string& stream) {
            auto proj   = std::dynamic_pointer_cast<Linear>(blocks["attn.proj_qkv." + stream]);
            auto norm_q = std::dynamic_pointer_cast<RMSNorm>(blocks["attn.norm_q." + stream]);
            auto norm_k = std::dynamic_pointer_cast<RMSNorm>(blocks["attn.norm_k." + stream]);

            const int64_t n = x->ne[1];
            auto qkv        = ggml_reshape_4d(ctx->ggml_ctx, proj->forward(ctx, x), config.head_dim, config.num_heads, 3, n);
            auto q          = ggml_view_3d(ctx->ggml_ctx, qkv, config.head_dim, config.num_heads, n, qkv->nb[1], qkv->nb[3], 0);
            auto k          = ggml_view_3d(ctx->ggml_ctx, qkv, config.head_dim, config.num_heads, n, qkv->nb[1], qkv->nb[3], qkv->nb[2]);
            auto v          = ggml_view_3d(ctx->ggml_ctx, qkv, config.head_dim, config.num_heads, n, qkv->nb[1], qkv->nb[3], 2 * qkv->nb[2]);
            return {norm_q->forward(ctx, q), norm_k->forward(ctx, k), ggml_cont(ctx->ggml_ctx, v)};
        }

        // Every window attends over its own video tokens plus all text
        // tokens. Video rows return to token order and the text rows are
        // averaged over all windows.
        std::pair<ggml_tensor*, ggml_tensor*> attention(GGMLRunnerContext* ctx,
                                                        ggml_tensor* vid,
                                                        ggml_tensor* txt,
                                                        const WindowInputs& windows) {
            auto g              = ctx->ggml_ctx;
            const int64_t D     = config.head_dim;
            const int64_t heads = config.num_heads;
            const int64_t C     = config.hidden_size;
            const int64_t L     = vid->ne[1];
            const int64_t Lt    = txt->ne[1];

            auto [vq, vk, vv] = project_qkv(ctx, vid, "vid");
            auto [tq, tk, tv] = project_qkv(ctx, txt, "txt");
            vq                = ggml_reshape_2d(g, vq, C, L);
            vk                = ggml_reshape_2d(g, vk, C, L);
            vv                = ggml_reshape_2d(g, vv, C, L);
            tq                = ggml_cont(g, ggml_permute(g, tq, 0, 2, 1, 3));
            tk                = ggml_cont(g, ggml_permute(g, tk, 0, 2, 1, 3));
            tv                = ggml_reshape_4d(g, tv, D, heads, Lt, 1);

            std::vector<ggml_tensor*> vid_parts;
            ggml_tensor* txt_sum = nullptr;
            for (size_t i = 0; i < windows.plan->groups.size(); ++i) {
                const auto& group = windows.plan->groups[i];
                const int64_t Lw  = group.frames * group.rows * group.cols;
                const int64_t n   = group.count;

                auto q = ggml_reshape_4d(g, ggml_get_rows(g, vq, windows.tokens[i]), D, heads, Lw, n);
                auto k = ggml_reshape_4d(g, ggml_get_rows(g, vk, windows.tokens[i]), D, heads, Lw, n);
                auto v = ggml_reshape_4d(g, ggml_get_rows(g, vv, windows.tokens[i]), D, heads, Lw, n);
                q      = Rope::apply_rope(ctx, q, windows.pe[i], true);
                k      = Rope::apply_rope(ctx, k, windows.pe[i], true);

                auto text_heads = ggml_new_tensor_3d(g, GGML_TYPE_F32, D, Lt, heads * n);
                q               = ggml_concat(g, q, ggml_repeat(g, tq, text_heads), 1);
                k               = ggml_concat(g, k, ggml_repeat(g, tk, text_heads), 1);
                auto text_v     = n > 1 ? ggml_repeat(g, tv, ggml_new_tensor_4d(g, GGML_TYPE_F32, D, heads, Lt, n)) : tv;
                v               = ggml_concat(g, v, text_v, 2);

                auto out = ggml_ext_attention_ext(ctx, q, k, v, heads, nullptr, true, ctx->flash_attn_enabled);
                vid_parts.push_back(ggml_reshape_2d(g, ggml_ext_slice(g, out, 1, 0, Lw), C, Lw * n));
                auto text_out = ggml_ext_slice(g, out, 1, Lw, Lw + Lt);
                if (n > 1) {
                    text_out = ggml_reshape_2d(g, text_out, C * Lt, n);
                    text_out = ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, text_out)));
                    text_out = ggml_reshape_2d(g, text_out, C, Lt);
                }
                txt_sum = txt_sum == nullptr ? text_out : ggml_add(g, txt_sum, text_out);
            }

            auto vid_out = ggml_get_rows(g, ggml_ext_vec_concat(g, vid_parts, 1), windows.inverse);
            auto txt_out = ggml_ext_scale(g, txt_sum, 1.f / static_cast<float>(windows.plan->windows));
            vid_out      = std::dynamic_pointer_cast<Linear>(blocks["attn.proj_out.vid"])->forward(ctx, vid_out);
            txt_out      = std::dynamic_pointer_cast<Linear>(blocks["attn.proj_out.txt"])->forward(ctx, txt_out);
            return {vid_out, txt_out};
        }

        ggml_tensor* mlp(GGMLRunnerContext* ctx, ggml_tensor* x, const std::string& stream) {
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["mlp." + stream + ".proj_in"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["mlp." + stream + ".proj_out"]);
            return proj_out->forward(ctx, ggml_gelu_inplace(ctx->ggml_ctx, proj_in->forward(ctx, x)));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* vid,
                                                      ggml_tensor* txt,
                                                      const Modulation& emb,
                                                      const WindowInputs& windows) {
            auto [vid_attn, txt_attn] = attention(ctx, modulate(ctx, vid, emb, "vid", 0), modulate(ctx, txt, emb, "txt", 0), windows);
            vid                       = ggml_add(ctx->ggml_ctx, vid, gate(ctx, vid_attn, emb, "vid", 0));
            txt                       = ggml_add(ctx->ggml_ctx, txt, gate(ctx, txt_attn, emb, "txt", 0));
            vid                       = ggml_add(ctx->ggml_ctx, vid, gate(ctx, mlp(ctx, modulate(ctx, vid, emb, "vid", 1), "vid"), emb, "vid", 1));
            txt                       = ggml_add(ctx->ggml_ctx, txt, gate(ctx, mlp(ctx, modulate(ctx, txt, emb, "txt", 1), "txt"), emb, "txt", 1));
            return {vid, txt};
        }
    };

    struct NaDiT : public GGMLBlock {
        SeedVR2Config config;

        explicit NaDiT(const SeedVR2Config& config)
            : config(config) {
            const int64_t hidden      = config.hidden_size;
            blocks["vid_in.proj"]     = std::make_shared<Linear>(config.in_channels * 4, hidden, true);
            blocks["txt_in"]          = std::make_shared<Linear>(config.text_dim, hidden, true);
            blocks["emb_in.proj_in"]  = std::make_shared<Linear>(256, hidden, true);
            blocks["emb_in.proj_hid"] = std::make_shared<Linear>(hidden, hidden, true);
            blocks["emb_in.proj_out"] = std::make_shared<Linear>(hidden, 6 * hidden, true);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TransformerBlock>(config);
            }
            blocks["vid_out.proj"] = std::make_shared<Linear>(hidden, config.out_channels * 4, true);
        }

        // [W, H, T, C] to [4C, T*H/2*W/2], features ordered (ph, pw, c) with c innermost.
        static ggml_tensor* patchify(ggml_context* g, ggml_tensor* x) {
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t T = x->ne[2];
            const int64_t C = x->ne[3];
            x               = ggml_reshape_4d(g, x, W, 2, (H / 2) * T, C);
            x               = ggml_cont(g, ggml_permute(g, x, 1, 2, 3, 0));
            x               = ggml_reshape_4d(g, x, 2 * C, W / 2, 2, (H / 2) * T);
            x               = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
            return ggml_reshape_2d(g, x, 4 * C, T * (H / 2) * (W / 2));
        }

        static ggml_tensor* unpatchify(ggml_context* g, ggml_tensor* x, int64_t W, int64_t H, int64_t T, int64_t C) {
            x = ggml_reshape_4d(g, x, 2 * C, 2, W / 2, (H / 2) * T);
            x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
            x = ggml_reshape_4d(g, x, C, W, 2, (H / 2) * T);
            x = ggml_cont(g, ggml_permute(g, x, 3, 0, 1, 2));
            return ggml_reshape_4d(g, x, W, H, T, C);
        }

        Modulation time_embedding(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto g               = ctx->ggml_ctx;
            const int64_t hidden = config.hidden_size;
            auto sinusoid        = ggml_timestep_embedding(g, timesteps, 256, 10000);
            auto cos_half        = ggml_view_2d(g, sinusoid, 128, sinusoid->ne[1], sinusoid->nb[1], 0);
            auto sin_half        = ggml_view_2d(g, sinusoid, 128, sinusoid->ne[1], sinusoid->nb[1], 128 * sizeof(float));
            auto emb             = ggml_concat(g, sin_half, cos_half, 0);
            emb                  = ggml_silu(g, std::dynamic_pointer_cast<Linear>(blocks["emb_in.proj_in"])->forward(ctx, emb));
            emb                  = ggml_silu(g, std::dynamic_pointer_cast<Linear>(blocks["emb_in.proj_hid"])->forward(ctx, emb));
            emb                  = std::dynamic_pointer_cast<Linear>(blocks["emb_in.proj_out"])->forward(ctx, emb);
            emb                  = ggml_reshape_2d(g, emb, 6, hidden);
            std::array<ggml_tensor*, 6> rows;
            for (int k = 0; k < 6; ++k) {
                rows[k] = ggml_reshape_1d(g, ggml_cont(g, ggml_view_2d(g, emb, 1, hidden, emb->nb[1], k * sizeof(float))), hidden);
            }
            return {{rows[0], rows[3]}, {rows[1], rows[4]}, {rows[2], rows[5]}};
        }

        // x is [W, H, T, in_channels], context is [text_dim, L_text]. Even
        // layers use the plain windows, odd layers the shifted ones.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             ggml_tensor* timesteps,
                             const std::array<WindowInputs, 2>& windows) {
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t T = x->ne[2];

            auto vid = std::dynamic_pointer_cast<Linear>(blocks["vid_in.proj"])->forward(ctx, patchify(ctx->ggml_ctx, x));
            auto txt = std::dynamic_pointer_cast<Linear>(blocks["txt_in"])->forward(ctx, context);
            auto emb = time_embedding(ctx, timesteps);
            for (int i = 0; i < config.num_layers; ++i) {
                auto block         = std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(i)]);
                std::tie(vid, txt) = block->forward(ctx, vid, txt, emb, windows[i % 2]);
            }
            vid = std::dynamic_pointer_cast<Linear>(blocks["vid_out.proj"])->forward(ctx, vid);
            return unpatchify(ctx->ggml_ctx, vid, W, H, T, config.out_channels);
        }
    };

    struct SeedVR2Runner : public DiffusionModelRunner {
        SeedVR2Config config;
        NaDiT model;
        std::array<WindowPlan, 2> plans;

        SeedVR2Runner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map      = {},
                      const std::string prefix                            = "",
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(SeedVR2Config::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "seedvr2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& c_concat_tensor) {
            ggml_cgraph* gf        = new_graph_custom(SEEDVR2_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_input(context_tensor);
            ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
            if (c_concat != nullptr) {
                x = ggml_concat(compute_ctx, x, c_concat, 3);
            }

            const int64_t t = x->ne[2];
            const int64_t h = x->ne[1] / 2;
            const int64_t w = x->ne[0] / 2;
            std::array<WindowInputs, 2> windows;
            for (int p = 0; p < 2; ++p) {
                plans[p]           = plan_windows(t, h, w, p == 1, config.head_dim);
                windows[p].plan    = &plans[p];
                windows[p].inverse = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, t * h * w);
                set_backend_tensor_data(windows[p].inverse, plans[p].inverse.data());
                for (const auto& group : plans[p].groups) {
                    auto tokens = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, static_cast<int64_t>(group.tokens.size()));
                    auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, group.frames * group.rows * group.cols);
                    set_backend_tensor_data(tokens, group.tokens.data());
                    set_backend_tensor_data(pe, group.pe.data());
                    windows[p].tokens.push_back(tokens);
                    windows[p].pe.push_back(pe);
                }
            }

            auto runner_ctx = get_context();
            ggml_build_forward_expand(gf, model.forward(&runner_ctx, x, context, timesteps, windows));
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& c_concat = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            GGML_ASSERT(diffusion_params.context != nullptr);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           *diffusion_params.context,
                           tensor_or_empty(diffusion_params.c_concat));
        }
    };

}  // namespace SeedVR2

#endif  // __SD_MODEL_DIFFUSION_SEEDVR2_HPP__
