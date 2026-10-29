#ifndef __SD_MODEL_VAE_SEEDVR2_VAE_HPP__
#define __SD_MODEL_VAE_SEEDVR2_VAE_HPP__

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model/vae/vae.hpp"

namespace SeedVR2VAE {

    constexpr int SEEDVR2_VAE_GRAPH_SIZE_PER_FRAME = 8192;
    constexpr int SEEDVR2_VAE_TEMPORAL_RATIO       = 4;
    constexpr int64_t SEEDVR2_VAE_IM2COL_BUDGET    = 1LL << 30;

    // Causal state of one encode or decode graph. Every layer is causal in
    // time, so frames may be fed through any stage in pieces as long as each
    // layer sees them in order. Carries are expanded into the graph as soon as
    // they exist so the padded input they are sliced from is freed early.
    struct CausalState {
        std::unordered_map<const void*, ggml_tensor*> carries;
        std::unordered_set<const void*> started;
        bool keep_carries = true;
    };

    // Causal in time: the first call repeats frame 0 kernel_t - 1 times,
    // later calls prepend the last kernel_t - stride_t padded input frames of
    // the previous call. Space is zero padded.
    struct CausalConv3d : public Conv3d {
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride = {1, 1, 1},
                     int spatial_padding              = 0)
            : Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     {0, spatial_padding, spatial_padding}) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            const int kernel_t = std::get<0>(kernel_size);
            const int stride_t = std::get<0>(stride);
            if (kernel_t > 1) {
                auto carry = state->carries.find(this);
                if (carry != state->carries.end()) {
                    x = ggml_concat(ctx->ggml_ctx, carry->second, x, 2);
                } else {
                    auto head = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                    for (int i = 0; i < kernel_t - 1; ++i) {
                        x = ggml_concat(ctx->ggml_ctx, head, x, 2);
                    }
                }
                const int64_t keep = kernel_t - stride_t;
                if (keep > 0 && state->keep_carries) {
                    auto next = ggml_ext_slice(ctx->ggml_ctx, x, 2, x->ne[2] - keep, x->ne[2]);
                    ggml_build_forward_expand(ctx->graph, next);
                    state->carries[this] = next;
                }
            }
            return conv_in_parts(ctx, x);
        }

    protected:
        ggml_tensor* conv_unpadded(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = nullptr;
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                if (w->type != GGML_TYPE_F16) {
                    w = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F16);
                }
            }
            if (bias) {
                b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
            }
            return ggml_ext_conv_3d(ctx->ggml_ctx, ctx->backend, x, w, b, in_channels,
                                    std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                    0, 0, 0,
                                    std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                    force_prec_f32, ctx->conv3d_direct_enabled);
        }

        // im2col holds the receptive field of every output voxel, so output
        // frames, and rows of a single frame if needed, are convolved in parts
        // that keep it under the budget.
        ggml_tensor* conv_in_parts(GGMLRunnerContext* ctx, ggml_tensor* x) {
            const int64_t kernel_t        = std::get<0>(kernel_size);
            const int64_t kernel_h        = std::get<1>(kernel_size);
            const int64_t kernel_w        = std::get<2>(kernel_size);
            const int64_t stride_t        = std::get<0>(stride);
            const int64_t stride_h        = std::get<1>(stride);
            const int64_t pad_h           = std::get<1>(padding);
            const int64_t pad_w           = std::get<2>(padding);
            const int64_t out_w           = (x->ne[0] + 2 * pad_w - kernel_w) / std::get<2>(stride) + 1;
            const int64_t out_h           = (x->ne[1] + 2 * pad_h - kernel_h) / stride_h + 1;
            const int64_t out_frames      = (x->ne[2] - kernel_t) / stride_t + 1;
            const int64_t row_bytes       = out_w * in_channels * kernel_t * kernel_h * kernel_w *
                                            static_cast<int64_t>(sizeof(ggml_fp16_t));
            const int64_t frame_bytes     = row_bytes * out_h;
            const int64_t frames_per_part = std::max<int64_t>(1, SEEDVR2_VAE_IM2COL_BUDGET / frame_bytes);
            const int64_t rows_per_part   = std::min(out_h, std::max<int64_t>(1, SEEDVR2_VAE_IM2COL_BUDGET / row_bytes));
            if (frames_per_part >= out_frames && rows_per_part >= out_h) {
                return Conv3d::forward(ctx, x);
            }
            if (rows_per_part < out_h && (pad_h > 0 || pad_w > 0)) {
                x = ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, x, pad_w, pad_w, pad_h, pad_h, 0, 0, 0, 0);
            }
            std::vector<ggml_tensor*> frames;
            for (int64_t start = 0; start < out_frames; start += frames_per_part) {
                const int64_t end = std::min(start + frames_per_part, out_frames);
                auto input        = ggml_ext_slice(ctx->ggml_ctx, x, 2, start * stride_t, (end - 1) * stride_t + kernel_t);
                if (rows_per_part >= out_h) {
                    frames.push_back(Conv3d::forward(ctx, input));
                    continue;
                }
                std::vector<ggml_tensor*> rows;
                for (int64_t row = 0; row < out_h; row += rows_per_part) {
                    const int64_t row_end = std::min(row + rows_per_part, out_h);
                    auto band             = ggml_ext_slice(ctx->ggml_ctx, input, 1, row * stride_h, (row_end - 1) * stride_h + kernel_h);
                    rows.push_back(conv_unpadded(ctx, band));
                }
                frames.push_back(ggml_ext_vec_concat(ctx->ggml_ctx, rows, 1));
            }
            return ggml_ext_vec_concat(ctx->ggml_ctx, frames, 2);
        }
    };

    // GroupNorm(32, eps 1e-6) with statistics per frame; x is [W, H, T, C].
    // Frames are folded into the group axis so a single group_norm covers
    // every frame.
    class FrameGroupNorm : public GGMLBlock {
    protected:
        static constexpr int num_groups = 32;
        int64_t num_channels;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        }

    public:
        explicit FrameGroupNorm(int64_t num_channels)
            : num_channels(num_channels) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = params["bias"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t T = x->ne[2];
            const int64_t C = x->ne[3];

            auto y = T == 1 ? x : ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
            y      = ggml_reshape_3d(ctx->ggml_ctx, y, W, H, C * T);
            y      = ggml_group_norm(ctx->ggml_ctx, y, num_groups * static_cast<int>(T), 1e-6f);
            y      = ggml_reshape_4d(ctx->ggml_ctx, y, W, H, C, T);
            y      = ggml_mul_inplace(ctx->ggml_ctx, y, ggml_reshape_4d(ctx->ggml_ctx, w, 1, 1, C, 1));
            y      = ggml_add_inplace(ctx->ggml_ctx, y, ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, C, 1));
            if (T == 1) {
                return ggml_reshape_4d(ctx->ggml_ctx, y, W, H, 1, C);
            }
            return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, y, 0, 1, 3, 2));
        }
    };

    struct ResnetBlock3D : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;

        ResnetBlock3D(int64_t in_channels, int64_t out_channels)
            : in_channels(in_channels), out_channels(out_channels) {
            blocks["norm1"] = std::make_shared<FrameGroupNorm>(in_channels);
            blocks["conv1"] = std::make_shared<CausalConv3d>(in_channels,
                                                             out_channels,
                                                             std::tuple{3, 3, 3},
                                                             std::tuple{1, 1, 1},
                                                             1);
            blocks["norm2"] = std::make_shared<FrameGroupNorm>(out_channels);
            blocks["conv2"] = std::make_shared<CausalConv3d>(out_channels,
                                                             out_channels,
                                                             std::tuple{3, 3, 3},
                                                             std::tuple{1, 1, 1},
                                                             1);
            if (in_channels != out_channels) {
                blocks["conv_shortcut"] = std::make_shared<CausalConv3d>(in_channels,
                                                                         out_channels,
                                                                         std::tuple{1, 1, 1});
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            auto norm1 = std::dynamic_pointer_cast<FrameGroupNorm>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<FrameGroupNorm>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto h = ggml_silu(ctx->ggml_ctx, norm1->forward(ctx, x));
            h      = conv1->forward(ctx, h, state);
            h      = ggml_silu(ctx->ggml_ctx, norm2->forward(ctx, h));
            h      = conv2->forward(ctx, h, state);
            if (in_channels != out_channels) {
                x = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_shortcut"])->forward(ctx, x, state);
            }
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct Downsample3D : public GGMLBlock {
        Downsample3D(int64_t channels, bool temporal) {
            blocks["conv"] = std::make_shared<CausalConv3d>(channels,
                                                            channels,
                                                            std::tuple{temporal ? 3 : 1, 3, 3},
                                                            std::tuple{temporal ? 2 : 1, 2, 2});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            x = ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, x, 0, 1, 0, 1, 0, 0, 0, 0);
            return std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"])->forward(ctx, x, state);
        }
    };

    // upscale_conv emits channels ordered (x y z c) with c innermost, where
    // x and y split H and W and z splits time. The first temporal call drops
    // the duplicated frame 1.
    struct Upsample3D : public GGMLBlock {
        static constexpr int spatial_ratio = 2;
        int64_t channels;
        int temporal_ratio;

        Upsample3D(int64_t channels, bool temporal)
            : channels(channels), temporal_ratio(temporal ? 2 : 1) {
            blocks["upscale_conv"] = std::make_shared<Conv3d>(channels,
                                                              channels * spatial_ratio * spatial_ratio * temporal_ratio,
                                                              std::tuple{1, 1, 1});
            blocks["conv"]         = std::make_shared<CausalConv3d>(channels,
                                                                    channels,
                                                                    std::tuple{3, 3, 3},
                                                                    std::tuple{1, 1, 1},
                                                                    1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            auto upscale = std::dynamic_pointer_cast<Conv3d>(blocks["upscale_conv"]);
            auto conv    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);

            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t F = x->ne[2];
            const int64_t S = spatial_ratio;
            const int64_t Z = temporal_ratio;
            const int64_t C = channels;

            auto h = upscale->forward(ctx, x);
            h      = ggml_reshape_4d(ctx->ggml_ctx, h, W * H, F, C, Z * S * S);
            h      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 0, 2, 3, 1));
            h      = ggml_reshape_4d(ctx->ggml_ctx, h, W, H * Z, S, S * F * C);
            h      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 1, 2, 0, 3));
            h      = ggml_reshape_4d(ctx->ggml_ctx, h, S * W, H * Z, S, F * C);
            h      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 0, 2, 1, 3));
            h      = ggml_reshape_4d(ctx->ggml_ctx, h, S * W, S * H, Z * F, C);

            if (Z > 1 && state->started.insert(this).second) {
                auto first = ggml_ext_slice(ctx->ggml_ctx, h, 2, 0, 1);
                if (h->ne[2] > 2) {
                    h = ggml_concat(ctx->ggml_ctx, first, ggml_ext_slice(ctx->ggml_ctx, h, 2, 2, h->ne[2]), 2);
                } else {
                    h = first;
                }
            }
            return conv->forward(ctx, h, state);
        }
    };

    // Single head attention over the H*W tokens of each frame.
    struct MidAttention : public GGMLBlock {
        int64_t channels;

        explicit MidAttention(int64_t channels)
            : channels(channels) {
            blocks["group_norm"] = std::make_shared<FrameGroupNorm>(channels);
            blocks["to_q"]       = std::make_shared<Linear>(channels, channels, true);
            blocks["to_k"]       = std::make_shared<Linear>(channels, channels, true);
            blocks["to_v"]       = std::make_shared<Linear>(channels, channels, true);
            blocks["to_out.0"]   = std::make_shared<Linear>(channels, channels, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm   = std::dynamic_pointer_cast<FrameGroupNorm>(blocks["group_norm"]);
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t T = x->ne[2];

            auto h   = norm->forward(ctx, x);
            h        = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));
            h        = ggml_reshape_3d(ctx->ggml_ctx, h, channels, W * H, T);
            auto q   = to_q->forward(ctx, h);
            auto k   = to_k->forward(ctx, h);
            auto v   = to_v->forward(ctx, h);
            auto out = ggml_ext_attention_ext(ctx, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);
            out      = to_out->forward(ctx, out);
            out      = ggml_reshape_4d(ctx->ggml_ctx, out, channels, W, H, T);
            out      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 3, 0, 1, 2));
            return ggml_add(ctx->ggml_ctx, x, out);
        }
    };

    struct MidBlock : public GGMLBlock {
        explicit MidBlock(int64_t channels) {
            blocks["resnets.0"]    = std::make_shared<ResnetBlock3D>(channels, channels);
            blocks["attentions.0"] = std::make_shared<MidAttention>(channels);
            blocks["resnets.1"]    = std::make_shared<ResnetBlock3D>(channels, channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            x = std::dynamic_pointer_cast<ResnetBlock3D>(blocks["resnets.0"])->forward(ctx, x, state);
            x = std::dynamic_pointer_cast<MidAttention>(blocks["attentions.0"])->forward(ctx, x);
            return std::dynamic_pointer_cast<ResnetBlock3D>(blocks["resnets.1"])->forward(ctx, x, state);
        }
    };

    struct Encoder : public GGMLBlock {
        static constexpr int levels                             = 4;
        static constexpr std::array<int64_t, levels> channels   = {128, 256, 512, 512};
        static constexpr std::array<bool, levels> temporal_down = {false, true, true, false};
        static constexpr int64_t latent_channels                = 16;

        Encoder() {
            blocks["conv_in"] = std::make_shared<CausalConv3d>(3,
                                                               channels[0],
                                                               std::tuple{3, 3, 3},
                                                               std::tuple{1, 1, 1},
                                                               1);
            int64_t previous  = channels[0];
            for (int i = 0; i < levels; ++i) {
                const std::string prefix = "down_blocks." + std::to_string(i);
                for (int j = 0; j < 2; ++j) {
                    blocks[prefix + ".resnets." + std::to_string(j)] =
                        std::make_shared<ResnetBlock3D>(j == 0 ? previous : channels[i], channels[i]);
                }
                if (i < levels - 1) {
                    blocks[prefix + ".downsamplers.0"] = std::make_shared<Downsample3D>(channels[i], temporal_down[i]);
                }
                previous = channels[i];
            }
            blocks["mid_block"]     = std::make_shared<MidBlock>(previous);
            blocks["conv_norm_out"] = std::make_shared<FrameGroupNorm>(previous);
            blocks["conv_out"]      = std::make_shared<CausalConv3d>(previous,
                                                                     2 * latent_channels,
                                                                     std::tuple{3, 3, 3},
                                                                     std::tuple{1, 1, 1},
                                                                     1);
        }

        ggml_tensor* forward_block(GGMLRunnerContext* ctx, int i, ggml_tensor* x, CausalState* state) {
            const std::string prefix = "down_blocks." + std::to_string(i);
            for (int j = 0; j < 2; ++j) {
                x = std::dynamic_pointer_cast<ResnetBlock3D>(blocks[prefix + ".resnets." + std::to_string(j)])
                        ->forward(ctx, x, state);
            }
            if (i < levels - 1) {
                x = std::dynamic_pointer_cast<Downsample3D>(blocks[prefix + ".downsamplers.0"])->forward(ctx, x, state);
            }
            return x;
        }

        ggml_tensor* forward_full_res(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            x = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"])->forward(ctx, x, state);
            return forward_block(ctx, 0, x, state);
        }

        ggml_tensor* forward_low_res(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            for (int i = 1; i < levels; ++i) {
                x = forward_block(ctx, i, x, state);
            }
            x         = std::dynamic_pointer_cast<MidBlock>(blocks["mid_block"])->forward(ctx, x, state);
            auto norm = std::dynamic_pointer_cast<FrameGroupNorm>(blocks["conv_norm_out"]);
            x         = ggml_silu(ctx->ggml_ctx, norm->forward(ctx, x));
            x         = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"])->forward(ctx, x, state);
            return ggml_ext_slice(ctx->ggml_ctx, x, 3, 0, latent_channels);
        }

        // x is [W, H, 4n+1, 3]. Frame 0 forms the first chunk and every later
        // group of 4 frames yields one latent frame. The full resolution
        // stage, which ends with a time-preserving downsample, runs one frame
        // at a time.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            const int64_t frames = x->ne[2];
            CausalState state;
            std::vector<ggml_tensor*> outputs;
            for (int64_t start = 0; start < frames;) {
                const int64_t end = start == 0 ? 1 : std::min<int64_t>(start + SEEDVR2_VAE_TEMPORAL_RATIO, frames);
                std::vector<ggml_tensor*> parts;
                for (int64_t t = start; t < end; ++t) {
                    state.keep_carries = t + 1 < frames;
                    auto part          = forward_full_res(ctx, ggml_ext_slice(ctx->ggml_ctx, x, 2, t, t + 1), &state);
                    ggml_build_forward_expand(ctx->graph, part);
                    parts.push_back(part);
                }
                state.keep_carries = end < frames;
                auto latent        = forward_low_res(ctx, ggml_ext_vec_concat(ctx->ggml_ctx, parts, 2), &state);
                ggml_build_forward_expand(ctx->graph, latent);
                outputs.push_back(latent);
                start = end;
            }
            return ggml_ext_vec_concat(ctx->ggml_ctx, outputs, 2);
        }
    };

    struct Decoder : public GGMLBlock {
        static constexpr int levels                           = 4;
        static constexpr std::array<int64_t, levels> channels = {512, 512, 256, 128};
        static constexpr std::array<bool, levels> temporal_up = {true, true, false, false};
        static constexpr int64_t latent_channels              = 16;

        Decoder() {
            blocks["conv_in"]   = std::make_shared<CausalConv3d>(latent_channels,
                                                                 channels[0],
                                                                 std::tuple{3, 3, 3},
                                                                 std::tuple{1, 1, 1},
                                                                 1);
            blocks["mid_block"] = std::make_shared<MidBlock>(channels[0]);
            int64_t previous    = channels[0];
            for (int i = 0; i < levels; ++i) {
                const std::string prefix = "up_blocks." + std::to_string(i);
                for (int j = 0; j < 3; ++j) {
                    blocks[prefix + ".resnets." + std::to_string(j)] =
                        std::make_shared<ResnetBlock3D>(j == 0 ? previous : channels[i], channels[i]);
                }
                if (i < levels - 1) {
                    blocks[prefix + ".upsamplers.0"] = std::make_shared<Upsample3D>(channels[i], temporal_up[i]);
                }
                previous = channels[i];
            }
            blocks["conv_norm_out"] = std::make_shared<FrameGroupNorm>(previous);
            blocks["conv_out"]      = std::make_shared<CausalConv3d>(previous,
                                                                     3,
                                                                     std::tuple{3, 3, 3},
                                                                     std::tuple{1, 1, 1},
                                                                     1);
        }

        ggml_tensor* forward_block(GGMLRunnerContext* ctx, int i, ggml_tensor* x, CausalState* state) {
            const std::string prefix = "up_blocks." + std::to_string(i);
            for (int j = 0; j < 3; ++j) {
                x = std::dynamic_pointer_cast<ResnetBlock3D>(blocks[prefix + ".resnets." + std::to_string(j)])
                        ->forward(ctx, x, state);
            }
            if (i < levels - 1) {
                x = std::dynamic_pointer_cast<Upsample3D>(blocks[prefix + ".upsamplers.0"])->forward(ctx, x, state);
            }
            return x;
        }

        ggml_tensor* forward_low_res(GGMLRunnerContext* ctx, ggml_tensor* z, CausalState* state) {
            auto x = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"])->forward(ctx, z, state);
            x      = std::dynamic_pointer_cast<MidBlock>(blocks["mid_block"])->forward(ctx, x, state);
            x      = forward_block(ctx, 0, x, state);
            return forward_block(ctx, 1, x, state);
        }

        ggml_tensor* forward_full_res(GGMLRunnerContext* ctx, ggml_tensor* x, CausalState* state) {
            x         = forward_block(ctx, 2, x, state);
            x         = forward_block(ctx, 3, x, state);
            auto norm = std::dynamic_pointer_cast<FrameGroupNorm>(blocks["conv_norm_out"]);
            x         = ggml_silu(ctx->ggml_ctx, norm->forward(ctx, x));
            return std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"])->forward(ctx, x, state);
        }

        // z is [W, H, n+1, 16]. Each latent frame is one chunk through the
        // time-upsampling stages, which emit one frame for the first latent
        // and four for every later one. The stages past the last temporal
        // upsample run one frame at a time.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) {
            const int64_t latents = z->ne[2];
            CausalState state;
            std::vector<ggml_tensor*> outputs;
            for (int64_t t = 0; t < latents; ++t) {
                const bool last    = t + 1 == latents;
                state.keep_carries = !last;
                auto h             = forward_low_res(ctx, ggml_ext_slice(ctx->ggml_ctx, z, 2, t, t + 1), &state);
                ggml_build_forward_expand(ctx->graph, h);
                for (int64_t f = 0; f < h->ne[2]; ++f) {
                    state.keep_carries = !last || f + 1 < h->ne[2];
                    auto frame         = forward_full_res(ctx, ggml_ext_slice(ctx->ggml_ctx, h, 2, f, f + 1), &state);
                    ggml_build_forward_expand(ctx->graph, frame);
                    outputs.push_back(frame);
                }
            }
            return ggml_ext_vec_concat(ctx->ggml_ctx, outputs, 2);
        }
    };

    struct VideoAutoencoder : public GGMLBlock {
        VideoAutoencoder() {
            blocks["encoder"] = std::make_shared<Encoder>();
            blocks["decoder"] = std::make_shared<Decoder>();
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
            return std::dynamic_pointer_cast<Encoder>(blocks["encoder"])->forward(ctx, x);
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
            return std::dynamic_pointer_cast<Decoder>(blocks["decoder"])->forward(ctx, z);
        }
    };

    struct SeedVR2VAERunner : public VAE {
        static constexpr float scaling_factor = 0.9152f;
        VideoAutoencoder model;

        SeedVR2VAERunner(ggml_backend_t backend,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string& prefix,
                         SDVersion version,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : VAE(version, backend, prefix, weight_manager) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "seedvr2_vae";
        }

        int get_encoder_output_channels(int input_channels) override {
            SD_UNUSED(input_channels);
            return static_cast<int>(Encoder::latent_channels);
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            model.get_param_tensors(tensors, weight_prefix);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            return vae_output;
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents / scaling_factor;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return latents * scaling_factor;
        }

        static sd::Tensor<float> ensure_video_shape(const sd::Tensor<float>& tensor) {
            if (tensor.dim() == 5) {
                return tensor;
            }
            GGML_ASSERT(tensor.dim() == 4);
            return tensor.reshape({tensor.shape()[0],
                                   tensor.shape()[1],
                                   1,
                                   tensor.shape()[2],
                                   tensor.shape()[3]});
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            auto input           = ensure_video_shape(z);
            const int64_t frames = input.shape()[2];
            if (input.shape()[4] != 1) {
                LOG_ERROR("%s supports batch size 1 only", get_desc().c_str());
                return {};
            }
            if (!decode_graph && (frames - 1) % SEEDVR2_VAE_TEMPORAL_RATIO != 0) {
                LOG_ERROR("%s encode needs 4n+1 frames, got %lld", get_desc().c_str(), (long long)frames);
                return {};
            }
            const int64_t pixel_frames = decode_graph ? (frames - 1) * SEEDVR2_VAE_TEMPORAL_RATIO + 1 : frames;
            auto get_graph             = [&]() -> ggml_cgraph* {
                auto graph      = new_graph_custom(std::min<size_t>(MAX_GRAPH_SIZE, SEEDVR2_VAE_GRAPH_SIZE_PER_FRAME * (pixel_frames + 1)));
                auto value      = make_input(input);
                auto runner_ctx = get_context(graph);
                auto out        = decode_graph ? model.decode(&runner_ctx, value) : model.encode(&runner_ctx, value);
                ggml_build_forward_expand(graph, out);
                return graph;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), 5);
        }
    };

}  // namespace SeedVR2VAE

#endif  // __SD_MODEL_VAE_SEEDVR2_VAE_HPP__
