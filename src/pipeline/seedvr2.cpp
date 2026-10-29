#include "seedvr2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "conditioning/conditioner.hpp"
#include "core/util.h"
#include "diffusion_engine.h"
#include "model/diffusion/model.hpp"

namespace sd::pipeline {

    namespace {

        constexpr float SEEDVR2_TIMESTEP         = 1000.f;
        constexpr float SEEDVR2_LUMINANCE_WEIGHT = 0.8f;

        struct RestoreOptions {
            int64_t window     = 21;
            int64_t overlap    = 4;
            int64_t resolution = 0;
            std::string color  = "lab";
        };

        bool parse_restore_options(const char* args, RestoreOptions& options) {
            for (const auto& [key, value] : parse_key_value_args(args, "seedvr2 arg")) {
                if (key == "window") {
                    options.window = std::atoll(value.c_str());
                } else if (key == "overlap") {
                    options.overlap = std::atoll(value.c_str());
                } else if (key == "resolution") {
                    options.resolution = std::atoll(value.c_str());
                } else if (key == "color") {
                    options.color = value;
                } else {
                    LOG_WARN("ignoring unknown SeedVR2 arg '%s'", key.c_str());
                }
            }
            if (options.window < 1 || (options.window - 1) % 4 != 0) {
                LOG_ERROR("SeedVR2 window must be 4n+1 frames, got %lld", (long long)options.window);
                return false;
            }
            if (options.overlap < 0 || options.resolution < 0) {
                LOG_ERROR("SeedVR2 overlap and resolution must be non-negative");
                return false;
            }
            if (options.overlap >= options.window) {
                options.overlap = 0;
            }
            if (options.overlap > options.window - options.overlap) {
                LOG_ERROR("SeedVR2 overlap must be at most half the window, got %lld of %lld",
                          (long long)options.overlap, (long long)options.window);
                return false;
            }
            if (options.color != "lab" && options.color != "wavelet" && options.color != "adain" && options.color != "none") {
                LOG_ERROR("SeedVR2 color must be lab, wavelet, adain or none, got '%s'", options.color.c_str());
                return false;
            }
            return true;
        }

        int64_t frame_width(const sd::Tensor<float>& clip) {
            return clip.shape()[0];
        }

        int64_t frame_height(const sd::Tensor<float>& clip) {
            return clip.shape()[1];
        }

        int64_t frame_count(const sd::Tensor<float>& clip) {
            return clip.shape()[2];
        }

        // Frames [start, start + count) as a [W, H, T, 3, 1] clip in [0, 1].
        sd::Tensor<float> frames_to_clip(const sd_image_t* frames, int64_t start, int64_t count) {
            const int64_t W = frames[start].width;
            const int64_t H = frames[start].height;
            sd::Tensor<float> clip({W, H, count, 3, 1});
            float* out = clip.data();
            for (int64_t t = 0; t < count; ++t) {
                const sd_image_t& image = frames[start + t];
                const int64_t channels  = image.channel;
                for (int64_t c = 0; c < 3; ++c) {
                    const int64_t source = std::min<int64_t>(c, channels - 1);
                    float* plane         = out + (c * count + t) * H * W;
                    for (int64_t i = 0; i < H * W; ++i) {
                        plane[i] = image.data[i * channels + source] / 255.f;
                    }
                }
            }
            return clip;
        }

        sd::Tensor<float> gather_frames(const sd::Tensor<float>& clip, const std::vector<int64_t>& order) {
            const int64_t W = frame_width(clip);
            const int64_t H = frame_height(clip);
            const int64_t T = frame_count(clip);
            const int64_t C = clip.shape()[3];
            const int64_t N = static_cast<int64_t>(order.size());
            sd::Tensor<float> out({W, H, N, C, 1});
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t t = 0; t < N; ++t) {
                    std::memcpy(out.data() + (c * N + t) * H * W,
                                clip.data() + (c * T + order[t]) * H * W,
                                sizeof(float) * H * W);
                }
            }
            return out;
        }

        // Pads to 4n+1 frames the way the reference does: a reflection that
        // skips the last frame, then repeats of the last frame if that is not
        // enough.
        sd::Tensor<float> pad_frames_4n1(const sd::Tensor<float>& clip) {
            const int64_t t = frame_count(clip);
            if (t % 4 == 1) {
                return clip;
            }
            const int64_t count = ((t - 1) / 4 + 1) * 4 + 1 - t;
            std::vector<int64_t> order(t);
            std::iota(order.begin(), order.end(), 0);
            if (count >= t) {
                for (int64_t i = t - 1; i >= 1; --i) {
                    order.push_back(i);
                }
                for (int64_t i = 0; i < count - t + 1; ++i) {
                    order.push_back(t - 1);
                }
            } else {
                for (int64_t i = t - 2; i >= t - 1 - count; --i) {
                    order.push_back(i);
                }
            }
            return gather_frames(clip, order);
        }

        sd::Tensor<float> resize_clip(const sd::Tensor<float>& clip, int64_t W, int64_t H) {
            if (frame_width(clip) == W && frame_height(clip) == H) {
                return clip;
            }
            auto resized = sd::ops::interpolate(clip,
                                                std::optional<std::vector<int64_t>>(std::vector<int64_t>{W, H}),
                                                std::optional<std::vector<double>>(),
                                                sd::ops::InterpolateMode::Bicubic,
                                                false,
                                                true);
            return sd::ops::clamp(resized, 0.f, 1.f);
        }

        // Copies the top-left W x H of every frame into a clip of Wp x Hp
        // filled with fill.
        sd::Tensor<float> reframe_clip(const sd::Tensor<float>& clip, int64_t W, int64_t H, int64_t Wp, int64_t Hp, float fill) {
            const int64_t Wi = frame_width(clip);
            const int64_t Hi = frame_height(clip);
            const int64_t T  = frame_count(clip);
            const int64_t C  = clip.shape()[3];
            sd::Tensor<float> out({Wp, Hp, T, C, 1});
            std::fill(out.data(), out.data() + out.numel(), fill);
            for (int64_t p = 0; p < T * C; ++p) {
                for (int64_t y = 0; y < H; ++y) {
                    std::memcpy(out.data() + (p * Hp + y) * Wp,
                                clip.data() + (p * Hi + y) * Wi,
                                sizeof(float) * W);
                }
            }
            return out;
        }

        // Separable [1 2 1] / 4 blur at the given dilation with replicated
        // borders, matching the reference 3x3 binomial kernel.
        void wavelet_blur(const float* src, float* dst, std::vector<float>& scratch, int64_t W, int64_t H, int64_t radius) {
            scratch.resize(static_cast<size_t>(W * H));
            for (int64_t y = 0; y < H; ++y) {
                const float* row = src + y * W;
                for (int64_t x = 0; x < W; ++x) {
                    const float left   = row[std::max<int64_t>(x - radius, 0)];
                    const float right  = row[std::min<int64_t>(x + radius, W - 1)];
                    scratch[y * W + x] = 0.25f * left + 0.5f * row[x] + 0.25f * right;
                }
            }
            for (int64_t y = 0; y < H; ++y) {
                const float* up   = scratch.data() + std::max<int64_t>(y - radius, 0) * W;
                const float* mid  = scratch.data() + y * W;
                const float* down = scratch.data() + std::min<int64_t>(y + radius, H - 1) * W;
                for (int64_t x = 0; x < W; ++x) {
                    dst[y * W + x] = 0.25f * up[x] + 0.5f * mid[x] + 0.25f * down[x];
                }
            }
        }

        // Low frequency band after five blurs at dilations 1, 2, 4, 8 and 16.
        sd::Tensor<float> wavelet_low(const sd::Tensor<float>& clip) {
            const int64_t W          = frame_width(clip);
            const int64_t H          = frame_height(clip);
            const int64_t planes     = clip.numel() / (W * H);
            const int64_t max_radius = std::max<int64_t>(1, std::min(W, H) / 8);
            sd::Tensor<float> low    = clip;
            std::vector<float> scratch;
            std::vector<float> next(static_cast<size_t>(W * H));
            for (int64_t p = 0; p < planes; ++p) {
                float* plane = low.data() + p * W * H;
                for (int level = 0; level < 5; ++level) {
                    wavelet_blur(plane, next.data(), scratch, W, H, std::min<int64_t>(int64_t(1) << level, max_radius));
                    std::memcpy(plane, next.data(), sizeof(float) * W * H);
                }
            }
            return low;
        }

        sd::Tensor<float> wavelet_reconstruction(const sd::Tensor<float>& content, const sd::Tensor<float>& style) {
            auto content_low = wavelet_low(content);
            auto style_low   = wavelet_low(style);
            sd::Tensor<float> out(content.shape());
            for (int64_t i = 0; i < out.numel(); ++i) {
                out[i] = std::clamp(content[i] - content_low[i] + style_low[i], -1.f, 1.f);
            }
            return out;
        }

        constexpr float LAB_EPSILON = 6.f / 29.f;
        constexpr float LAB_KAPPA   = (29.f / 3.f) * (29.f / 3.f) * (29.f / 3.f);

        float lab_f(float t) {
            return t > LAB_EPSILON * LAB_EPSILON * LAB_EPSILON ? std::cbrt(t) : (LAB_KAPPA * t + 16.f) / 116.f;
        }

        float lab_f_inverse(float f) {
            return f > LAB_EPSILON ? f * f * f : (116.f * f - 16.f) / LAB_KAPPA;
        }

        // [W, H, T, 3] RGB in [-1, 1] to three planes of L, a and b.
        std::vector<float> rgb_to_lab(const sd::Tensor<float>& rgb) {
            const int64_t n = rgb.numel() / 3;
            std::vector<float> lab(static_cast<size_t>(3 * n));
            for (int64_t i = 0; i < n; ++i) {
                float linear[3];
                for (int c = 0; c < 3; ++c) {
                    const float v = std::clamp((rgb[c * n + i] + 1.f) * 0.5f, 0.f, 1.f);
                    linear[c]     = v > 0.04045f ? std::pow((v + 0.055f) / 1.055f, 2.4f) : v / 12.92f;
                }
                const float x  = (0.4124564f * linear[0] + 0.3575761f * linear[1] + 0.1804375f * linear[2]) / 0.95047f;
                const float y  = 0.2126729f * linear[0] + 0.7151522f * linear[1] + 0.0721750f * linear[2];
                const float z  = (0.0193339f * linear[0] + 0.1191920f * linear[1] + 0.9503041f * linear[2]) / 1.08883f;
                const float fx = lab_f(x);
                const float fy = lab_f(y);
                const float fz = lab_f(z);
                lab[i]         = 116.f * fy - 16.f;
                lab[n + i]     = 500.f * (fx - fy);
                lab[2 * n + i] = 200.f * (fy - fz);
            }
            return lab;
        }

        sd::Tensor<float> lab_to_rgb(const std::vector<float>& lab, const std::vector<int64_t>& shape) {
            sd::Tensor<float> rgb(shape);
            const int64_t n = rgb.numel() / 3;
            for (int64_t i = 0; i < n; ++i) {
                const float fy        = (lab[i] + 16.f) / 116.f;
                const float fx        = lab[n + i] / 500.f + fy;
                const float fz        = fy - lab[2 * n + i] / 200.f;
                const float x         = lab_f_inverse(fx) * 0.95047f;
                const float y         = lab_f_inverse(fy);
                const float z         = lab_f_inverse(fz) * 1.08883f;
                const float linear[3] = {3.2404542f * x - 1.5371385f * y - 0.4985314f * z,
                                         -0.9692660f * x + 1.8760108f * y + 0.0415560f * z,
                                         0.0556434f * x - 0.2040259f * y + 1.0572252f * z};
                for (int c = 0; c < 3; ++c) {
                    const float v  = linear[c] > 0.0031308f ? 1.055f * std::pow(std::max(linear[c], 0.f), 1.f / 2.4f) - 0.055f
                                                            : 12.92f * linear[c];
                    rgb[c * n + i] = std::clamp(v, 0.f, 1.f) * 2.f - 1.f;
                }
            }
            return rgb;
        }

        // Exact quantile mapping of source onto reference, both n values long.
        void histogram_match(float* source, const float* reference, int64_t n) {
            std::vector<int32_t> order(static_cast<size_t>(n));
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [source](int32_t a, int32_t b) { return source[a] < source[b]; });
            std::vector<float> sorted(reference, reference + n);
            std::sort(sorted.begin(), sorted.end());
            for (int64_t k = 0; k < n; ++k) {
                source[order[k]] = sorted[k];
            }
        }

        sd::Tensor<float> lab_color_transfer(const sd::Tensor<float>& content, const sd::Tensor<float>& style) {
            const auto filtered = wavelet_reconstruction(content, style);
            auto lab            = rgb_to_lab(filtered);
            const auto ref      = rgb_to_lab(style);
            const int64_t n     = static_cast<int64_t>(lab.size() / 3);
            histogram_match(lab.data() + n, ref.data() + n, n);
            histogram_match(lab.data() + 2 * n, ref.data() + 2 * n, n);
            std::vector<float> matched_l(lab.begin(), lab.begin() + n);
            histogram_match(matched_l.data(), ref.data(), n);
            for (int64_t i = 0; i < n; ++i) {
                lab[i] = SEEDVR2_LUMINANCE_WEIGHT * lab[i] + (1.f - SEEDVR2_LUMINANCE_WEIGHT) * matched_l[i];
            }
            return lab_to_rgb(lab, filtered.shape());
        }

        // Per frame and channel mean and unbiased std transfer.
        sd::Tensor<float> adain_color_transfer(const sd::Tensor<float>& content, const sd::Tensor<float>& style) {
            const int64_t plane = frame_width(content) * frame_height(content);
            const int64_t count = content.numel() / plane;
            sd::Tensor<float> out(content.shape());
            auto stats = [plane](const float* p) {
                double sum = 0.0, sum_sq = 0.0;
                for (int64_t i = 0; i < plane; ++i) {
                    sum += p[i];
                    sum_sq += static_cast<double>(p[i]) * p[i];
                }
                const double mean = sum / plane;
                const double var  = plane > 1 ? (sum_sq - sum * mean) / (plane - 1) : 0.0;
                return std::pair<double, double>(mean, std::sqrt(std::max(var, 0.0) + 1e-5));
            };
            for (int64_t p = 0; p < count; ++p) {
                const float* c      = content.data() + p * plane;
                const auto [cm, cs] = stats(c);
                const auto [sm, ss] = stats(style.data() + p * plane);
                float* o            = out.data() + p * plane;
                for (int64_t i = 0; i < plane; ++i) {
                    o[i] = static_cast<float>((c[i] - cm) / cs * ss + sm);
                }
            }
            return out;
        }

        sd::Tensor<float> color_correct(const sd::Tensor<float>& content, const sd::Tensor<float>& style, const std::string& mode) {
            if (mode == "lab") {
                return lab_color_transfer(content, style);
            }
            if (mode == "wavelet") {
                return wavelet_reconstruction(content, style);
            }
            if (mode == "adain") {
                return adain_color_transfer(content, style);
            }
            return content;
        }

        float torch_linspace_point(int64_t i, int64_t steps) {
            if (steps == 1) {
                return 0.f;
            }
            const float step = 1.f / static_cast<float>(steps - 1);
            return i < steps / 2 ? step * i : 1.f - step * (steps - 1 - i);
        }

        // Blends the first overlap frames of head into the last overlap frames
        // of tail with the reference Hann ramp, which keeps the previous
        // window for the first third and switches over by two thirds.
        void blend_overlap(sd::Tensor<float>& tail, const sd::Tensor<float>& head, int64_t overlap) {
            const int64_t plane = frame_width(tail) * frame_height(tail);
            const int64_t Tt    = frame_count(tail);
            const int64_t Th    = frame_count(head);
            for (int64_t k = 0; k < overlap; ++k) {
                float w_prev;
                if (overlap >= 3) {
                    const float t = torch_linspace_point(k, overlap);
                    const float u = std::clamp((t - 1.f / 3.f) * 3.f, 0.f, 1.f);
                    w_prev        = 0.5f + 0.5f * std::cos(static_cast<float>(M_PI) * u);
                } else {
                    w_prev = overlap == 1 ? 1.f : 1.f - static_cast<float>(k) / static_cast<float>(overlap - 1);
                }
                for (int64_t c = 0; c < 3; ++c) {
                    float* dst       = tail.data() + (c * Tt + Tt - overlap + k) * plane;
                    const float* src = head.data() + (c * Th + k) * plane;
                    for (int64_t i = 0; i < plane; ++i) {
                        dst[i] = dst[i] * w_prev + src[i] * (1.f - w_prev);
                    }
                }
            }
        }

        sd_audio_t* copy_audio(const sd_audio_t& audio) {
            if (audio.data == nullptr || audio.sample_count == 0 || audio.channels == 0) {
                return nullptr;
            }
            sd_audio_t* copy = (sd_audio_t*)malloc(sizeof(sd_audio_t));
            if (copy == nullptr) {
                return nullptr;
            }
            *copy             = audio;
            const size_t size = sizeof(float) * audio.sample_count * audio.channels;
            copy->data        = (float*)malloc(size);
            if (copy->data == nullptr) {
                free(copy);
                return nullptr;
            }
            std::memcpy(copy->data, audio.data, size);
            return copy;
        }

    }  // namespace

    bool restore_seedvr2_video(StableDiffusionGGML* sd,
                               const sd_vid_gen_params_t* sd_vid_gen_params,
                               sd_image_t** frames_out,
                               int* num_frames_out,
                               sd_audio_t** audio_out,
                               int* fps_out) {
        if (sd_vid_gen_params->ref_videos_count < 1 || sd_vid_gen_params->ref_videos[0].frame_count < 1) {
            LOG_ERROR("SeedVR2 restores the frames given with --ref-video");
            return false;
        }
        const sd_ref_video_t& source = sd_vid_gen_params->ref_videos[0];
        const int64_t total          = source.frame_count;
        const int64_t in_w           = source.frames[0].width;
        const int64_t in_h           = source.frames[0].height;
        for (int64_t i = 0; i < total; ++i) {
            if (source.frames[i].width != in_w || source.frames[i].height != in_h || source.frames[i].data == nullptr) {
                LOG_ERROR("SeedVR2 input frames must all be %lldx%lld", (long long)in_w, (long long)in_h);
                return false;
            }
        }

        RestoreOptions options;
        if (!parse_restore_options(sd_vid_gen_params->sample_params.extra_sample_args, options)) {
            return false;
        }

        int64_t width  = in_w;
        int64_t height = in_h;
        if (options.resolution > 0) {
            const int64_t short_side = std::min(in_w, in_h);
            const int64_t long_side  = std::max(in_w, in_h) * options.resolution / short_side;
            width                    = in_w <= in_h ? options.resolution : long_side;
            height                   = in_w <= in_h ? long_side : options.resolution;
        }
        const int64_t out_w    = width / 2 * 2;
        const int64_t out_h    = height / 2 * 2;
        const int64_t padded_w = (width + 15) / 16 * 16;
        const int64_t padded_h = (height + 15) / 16 * 16;
        const int64_t step     = options.window - options.overlap;

        std::vector<std::pair<int64_t, int64_t>> windows;
        for (int64_t start = 0; start < total; start += step) {
            const int64_t end = std::min(start + options.window, total);
            if (start > 0 && end - start <= options.overlap) {
                break;
            }
            windows.push_back({start, end});
        }
        LOG_INFO("SeedVR2 restore %lld frames %lldx%lld -> %lldx%lld, %zu windows of %lld, overlap %lld, color %s",
                 (long long)total, (long long)in_w, (long long)in_h, (long long)out_w, (long long)out_h,
                 windows.size(), (long long)options.window, (long long)options.overlap, options.color.c_str());

        const int64_t seed = sd_vid_gen_params->seed >= 0 ? sd_vid_gen_params->seed : static_cast<int64_t>(std::random_device{}());
        if (!sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count)) {
            return false;
        }
        const SDCondition condition = sd->cond_stage_model->get_learned_condition(sd->n_threads, ConditionerParams{});
        const sd::Tensor<float> timesteps({1}, std::vector<float>{SEEDVR2_TIMESTEP});

        auto prepare = [&](int64_t start, int64_t count) {
            return resize_clip(frames_to_clip(source.frames, start, count), width, height);
        };

        std::vector<sd_image_t> restored;
        restored.reserve(static_cast<size_t>(total));
        auto emit = [&](const sd::Tensor<float>& slice, int64_t ref_start) {
            auto reference = reframe_clip(prepare(ref_start, frame_count(slice)), out_w, out_h, out_w, out_h, 0.f) * 2.f - 1.f;
            auto corrected = color_correct(slice, reference, options.color);
            auto pixels    = sd::ops::clamp((corrected + 1.f) * 0.5f, 0.f, 1.f);
            for (int64_t t = 0; t < frame_count(pixels); ++t) {
                restored.push_back(tensor_to_sd_image(pixels, static_cast<int>(t)));
            }
        };

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> pending;
        int64_t pending_start = 0;
        for (size_t w = 0; w < windows.size(); ++w) {
            if (sd->get_cancel_flag() == SD_CANCEL_ALL) {
                LOG_WARN("SeedVR2 restore cancelled");
                for (auto& image : restored) {
                    free(image.data);
                }
                return false;
            }
            const auto [start, end] = windows[w];
            const int64_t length    = end - start;
            LOG_INFO("SeedVR2 window %zu/%zu, frames [%lld, %lld)", w + 1, windows.size(), (long long)start, (long long)end);

            auto clip   = pad_frames_4n1(reframe_clip(prepare(start, length), width, height, padded_w, padded_h, 0.f));
            auto latent = sd->encode_first_stage(clip);
            if (latent.empty()) {
                LOG_ERROR("SeedVR2 VAE encode failed");
                return false;
            }
            sd->rng->manual_seed(seed);
            auto noise  = sd::Tensor<float>::randn_like(latent, sd->rng);
            auto shape  = latent.shape();
            shape[3]    = 1;
            auto concat = sd::ops::concat(latent, sd::Tensor<float>::ones(shape), 3);
            DiffusionParams params;
            params.x         = &noise;
            params.timesteps = &timesteps;
            params.context   = &condition.c_crossattn;
            params.c_concat  = &concat;
            auto velocity    = sd->diffusion_model->compute(sd->n_threads, params);
            sd->diffusion_model->runner_end();
            if (velocity.empty()) {
                LOG_ERROR("SeedVR2 DiT failed");
                return false;
            }
            auto decoded = sd->decode_first_stage(noise - velocity, true);
            if (decoded.empty()) {
                LOG_ERROR("SeedVR2 VAE decode failed");
                return false;
            }
            decoded = reframe_clip(sd::ops::slice(decoded, 2, 0, length), out_w, out_h, out_w, out_h, 0.f) * 2.f - 1.f;

            if (w == 0) {
                pending       = std::move(decoded);
                pending_start = start;
                continue;
            }
            blend_overlap(pending, decoded, options.overlap);
            emit(pending, pending_start);
            pending       = sd::ops::slice(decoded, 2, options.overlap, length);
            pending_start = start + options.overlap;
        }
        emit(pending, pending_start);
        LOG_INFO("SeedVR2 restore completed in %.2fs", (ggml_time_ms() - t0) / 1000.f);

        sd_image_t* result = (sd_image_t*)calloc(restored.size(), sizeof(sd_image_t));
        if (result == nullptr) {
            for (auto& image : restored) {
                free(image.data);
            }
            return false;
        }
        std::copy(restored.begin(), restored.end(), result);
        if (frames_out != nullptr) {
            *frames_out = result;
        } else {
            for (auto& image : restored) {
                free(image.data);
            }
            free(result);
        }
        if (num_frames_out != nullptr) {
            *num_frames_out = static_cast<int>(restored.size());
        }
        if (fps_out != nullptr) {
            *fps_out = sd_vid_gen_params->fps > 0 ? sd_vid_gen_params->fps : source.fps;
        }
        if (audio_out != nullptr) {
            *audio_out = copy_audio(source.audio);
        }
        return true;
    }

}  // namespace sd::pipeline
