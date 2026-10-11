#ifndef __SD_CORE_RNG_WANG_HPP__
#define __SD_CORE_RNG_WANG_HPP__

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "core/rng.hpp"
#include "ggml-wanghash.h"

// Host twin of the stateless WangHash noise in ggml's GGML_OP_SAMPLE_STEP. The integer part comes
// from ggml-wanghash.h, so noise for element i of sampler step k depends only on (seed, k, i) and the
// host sampler and the GPU-resident sampler draw identical noise.
namespace sd {
    namespace wang {

        inline uint32_t key(uint32_t seed, uint32_t step) {
            return ggml_wang_key(seed, step);
        }

        inline float normal(uint32_t step_key, uint32_t i) {
            float u1 = 0.f;
            float u2 = 0.f;
            ggml_wang_uniform2(step_key, i, &u1, &u2);
            return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
        }

        inline uint32_t seed_from_rng(const std::shared_ptr<RNG>& rng) {
            const std::vector<float> v = rng->randn(1);
            uint32_t bits              = 0;
            std::memcpy(&bits, v.data(), sizeof(bits));
            return ggml_wang_hash(bits);
        }

    }  // namespace wang
}  // namespace sd

#endif  // __SD_CORE_RNG_WANG_HPP__
