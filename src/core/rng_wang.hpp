#ifndef __SD_CORE_RNG_WANG_HPP__
#define __SD_CORE_RNG_WANG_HPP__

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "core/rng.hpp"

// Host twin of the stateless WangHash noise in ggml's GGML_OP_SAMPLE_STEP (ggml/src/ggml-wanghash.h).
// Noise for element i of sampler step k depends only on (seed, k, i), so the host sampler and the
// GPU-resident sampler draw identical noise. Keep the integer math in sync with ggml.
namespace sd {
    namespace wang {

        inline uint32_t hash(uint32_t s) {
            s = (s ^ 61u) ^ (s >> 16);
            s *= 9u;
            s ^= s >> 4;
            s *= 0x27d4eb2du;
            s ^= s >> 15;
            return s;
        }

        inline uint32_t key(uint32_t seed, uint32_t step) {
            return hash(seed ^ hash(step + 0x9e3779b9u));
        }

        inline float normal(uint32_t step_key, uint32_t i) {
            const uint32_t h1 = hash(step_key ^ (2u * i));
            const uint32_t h2 = hash(h1 ^ (2u * i + 1u));
            const float u1    = ((float)(h1 >> 8) + 0.5f) * (1.0f / 16777216.0f);
            const float u2    = (float)(h2 >> 8) * (1.0f / 16777216.0f);
            return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
        }

        inline uint32_t seed_from_rng(const std::shared_ptr<RNG>& rng) {
            const std::vector<float> v = rng->randn(1);
            uint32_t bits              = 0;
            std::memcpy(&bits, v.data(), sizeof(bits));
            return hash(bits);
        }

    }  // namespace wang
}  // namespace sd

#endif  // __SD_CORE_RNG_WANG_HPP__
