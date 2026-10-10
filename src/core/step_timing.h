#ifndef __SD_CORE_STEP_TIMING_H__
#define __SD_CORE_STEP_TIMING_H__

#include <cstddef>
#include <cstdint>

namespace sd {
    namespace step_timing {

        enum Bucket {
            HOST_INPUTS,
            GRAPH_BUILD,
            PLAN,
            ALLOC,
            UPLOAD,
            COMPUTE,
            CAPTURE,
            READBACK,
            DENOISE,
            SAMPLER,
            BUCKET_COUNT,
        };

        bool enabled();
        int64_t now();
        void add(Bucket bucket, int64_t us);
        void add_since(Bucket bucket, int64_t start_us);
        void add_upload_bytes(size_t bytes);
        void add_compute_call();
        void step_begin(int step);
        void step_end();
        void run_end();

        class Scope {
            Bucket bucket_;
            int64_t start_us_;

        public:
            explicit Scope(Bucket bucket);
            ~Scope();
            Scope(const Scope&)            = delete;
            Scope& operator=(const Scope&) = delete;
        };

    }  // namespace step_timing
}  // namespace sd

#endif  // __SD_CORE_STEP_TIMING_H__
