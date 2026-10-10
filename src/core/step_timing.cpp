#include "core/step_timing.h"

#include <array>
#include <cstdlib>

#include "core/util.h"
#include "ggml.h"

namespace sd {
    namespace step_timing {

        namespace {
            struct Counters {
                std::array<int64_t, BUCKET_COUNT> us{};
                size_t upload_bytes = 0;
                int compute_calls   = 0;
                int64_t wall_us     = 0;

                void accumulate(const Counters& other) {
                    for (int i = 0; i < BUCKET_COUNT; ++i) {
                        us[i] += other.us[i];
                    }
                    upload_bytes += other.upload_bytes;
                    compute_calls += other.compute_calls;
                    wall_us += other.wall_us;
                }
            };

            struct State {
                bool active            = false;
                bool step_open         = false;
                bool sampler_pending   = false;
                int step               = 0;
                int steps              = 0;
                int64_t step_start_us  = 0;
                int64_t denoise_start  = 0;
                int64_t denoise_exit   = 0;
                Counters current;
                Counters total;
            };

            State& state() {
                static State s;
                return s;
            }

            double ms(int64_t us) {
                return us / 1000.0;
            }

            void print(const char* label, int step, const Counters& c, int divisor) {
                const double d   = divisor > 0 ? (double)divisor : 1.0;
                int64_t runner = 0;
                for (int b = HOST_INPUTS; b <= READBACK; ++b) {
                    runner += c.us[b];
                }
                const int64_t host_denoise = c.us[DENOISE] - runner;
                LOG_INFO("%s %3d wall %9.2f ms | inputs %7.2f build %7.2f plan %7.2f alloc %7.2f upload %7.2f (%8.1f KiB) compute %9.2f capture %6.2f readback %6.2f host-denoise %7.2f sampler %7.2f | calls %.1f",
                         label,
                         step,
                         ms(c.wall_us) / d,
                         ms(c.us[HOST_INPUTS]) / d,
                         ms(c.us[GRAPH_BUILD]) / d,
                         ms(c.us[PLAN]) / d,
                         ms(c.us[ALLOC]) / d,
                         ms(c.us[UPLOAD]) / d,
                         c.upload_bytes / 1024.0 / d,
                         ms(c.us[COMPUTE]) / d,
                         ms(c.us[CAPTURE]) / d,
                         ms(c.us[READBACK]) / d,
                         ms(host_denoise) / d,
                         ms(c.us[SAMPLER]) / d,
                         c.compute_calls / d);
            }

            void close_step(State& s, int64_t now) {
                if (!s.step_open) {
                    return;
                }
                s.current.wall_us = now - s.step_start_us;
                print("step", s.step, s.current, 1);
                s.total.accumulate(s.current);
                s.steps++;
                s.current   = Counters();
                s.step_open = false;
            }
        }  // namespace

        bool enabled() {
            static const bool on = [] {
                const char* env = std::getenv("SD_STEP_TIMING");
                return env != nullptr && env[0] != '\0' && env[0] != '0';
            }();
            return on;
        }

        int64_t now() {
            return enabled() ? ggml_time_us() : 0;
        }

        void add_since(Bucket bucket, int64_t start_us) {
            if (start_us != 0) {
                add(bucket, ggml_time_us() - start_us);
            }
        }

        void add(Bucket bucket, int64_t us) {
            State& s = state();
            if (s.active) {
                s.current.us[bucket] += us;
            }
        }

        void add_upload_bytes(size_t bytes) {
            State& s = state();
            if (s.active) {
                s.current.upload_bytes += bytes;
            }
        }

        void add_compute_call() {
            State& s = state();
            if (s.active) {
                s.current.compute_calls++;
            }
        }

        void step_begin(int step) {
            if (!enabled()) {
                return;
            }
            State& s          = state();
            const int64_t now = ggml_time_us();
            if (!s.active) {
                s        = State();
                s.active = true;
            }
            if (s.sampler_pending) {
                s.current.us[SAMPLER] += now - s.denoise_exit;
                s.sampler_pending = false;
            }
            close_step(s, now);
            s.step          = step;
            s.step_open     = true;
            s.step_start_us = now;
            s.denoise_start = now;
        }

        void step_end() {
            if (!enabled()) {
                return;
            }
            State& s = state();
            if (!s.active) {
                return;
            }
            const int64_t now = ggml_time_us();
            s.current.us[DENOISE] += now - s.denoise_start;
            s.denoise_exit    = now;
            s.sampler_pending = true;
        }

        void run_end() {
            if (!enabled()) {
                return;
            }
            State& s = state();
            if (!s.active) {
                return;
            }
            const int64_t now = ggml_time_us();
            if (s.sampler_pending) {
                s.current.us[SAMPLER] += now - s.denoise_exit;
                s.sampler_pending = false;
            }
            close_step(s, now);
            print("total", s.steps, s.total, 1);
            print("mean ", s.steps, s.total, s.steps);
            s.active = false;
        }

        Scope::Scope(Bucket bucket)
            : bucket_(bucket), start_us_(now()) {
        }

        Scope::~Scope() {
            add_since(bucket_, start_us_);
        }

    }  // namespace step_timing
}  // namespace sd
