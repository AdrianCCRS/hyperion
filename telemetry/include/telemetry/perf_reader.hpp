#pragma once
#include "metrics.hpp"
#include <linux/perf_event.h>
#include <sys/types.h>
#include <vector>
#include <cstdint>

namespace telemetry {
    namespace detail {
        /**
         * @brief Scale a perf counter when the PMU was multiplexed.
         *
         * Perf reports raw counts with time_enabled/time_running. When the event
         * did not run for the full enabled window, this helper applies the
         * standard linear scale factor. A zero running time returns zero.
         */
        uint64_t scale_perf_count(uint64_t value,
                                  uint64_t time_enabled,
                                  uint64_t time_running) noexcept;
    }

    /**
     * @brief perf_event reader attached to an external process by PID.
     *
     * This is the CPU counter backend used by telemetry_kernel_launcher: it
     * opens hardware counters on a target PID (typically a freshly forked and
     * still-stopped child) with inherit=1, so descendant processes spawned by
     * the measured workload are covered too. inherit=1 forces the kernel to
     * reject grouped reads (PERF_FORMAT_GROUP), so each event is opened as an
     * independent file descriptor and read separately; this introduces a
     * microsecond-scale skew between events that is negligible against
     * millisecond sampling intervals.
     */
    class PerfReader {
        public:
        /**
         * @param pid Process to monitor. pid=0 means the calling process.
         * @param cpu CPU to monitor. cpu=-1 lets perf choose any CPU allowed by
         * the selected pid scope.
         *
         * inherit=1 is always set: descendants of pid (if any) are covered by
         * the same counters. Reading remains live per-fd while pid is alive;
         * inherited children are only folded back on their own exit, which is
         * irrelevant here because pid itself is the one being sampled.
         */
        explicit PerfReader(pid_t pid = 0, int cpu = -1);
        ~PerfReader();

        /**
         * @brief Open one independent fd per hardware counter and enable them.
         *
         * Throws std::runtime_error on permission or perf_event_open failures.
         */
        void open();

        /** @brief Disable and close every perf file descriptor. */
        void close() noexcept;

        /**
         * @brief Read every counter into a CpuSample.
         *
         * Returns false if the reader is closed or the kernel returns an
         * unexpected short/invalid read for any event.
         */
        bool read(CpuSample& out) noexcept;

        /** @brief Enable every counter fd. */
        void enable()  noexcept;

        /** @brief Disable every counter fd. */
        void disable() noexcept;

        /** @return true while the reader has open perf file descriptors. */
        bool is_open() const noexcept { return !fds_.empty(); }

        private:
            // Fixed open order, also the read/index order: instructions, cycles,
            // cache references, cache misses.
            static constexpr size_t kInstructions = 0;
            static constexpr size_t kCycles = 1;
            static constexpr size_t kCacheReferences = 2;
            static constexpr size_t kCacheMisses = 3;
            static constexpr size_t kEventCount = 4;

            pid_t pid_;
            int   cpu_;
            std::vector<int> fds_;

            /** Kernel read layout for one ungrouped fd with time diagnostics. */
            struct ReadFormat {
                uint64_t value;
                uint64_t time_enabled;
                uint64_t time_running;
            };
        };
}
