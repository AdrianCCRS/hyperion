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
     * @brief Simple process/CPU perf_event reader.
     *
     * This reader measures one process/CPU scope and is mainly useful for local
     * tests, smoke checks, and simple single-threaded use. The multithreaded
     * launcher uses PerfCgroupReader instead.
     */
    class PerfReader {
        public:
        /**
         * @param pid Process to monitor. pid=0 means the calling process.
         * @param cpu CPU to monitor. cpu=-1 lets perf choose any CPU allowed by
         * the selected pid scope.
         *
         * The implementation uses inherit=false to keep interpretation simple:
         * this class does not try to measure child processes or thread trees.
         */
        explicit PerfReader(pid_t pid = 0, int cpu = -1);
        ~PerfReader();
        
        /**
         * @brief Open and enable the hardware-counter group.
         *
         * Throws std::runtime_error on permission or perf_event_open failures.
         */
        void open();

        /** @brief Disable and close every perf file descriptor. */
        void close() noexcept;
    
        /**
         * @brief Read the whole perf group into a CpuSample.
         *
         * Returns false if the reader is closed or the kernel returns an
         * unexpected short/invalid read.
         */
        bool read(CpuSample& out) noexcept;
    
        /** @brief Enable the perf group. */
        void enable()  noexcept;

        /** @brief Disable the perf group. */
        void disable() noexcept;
    
        /** @return true when the leader file descriptor is open. */
        bool is_open() const noexcept { return group_fd_ >= 0; }

        private:
            static constexpr uint64_t kExpectedCounters = 4;
            static constexpr uint64_t kMaxReadCounters = 8;

            pid_t pid_;
            int   cpu_;
            int   group_fd_ = -1;   // leader fd
            std::vector<int> member_fds_;
        
            /**
             * @brief Kernel read layout for grouped perf reads.
             *
             * Event ids are not requested because the open order is fixed:
             * instructions, cycles, cache references, cache misses.
             */
            struct ReadFormat {
                uint64_t nr;
                uint64_t time_enabled;
                uint64_t time_running;
                uint64_t values[kMaxReadCounters];
            };
        };
}
