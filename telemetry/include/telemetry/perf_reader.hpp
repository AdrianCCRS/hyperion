#pragma once
#include "metrics.hpp"
#include <linux/perf_event.h>
#include <sys/types.h>
#include <vector>
#include <cstdint>

namespace telemetry {
    class PerfReader {
        public:
        //pid = 0 -> monitor calling process
        //cpu = -1 -> monitor on any CPU
        //flags: inherit=false (measure this process only, not children)
        explicit PerfReader(pid_t pid = 0, int cpu = -1);
        ~PerfReader();
        
        //Open a group of counters. Call once before any read()
        // Throws std::runtime_error on failure (e.g., perf_paranoid too high).
        void open();
    
        // Read all counters atomically. Fills out and returns true on success.
        bool read(CpuSample& out) noexcept;
    
        // Enable / disable counting (use to bracket the workload phase).
        void enable()  noexcept;
        void disable() noexcept;
    
        bool is_open() const noexcept { return group_fd_ >= 0; }

        private:
            pid_t pid_;
            int   cpu_;
            int   group_fd_ = -1;   // leader fd
            std::vector<int> member_fds_;
        
            // Format returned by PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED
            // | PERF_FORMAT_TOTAL_TIME_RUNNING
            struct ReadFormat {
                uint64_t nr;
                uint64_t time_enabled;
                uint64_t time_running;
                uint64_t values[8]; // Support up to 8 counters in the group (configurable in open()) and no id because we know the order of counters. Adjust as needed.
            };
        };
}
