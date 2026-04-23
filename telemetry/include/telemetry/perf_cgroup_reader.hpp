#pragma once

#include "metrics.hpp"

#include <cstdint>
#include <linux/perf_event.h>
#include <string>
#include <vector>

namespace telemetry {

    /**
     * @brief perf_event reader for a cgroup across explicit CPUs.
     *
     * Linux perf cgroup mode is opened per CPU with PERF_FLAG_PID_CGROUP. This
     * reader opens the same hardware-counter group on each requested CPU and
     * aggregates values into one CpuSample per read. It is the preferred CPU
     * backend for multithreaded workload experiments.
     */
    class PerfCgroupReader {
    public:
        /**
         * @param cgroup_path Directory path of a pre-created/delegated cgroup.
         * @param cpus CPU ids where perf events will be opened.
         */
        PerfCgroupReader(std::string cgroup_path, std::vector<int> cpus);
        ~PerfCgroupReader();

        /**
         * @brief Open the cgroup directory and one perf group per configured CPU.
         *
         * Throws on missing cgroup, empty CPU set, invalid CPU ids, or perf
         * permission/event failures.
         */
        void open();

        /** @brief Disable and close all perf groups plus the cgroup fd. */
        void close() noexcept;

        /**
         * @brief Aggregate one grouped perf read from every configured CPU.
         *
         * Returns false if any opened CPU returns an invalid read. The sample
         * timestamp is taken once before the aggregation loop.
         */
        bool read(CpuSample& out) noexcept;

        /** @return true when the cgroup directory descriptor is open. */
        bool is_open() const noexcept { return cgroup_fd_ >= 0; }

    private:
        static constexpr uint64_t kExpectedCounters = 4;
        static constexpr uint64_t kMaxReadCounters = 8;

        /** File descriptors for one CPU-local perf group. */
        struct CpuEvents {
            int group_fd = -1;
            std::vector<int> member_fds;
        };

        /** Kernel read layout for PERF_FORMAT_GROUP with time diagnostics. */
        struct ReadFormat {
            uint64_t nr;
            uint64_t time_enabled;
            uint64_t time_running;
            uint64_t values[kMaxReadCounters];
        };

        std::string cgroup_path_;
        std::vector<int> cpus_;
        int cgroup_fd_ = -1;
        std::vector<CpuEvents> cpu_events_;

        /** Close only CPU-local perf groups, keeping cgroup_fd_ ownership clear. */
        void close_events() noexcept;
    };

}
