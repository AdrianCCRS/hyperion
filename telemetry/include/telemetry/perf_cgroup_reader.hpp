#pragma once

#include "metrics.hpp"

#include <cstdint>
#include <linux/perf_event.h>
#include <string>
#include <vector>

namespace telemetry {

    class PerfCgroupReader {
    public:
        PerfCgroupReader(std::string cgroup_path, std::vector<int> cpus);
        ~PerfCgroupReader();

        void open();
        void close() noexcept;
        bool read(CpuSample& out) noexcept;
        bool is_open() const noexcept { return cgroup_fd_ >= 0; }

    private:
        static constexpr uint64_t kExpectedCounters = 4;
        static constexpr uint64_t kMaxReadCounters = 8;

        struct CpuEvents {
            int group_fd = -1;
            std::vector<int> member_fds;
        };

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

        void close_events() noexcept;
    };

}
