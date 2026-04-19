#include "telemetry/perf_cgroup_reader.hpp"
#include "telemetry/perf_reader.hpp"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

namespace telemetry {
    namespace {
        std::runtime_error errno_error(const char* context, const std::string& detail = {}) {
            std::string message = context;
            if(!detail.empty()) {
                message += ": ";
                message += detail;
            }
            message += ": ";
            message += std::strerror(errno);
            return std::runtime_error(message);
        }

        long perf_event_open(perf_event_attr* attr,
                             pid_t pid,
                             int cpu,
                             int group_fd,
                             unsigned long flags) {
            return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
        }

        perf_event_attr make_hw_attr(uint64_t config, bool is_leader) {
            perf_event_attr attr{};
            attr.type = PERF_TYPE_HARDWARE;
            attr.size = sizeof(perf_event_attr);
            attr.config = config;
            attr.disabled = is_leader ? 1 : 0;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            attr.inherit = 0;
            if(is_leader) {
                attr.read_format = PERF_FORMAT_GROUP |
                                   PERF_FORMAT_TOTAL_TIME_ENABLED |
                                   PERF_FORMAT_TOTAL_TIME_RUNNING;
            }
            return attr;
        }

        void close_fd(int& fd) noexcept {
            if(fd >= 0) {
                ::close(fd);
                fd = -1;
            }
        }
    }

    PerfCgroupReader::PerfCgroupReader(std::string cgroup_path, std::vector<int> cpus)
        : cgroup_path_(std::move(cgroup_path)),
          cpus_(std::move(cpus)) {}

    PerfCgroupReader::~PerfCgroupReader() {
        close();
    }

    void PerfCgroupReader::open() {
        if(is_open()) return;
        if(cgroup_path_.empty()) {
            throw std::invalid_argument("PerfCgroupReader requires a cgroup path");
        }
        if(cpus_.empty()) {
            throw std::invalid_argument("PerfCgroupReader requires at least one CPU");
        }

        cgroup_fd_ = ::open(cgroup_path_.c_str(), O_RDONLY | O_DIRECTORY);
        if(cgroup_fd_ < 0) {
            throw errno_error("Failed to open perf cgroup", cgroup_path_);
        }

        cpu_events_.clear();
        cpu_events_.reserve(cpus_.size());

        try {
            for(int cpu : cpus_) {
                if(cpu < 0) {
                    throw std::invalid_argument("PerfCgroupReader CPU ids must be non-negative");
                }

                CpuEvents events;
                auto cleanup_local = [&events]() noexcept {
                    for(int& fd : events.member_fds) {
                        close_fd(fd);
                    }
                    events.member_fds.clear();
                    close_fd(events.group_fd);
                };

                try {
                    auto a0 = make_hw_attr(PERF_COUNT_HW_INSTRUCTIONS, true);
                    events.group_fd = static_cast<int>(
                        perf_event_open(&a0, cgroup_fd_, cpu, -1, PERF_FLAG_PID_CGROUP)
                    );
                    if(events.group_fd < 0) {
                        throw errno_error("perf_event_open cgroup instructions failed");
                    }

                    auto open_member = [&](uint64_t config, const char* name) {
                        auto attr = make_hw_attr(config, false);
                        const int fd = static_cast<int>(
                            perf_event_open(&attr, cgroup_fd_, cpu, events.group_fd, PERF_FLAG_PID_CGROUP)
                        );
                        if(fd < 0) {
                            throw errno_error(name);
                        }
                        events.member_fds.push_back(fd);
                    };

                    events.member_fds.reserve(kExpectedCounters - 1);
                    open_member(PERF_COUNT_HW_CPU_CYCLES, "perf_event_open cgroup cycles failed");
                    open_member(PERF_COUNT_HW_CACHE_REFERENCES, "perf_event_open cgroup cache references failed");
                    open_member(PERF_COUNT_HW_CACHE_MISSES, "perf_event_open cgroup cache misses failed");

                    if(ioctl(events.group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0) {
                        throw errno_error("PERF_EVENT_IOC_RESET cgroup failed");
                    }
                    if(ioctl(events.group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) {
                        throw errno_error("PERF_EVENT_IOC_ENABLE cgroup failed");
                    }
                } catch (...) {
                    cleanup_local();
                    throw;
                }

                cpu_events_.push_back(std::move(events));
            }
        } catch (...) {
            close();
            throw;
        }
    }

    void PerfCgroupReader::close_events() noexcept {
        for(auto& events : cpu_events_) {
            if(events.group_fd >= 0) {
                ioctl(events.group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            }
            for(int& fd : events.member_fds) {
                close_fd(fd);
            }
            events.member_fds.clear();
            close_fd(events.group_fd);
        }
        cpu_events_.clear();
    }

    void PerfCgroupReader::close() noexcept {
        close_events();
        close_fd(cgroup_fd_);
    }

    bool PerfCgroupReader::read(CpuSample& out) noexcept {
        if(!is_open()) return false;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        CpuSample sample{};
        sample.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;

        for(const auto& events : cpu_events_) {
            ReadFormat rf{};
            const ssize_t n = ::read(events.group_fd, &rf, sizeof(rf));
            if(n < static_cast<ssize_t>(sizeof(uint64_t) * 3)) return false;
            if(rf.nr < kExpectedCounters || rf.nr > kMaxReadCounters) return false;

            const auto expected_bytes = static_cast<ssize_t>(
                sizeof(uint64_t) * (3 + rf.nr)
            );
            if(n < expected_bytes) return false;

            sample.instructions += detail::scale_perf_count(rf.values[0],
                                                            rf.time_enabled,
                                                            rf.time_running);
            sample.cycles += detail::scale_perf_count(rf.values[1],
                                                      rf.time_enabled,
                                                      rf.time_running);
            sample.cache_references += detail::scale_perf_count(rf.values[2],
                                                                rf.time_enabled,
                                                                rf.time_running);
            sample.cache_misses += detail::scale_perf_count(rf.values[3],
                                                            rf.time_enabled,
                                                            rf.time_running);
            sample.time_enabled_ns += rf.time_enabled;
            sample.time_running_ns += rf.time_running;
        }

        out = sample;
        return true;
    }
}
