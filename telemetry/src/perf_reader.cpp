#include "telemetry/perf_reader.hpp"
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <ctime>
#include <string>

/**
 * @file perf_reader.cpp
 * @brief perf_event reader attached to an external process (PID + inherit).
 *
 * See docs/retoma/Guia_Maestra_Fase1_DVFS.md section 3 for why pid+inherit
 * with live per-fd reads is required instead of pid=0 with inherited folding,
 * and why grouped reads (PERF_FORMAT_GROUP) are incompatible with inherit=1.
 */
namespace telemetry {
    namespace detail {
        uint64_t scale_perf_count(uint64_t value,
                                  uint64_t time_enabled,
                                  uint64_t time_running) noexcept {
            if(time_running == 0) return 0;
            if(time_running == time_enabled) return value;
            return static_cast<uint64_t>(
                static_cast<double>(value) *
                static_cast<double>(time_enabled) /
                static_cast<double>(time_running)
            );
        }
    }

    namespace {
        std::runtime_error errno_error(const char* context) {
            return std::runtime_error(std::string(context) + ": " + std::strerror(errno));
        }
    }

    //Wrapper for perf_event_open syscall, since it's not exposed in glibc headers.
    static long perf_event_open(perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
        return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
    }

    // Build one independent hardware counter. inherit=1 forbids grouped reads,
    // so every event carries its own TOTAL_TIME_ENABLED/RUNNING and is opened
    // with group_fd=-1 (its own leader).
    static perf_event_attr make_hw_attr(uint64_t config) {
        perf_event_attr attr{};
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(perf_event_attr);
        attr.config = config;
        attr.disabled = 1;       // armed explicitly via IOC_RESET + IOC_ENABLE
        attr.exclude_kernel = 1; //user-space only - reduces noise
        attr.exclude_hv = 1;     //exclude hypervisor - reduces noise
        attr.inherit = 1;        //cover descendants spawned by the measured pid
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        return attr;
    }

    PerfReader::PerfReader(pid_t pid, int cpu)
        : pid_(pid), cpu_(cpu) {}

    PerfReader::~PerfReader(){
        close();
    }

    void PerfReader::close() noexcept {
        disable();
        for(int fd : fds_){
            if(fd >= 0) ::close(fd);
        }
        fds_.clear();
    }

    void PerfReader::open(){
        if(is_open()) return;

        static constexpr uint64_t kConfigs[kEventCount] = {
            PERF_COUNT_HW_INSTRUCTIONS,
            PERF_COUNT_HW_CPU_CYCLES,
            PERF_COUNT_HW_CACHE_REFERENCES,
            PERF_COUNT_HW_CACHE_MISSES,
        };
        static constexpr const char* kContexts[kEventCount] = {
            "perf_event_open instructions failed",
            "perf_event_open cycles failed",
            "perf_event_open cache references failed",
            "perf_event_open cache misses failed",
        };

        std::vector<int> opened;
        opened.reserve(kEventCount);
        for(size_t i = 0; i < kEventCount; ++i) {
            auto attr = make_hw_attr(kConfigs[i]);
            // Every event is its own group leader (group_fd=-1): inherit=1
            // does not allow grouping siblings under one leader.
            const int fd = (int) perf_event_open(&attr, pid_, cpu_, -1, 0);
            if(fd < 0) {
                std::runtime_error error = errno_error(kContexts[i]);
                for(int f : opened) if(f >= 0) ::close(f);
                throw error;
            }
            opened.push_back(fd);
        }

        for(int fd : opened) {
            if(ioctl(fd, PERF_EVENT_IOC_RESET, 0) < 0) {
                std::runtime_error error = errno_error("PERF_EVENT_IOC_RESET failed");
                for(int f : opened) if(f >= 0) ::close(f);
                throw error;
            }
            if(ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
                std::runtime_error error = errno_error("PERF_EVENT_IOC_ENABLE failed");
                for(int f : opened) if(f >= 0) ::close(f);
                throw error;
            }
        }

        fds_ = std::move(opened);
    }

    bool PerfReader::read(CpuSample& out) noexcept {
        if(!is_open()) return false;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        uint64_t scaled[kEventCount] = {0, 0, 0, 0};
        uint64_t time_enabled = 0;
        uint64_t time_running = 0;

        for(size_t i = 0; i < kEventCount; ++i) {
            ReadFormat rf{};
            const ssize_t n = ::read(fds_[i], &rf, sizeof(rf));
            if(n != static_cast<ssize_t>(sizeof(rf))) return false;
            scaled[i] = detail::scale_perf_count(rf.value, rf.time_enabled, rf.time_running);
            if(i == kInstructions) {
                time_enabled = rf.time_enabled;
                time_running = rf.time_running;
            }
        }

        CpuSample sample{};
        sample.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;
        sample.instructions = scaled[kInstructions];
        sample.cycles = scaled[kCycles];
        sample.cache_references = scaled[kCacheReferences];
        sample.cache_misses = scaled[kCacheMisses];
        sample.time_enabled_ns = time_enabled;
        sample.time_running_ns = time_running;
        out = sample;
        return true;
    }

    void PerfReader::enable() noexcept {
        for(int fd : fds_) ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    void PerfReader::disable() noexcept {
        for(int fd : fds_) ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    }

}
