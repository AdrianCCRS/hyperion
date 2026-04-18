#include "telemetry/perf_reader.hpp"
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <ctime>

namespace telemetry {
    
    //Wrapper for perf_event_open syscall, since it's not exposed in glibc headers.
    static long perf_event_open(perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
        return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
    }

    //Helper: build attr for a HW counter 
    static perf_event_attr make_hw_attr(uint64_t config, bool is_leader) {
        perf_event_attr attr{};
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(perf_event_attr);
        attr.config = config;
        attr.disabled = is_leader ? 1 : 0; // Leader starts disabled, members start enabled
        attr.exclude_kernel = 1; //user-space only - reduces noise
        attr.exclude_hv = 1;     //exclude hypervisor - reduces noise
        attr.inherit = 0;        //don't count child processes - simplifies interpretation
        if(is_leader){
            attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        }
        return attr;
    }

    PerfReader::PerfReader(pid_t pid, int cpu)
        : pid_(pid), cpu_(cpu) {}

    PerfReader::~PerfReader(){
        close();
    }

    void PerfReader::close() noexcept {
        disable();
        for(int fd : member_fds_){
            if(fd >= 0) ::close(fd);
        }
        member_fds_.clear();
        if(group_fd_ >= 0){
            ::close(group_fd_);
            group_fd_ = -1;
        }
    }

    void PerfReader::open(){
        if(is_open()) return;

        auto cleanup_on_error = [this]() {
            for(int fd : member_fds_){
                if(fd >= 0) ::close(fd);
            }
            member_fds_.clear();
            if(group_fd_ >= 0){
                ::close(group_fd_);
                group_fd_ = -1;
            }
        };

        //Event 0 - leader: INSTRUCTIONS
        auto a0 = make_hw_attr(PERF_COUNT_HW_INSTRUCTIONS, true);
        group_fd_ = (int) perf_event_open(&a0, pid_, cpu_, -1, 0);
        if(group_fd_ < 0) throw std::runtime_error(strerror(errno));

        //Event 1 - member: CPU-CYCLES
        auto a1 = make_hw_attr(PERF_COUNT_HW_CPU_CYCLES, false);
        int fd1 = (int) perf_event_open(&a1, pid_, cpu_, group_fd_, 0);
        if(fd1 < 0){
            std::runtime_error error(strerror(errno));
            cleanup_on_error();
            throw error;
        }
        member_fds_.push_back(fd1);

        //Event 2 - member: CACHE-REFERENCES
        auto a2 = make_hw_attr(PERF_COUNT_HW_CACHE_REFERENCES, false);
        int fd2 = (int) perf_event_open(&a2, pid_, cpu_, group_fd_, 0);
        if(fd2 < 0){
            std::runtime_error error(strerror(errno));
            cleanup_on_error();
            throw error;
        }
        member_fds_.push_back(fd2);

        //Event 3 - member: CACHE-MISSES
        auto a3 = make_hw_attr(PERF_COUNT_HW_CACHE_MISSES, false);
        int fd3 = (int) perf_event_open(&a3, pid_, cpu_, group_fd_, 0);
        if(fd3 < 0){
            std::runtime_error error(strerror(errno));
            cleanup_on_error();
            throw error;
        }
        member_fds_.push_back(fd3);

        //Arm counting (RESET + ENABLE on the leader cascades to all members)
        ioctl(group_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(group_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    bool PerfReader::read(CpuSample& out) noexcept {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        out.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;

        ReadFormat rf{};
        //Validate we read the expected amount of data (at least the header with nr, time_enabled, time_running)
        ssize_t n = ::read(group_fd_, &rf, sizeof(rf));
        if(n < (ssize_t) sizeof(uint64_t) * 3) return false;

        //Scale if multiplexing is active. 
        //TODO: implement in consumer instead here
        // auto scale = [&](uint64_t value) -> uint64_t {
        //     if(rf.time_running == 0) return 0;
        //     if(rf.time_running == rf.time_enabled) return value; // No scaling needed
        //     return (uint64_t)((double)value * rf.time_enabled / rf.time_running); //Scale up to estimate full count, like a rate
        // };

        out.instructions = rf.values[0];
        out.cycles = rf.values[1];
        out.cache_references = rf.values[2];
        out.cache_misses = rf.values[3];
        return true;
    }

    void PerfReader::enable() noexcept {
        if(group_fd_ >= 0){
            ioctl(group_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        }
    }

    void PerfReader::disable() noexcept {
        if(group_fd_ >= 0){
            ioctl(group_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        }
    }

}
