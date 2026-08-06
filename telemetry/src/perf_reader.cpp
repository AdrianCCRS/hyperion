#include "telemetry/perf_reader.hpp"
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>

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

        // ARC-51: reads "cpu family"/"model" from /proc/cpuinfo. Fails
        // closed (returns false) on any parse problem -- the raw-event
        // fallback below must never fire on a CPU we could not positively
        // identify, since a wrong raw config silently measures the wrong
        // thing instead of failing loudly like the generic event does.
        bool detect_intel_family_model(int& family, int& model) noexcept {
            std::ifstream cpuinfo("/proc/cpuinfo");
            if(!cpuinfo.is_open()) return false;
            std::string line;
            bool has_family = false, has_model = false;
            while(std::getline(cpuinfo, line)) {
                const auto colon = line.find(':');
                if(colon == std::string::npos) continue;
                std::string key = line.substr(0, colon);
                std::string value = line.substr(colon + 1);
                const auto trim = [](std::string& s) {
                    const auto first = s.find_first_not_of(" \t");
                    const auto last = s.find_last_not_of(" \t");
                    s = (first == std::string::npos) ? std::string()
                                                       : s.substr(first, last - first + 1);
                };
                trim(key);
                trim(value);
                try {
                    if(key == "cpu family") { family = std::stoi(value); has_family = true; }
                    else if(key == "model") { model = std::stoi(value); has_model = true; }
                } catch(...) { return false; }
                if(has_family && has_model) return true;
            }
            return false;
        }

        // ARC-51: PERF_COUNT_HW_STALLED_CYCLES_BACKEND fails with ENOENT on
        // paccaA100 (Intel family=6 model=106, Ice Lake-SP) -- the kernel's
        // generic-to-raw event translation table lacks an entry for this
        // model, confirmed via direct syscall test, not a permission
        // problem. CYCLE_ACTIVITY.STALLS_TOTAL (event=0xA3, umask=0x04)
        // exists directly in hardware and can be opened via PERF_TYPE_RAW,
        // but ALSO requires CMask=0x04 in bits 24-31 -- omitting it (first
        // attempt) opened successfully but returned a physically impossible
        // reading (stalls > cycles). The full encoding was cross-checked
        // against LIKWID's validated per-microarchitecture event table
        // (`likwid-perfctr -e | grep -i stall`: "CYCLE_ACTIVITY_STALLS_TOTAL,
        // 0xA3, 0x4, PMC, THRESHOLD=0x4") and re-verified empirically
        // (stalls/cycles ratio came back plausible, 0 < ratio < 1) before
        // landing here. Scoped to this exact family/model: never applied to
        // a CPU we have not verified this encoding on.
        constexpr int kIceLakeSPFamily = 6;
        constexpr int kIceLakeSPModel = 106;
        constexpr uint64_t kIceLakeStallsTotalRawConfig =
            0xA3u | (0x04u << 8) | (0x04ull << 24);
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

    // ARC-51: same shape as make_hw_attr but PERF_TYPE_RAW, for the
    // Ice Lake-SP CYCLE_ACTIVITY.STALLS_TOTAL fallback.
    static perf_event_attr make_raw_attr(uint64_t config) {
        perf_event_attr attr = make_hw_attr(config);
        attr.type = PERF_TYPE_RAW;
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
            PERF_COUNT_HW_STALLED_CYCLES_BACKEND,
        };
        static constexpr const char* kContexts[kEventCount] = {
            "perf_event_open instructions failed",
            "perf_event_open cycles failed",
            "perf_event_open cache references failed",
            "perf_event_open cache misses failed",
            "perf_event_open stalled cycles backend failed",
        };

        std::vector<int> opened;
        opened.reserve(kEventCount);
        for(size_t i = 0; i < kEventCount; ++i) {
            auto attr = make_hw_attr(kConfigs[i]);
            // Every event is its own group leader (group_fd=-1): inherit=1
            // does not allow grouping siblings under one leader.
            int fd = (int) perf_event_open(&attr, pid_, cpu_, -1, 0);
            if(fd < 0 && i == kStalledCyclesBackend) {
                // ARC-51: the generic event is unmapped on this kernel/PMU
                // (ENOENT), but the underlying hardware counter may still
                // exist. Retry once via PERF_TYPE_RAW with a validated,
                // model-specific encoding -- never on a CPU we have not
                // confirmed this on.
                int family = 0, model = 0;
                if(detect_intel_family_model(family, model) &&
                   family == kIceLakeSPFamily && model == kIceLakeSPModel) {
                    auto raw_attr = make_raw_attr(kIceLakeStallsTotalRawConfig);
                    fd = (int) perf_event_open(&raw_attr, pid_, cpu_, -1, 0);
                }
            }
            if(fd < 0) {
                // ARC-50: events past kCoreEventCount are best-effort. A
                // kernel/PMU that does not map them (ENOENT on paccaA100,
                // Ice Lake + RHEL8) must not take down the 4 core counters
                // this node CAN provide.
                if(i < kCoreEventCount) {
                    std::runtime_error error = errno_error(kContexts[i]);
                    for(int f : opened) if(f >= 0) ::close(f);
                    throw error;
                }
                opened.push_back(-1);
                continue;
            }
            opened.push_back(fd);
        }

        for(int fd : opened) {
            if(fd < 0) continue; // optional event never opened, nothing to arm
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

        uint64_t scaled[kEventCount] = {0, 0, 0, 0, 0};
        uint64_t time_enabled = 0;
        uint64_t time_running = 0;

        for(size_t i = 0; i < kEventCount; ++i) {
            if(fds_[i] < 0) continue; // optional event unavailable on this node (ARC-50)
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
        sample.stalled_cycles_backend = scaled[kStalledCyclesBackend];
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
