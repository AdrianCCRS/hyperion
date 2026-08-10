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

        // ARC-63: L2_LINES_IN_ALL (event=0xF1, umask=0x1F -- no CMask needed,
        // unlike the stalls event above) has no PERF_TYPE_HARDWARE generic
        // mapping at all, so it is always opened as PERF_TYPE_RAW directly,
        // never attempted as a generic event first. Encoding cross-checked
        // against LIKWID's validated table (ARC-60) and confirmed physically
        // sane in production use: measured against STREAM's own known byte
        // count, L2_LINES_IN_ALL x line_size landed within 4.6% of the
        // theoretical value (close to cache-misses' own 5.3% deficit on this
        // node) -- this is a cache-line-granularity proxy for memory
        // traffic, still not real DRAM bytes (uncore stays blocked, ARC-59),
        // but a second independent cross-check against bytes_moved_window's
        // existing cache-misses-based estimate. Same Ice Lake-SP gate as the
        // stalls fallback: never applied to a CPU this was not verified on.
        constexpr uint64_t kIceLakeL2LinesInAllRawConfig = 0xF1u | (0x1Fu << 8);

        // ARC-97: FP_ARITH_INST_RETIRED (event=0xC7), double-precision
        // sub-events only -- no CMask, same bit layout as L2_LINES_IN_ALL
        // above (event in bits [7:0], umask in bits [15:8]). Encoding
        // cross-checked against LIKWID's validated per-microarchitecture
        // event table (`likwid-perfctr -e | grep -i FP_ARITH` on pacca:
        // "FP_ARITH_INST_RETIRED_SCALAR_DOUBLE, 0xC7, 0x1, PMC",
        // "..._128B_PACKED_DOUBLE, 0xC7, 0x4, PMC", "..._256B_PACKED_DOUBLE,
        // 0xC7, 0x10, PMC", "..._512B_PACKED_DOUBLE, 0xC7, 0x40, PMC") and
        // re-verified empirically: all 4 opened simultaneously alongside
        // the other 6 events (pacca's full pmc_count=10 budget, ARC-53)
        // with 0% multiplexing (time_running == time_enabled on every
        // counter), and the weighted sum (1*scalar + 2*128B + 4*256B +
        // 8*512B) tracked dgemm_bench's analytical 2*iterations*n^3 within
        // 0.29-0.30% across both single-threaded and 6-core-pinned runs.
        // Single-precision sub-events (umask 0x02/0x08/0x20/0x80) are
        // deliberately not opened: there is no budget left to measure both
        // (10/10 already spent). ARC-98: all 8 CPU catalog entries were
        // spot-checked with a variant probe using the SP umasks instead of
        // the DP ones above -- dgemm_bench and stream_c/ert_probe read
        // exactly 0 on all 4 SP sub-events; the 6 NPB kernels (bt/cg/sp/ft/
        // lu/mg) each read either 0 or 1-2 scalar-single counts against
        // 1e9-1e10 double-precision operations in the same run (statistically
        // negligible, plausibly startup/libc noise, not the kernel's own
        // arithmetic). Full 8/8 coverage of the CPU catalog, not a partial
        // sample: single precision is not a meaningful source of FLOPs on
        // any CPU kernel currently in this catalog. GPU catalog entries with
        // gpu_precision=fp32 (Rodinia hotspot/heartwall/myocyte,
        // gpu_ert_probe_fp32) are unaffected by this budget constraint --
        // GPU FLOPs are measured via Nsight Compute, a separate pipeline
        // with no interaction with this CPU PMU counter budget (section
        // sec:gpu-calibracion in the thesis). If a future CPU kernel is
        // added to the catalog, its precision should be spot-checked the
        // same way before assuming double precision applies to it too. Same
        // Ice Lake-SP gate as the other raw fallbacks: never applied to a
        // CPU this was not verified on.
        constexpr uint64_t kIceLakeFpScalarDoubleRawConfig    = 0xC7u | (0x01u << 8);
        constexpr uint64_t kIceLakeFp128bPackedDoubleRawConfig = 0xC7u | (0x04u << 8);
        constexpr uint64_t kIceLakeFp256bPackedDoubleRawConfig = 0xC7u | (0x10u << 8);
        constexpr uint64_t kIceLakeFp512bPackedDoubleRawConfig = 0xC7u | (0x40u << 8);
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
            0, // kL2LinesInAll: never opened via this generic path, see below
            0, // kFpScalarDouble: raw-only, see below
            0, // kFp128bPackedDouble: raw-only, see below
            0, // kFp256bPackedDouble: raw-only, see below
            0, // kFp512bPackedDouble: raw-only, see below
        };
        static constexpr const char* kContexts[kEventCount] = {
            "perf_event_open instructions failed",
            "perf_event_open cycles failed",
            "perf_event_open cache references failed",
            "perf_event_open cache misses failed",
            "perf_event_open stalled cycles backend failed",
            "perf_event_open l2 lines in all failed",
            "perf_event_open fp scalar double failed",
            "perf_event_open fp 128b packed double failed",
            "perf_event_open fp 256b packed double failed",
            "perf_event_open fp 512b packed double failed",
        };
        // ARC-97: raw-only events (no PERF_TYPE_HARDWARE generic mapping),
        // same treatment as kL2LinesInAll -- indexed by event index for the
        // branch below.
        static constexpr uint64_t kRawOnlyConfigs[kEventCount] = {
            0, 0, 0, 0, 0,
            kIceLakeL2LinesInAllRawConfig,
            kIceLakeFpScalarDoubleRawConfig,
            kIceLakeFp128bPackedDoubleRawConfig,
            kIceLakeFp256bPackedDoubleRawConfig,
            kIceLakeFp512bPackedDoubleRawConfig,
        };

        std::vector<int> opened;
        opened.reserve(kEventCount);
        for(size_t i = 0; i < kEventCount; ++i) {
            int fd = -1;
            if(i == kL2LinesInAll || i == kFpScalarDouble || i == kFp128bPackedDouble ||
               i == kFp256bPackedDouble || i == kFp512bPackedDouble) {
                // ARC-63/ARC-97: no generic PERF_TYPE_HARDWARE mapping exists
                // for these events at all, so skip straight to the
                // model-gated raw encoding instead of trying (and always
                // failing) a generic attempt first.
                int family = 0, model = 0;
                if(detect_intel_family_model(family, model) &&
                   family == kIceLakeSPFamily && model == kIceLakeSPModel) {
                    auto raw_attr = make_raw_attr(kRawOnlyConfigs[i]);
                    fd = (int) perf_event_open(&raw_attr, pid_, cpu_, -1, 0);
                }
            } else {
                auto attr = make_hw_attr(kConfigs[i]);
                // Every event is its own group leader (group_fd=-1): inherit=1
                // does not allow grouping siblings under one leader.
                fd = (int) perf_event_open(&attr, pid_, cpu_, -1, 0);
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

        // ARC-101: enforce the all-or-nothing FP_ARITH invariant this class
        // already documents (has_fp_arith()) but this loop never enforced.
        // Each of the 4 sub-events opens independently and best-effort
        // (i >= kCoreEventCount), so a partial open -- e.g. scalar opens but
        // 512B fails -- was possible and went undetected: has_fp_arith()
        // only checks fds_[kFpScalarDouble], so callers would read "FLOPs
        // available" while the missing width's counter silently reads back
        // 0, indistinguishable from a real zero. Force all 4 to fail
        // together if any one of them did.
        bool fp_arith_partial = false;
        for(size_t i = kFpScalarDouble; i <= kFp512bPackedDouble; ++i) {
            if(opened[i] < 0) fp_arith_partial = true;
        }
        if(fp_arith_partial) {
            for(size_t i = kFpScalarDouble; i <= kFp512bPackedDouble; ++i) {
                if(opened[i] >= 0) {
                    ::close(opened[i]);
                    opened[i] = -1;
                }
            }
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

        uint64_t scaled[kEventCount] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
        sample.l2_lines_in_all = scaled[kL2LinesInAll];
        sample.fp_scalar_double = scaled[kFpScalarDouble];
        sample.fp_128b_packed_double = scaled[kFp128bPackedDouble];
        sample.fp_256b_packed_double = scaled[kFp256bPackedDouble];
        sample.fp_512b_packed_double = scaled[kFp512bPackedDouble];
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
