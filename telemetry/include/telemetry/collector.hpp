#pragma once
#include "metrics.hpp"
#include "spsc_ring.hpp"
#include "perf_reader.hpp"
#include "perf_cgroup_reader.hpp"
#include "rapl_reader.hpp"
#include "uncore_reader.hpp"
#include "nvml_reader.hpp"
#include "cpu_freq_reader.hpp"
#include <pthread.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace telemetry {
    /**
     * @brief Capacity used by the default telemetry ring.
     *
     * The value is a power of two so SPSCRing can wrap indices with a bit mask
     * instead of modulo. At a nominal 1 kHz sampling interval it gives several
     * seconds of slack, which is useful when the consumer is briefly descheduled.
     */
    static constexpr size_t RING_CAPACITY = 16384;

    /**
     * @brief Runtime configuration for the telemetry producer thread.
     *
     * The collector is intentionally backend-aware: each backend is opened only
     * when the corresponding option makes it necessary. This keeps local tests
     * usable on machines without PMU permissions, RAPL domains, or NVML.
     */
    struct CollectorConfig {
        /** CPU where the producer thread is pinned. -1 leaves scheduling free. */
        int producer_cpu = -1;

        /** Nominal sampling period in nanoseconds. Default is 1 ms. */
        long interval_ns = 1'000'000;

        /** Enables CPU counters through PerfReader (or the deprecated PerfCgroupReader). */
        bool enable_perf = true;

        /** Enables NVML GPU samples. Requires TELEMETRY_WITH_GPU at build time. */
        bool enable_gpu = false;

        /**
         * ARC-65: minimum nanoseconds between two NVML reads, gated inside the
         * same 1 ms producer loop (the ring is strictly single-producer, see
         * spsc_ring.hpp -- a second thread cannot push into it, so this is a
         * timing gate, not a second thread/ring). NVML's own internal
         * utilization counter only updates on the order of ~1s on many
         * drivers; polling it every 1 ms stored ~1000 duplicate rows per
         * second for nothing, and every single tick paid the syscall/ioctl
         * cost of an NVML call even when perf/RAPL were the only things that
         * actually needed that instant. Defaults to 100 ms: frequent enough
         * to attribute GPU activity at phase granularity, coarse enough that
         * only 1 in ~100 ticks can ever see NVML's latency, not all of them.
         */
        long gpu_interval_ns = 100'000'000;

        /**
         * Target PID for PerfReader (PID + inherit=1). This is the launcher's
         * only CPU measurement path; 0 means current process.
         */
        pid_t target_pid = 0;

        /**
         * Cgroup path for the deprecated PerfCgroupReader backend. Left empty
         * by telemetry_kernel_launcher; kept only for existing tests.
         */
        std::string perf_cgroup_path;

        /** CPUs where perf cgroup events are opened. This does not pin workload. */
        std::vector<int> perf_cpus;

        /** RAPL package domain directory containing energy_uj. Empty disables RAPL. */
        std::string rapl_pkg_path;

        /** Optional RAPL DRAM domain directory containing energy_uj. */
        std::string rapl_dram_path;

        /**
         * Enables node-wide uncore_imc CAS_COUNT sampling (real DRAM bytes,
         * replacing the cache-misses proxy in operational_intensity when
         * available). Unlike RAPL/perf, there is no path to gate this on --
         * boxes are auto-discovered from sysfs -- so it is an explicit flag,
         * requiring the caller (orchestrator preflight) to have already
         * confirmed an exclusive node allocation: these are system-scope
         * counters, not scoped to target_pid.
         */
        bool enable_uncore = false;

        /**
         * ARC-131: logical CPU to pin the uncore `perf stat` child process
         * to, or -1 (default) for no pinning -- see UncoreReader's
         * constructor doc. Should be set to a CPU outside the run's
         * delegated/collector/consumer CPUs when enable_uncore is true, to
         * avoid scheduling contention with the workload's own
         * FP_ARITH_INST_RETIRED counters (found empirically on paccaA100,
         * ARC-130).
         */
        int uncore_pin_cpu = -1;

        /**
         * ARC-135: full sysfs path to one representative delegated CPU's
         * cpufreq/scaling_cur_freq (e.g.
         * "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq"). Empty
         * (default) disables this signal -- CpuSample::scaling_cur_freq_khz
         * stays 0 ("not sampled"), never a fabricated reading. Sampled on
         * the SAME producer tick as the PMU counters, replacing the single
         * post-hoc Python read (taken after the workload process already
         * exited) that campaign.py used before this existed.
         */
        std::string cpu_freq_sysfs_path;

        /**
         * ARC-142: full sysfs paths for the REMAINING delegated CPUs (every
         * one after the representative CPU above), same file shape
         * ("cpufreq/scaling_cur_freq"). Empty (default) samples only the
         * representative CPU, exactly like before this field existed --
         * CpuSample::scaling_cur_freq_khz_count stays at 1 (or 0 if the
         * representative path is also empty). Pacca's cpufreq domain is
         * per-core (not per-socket like felix's), so the other delegated
         * CPUs can run at a different clock than the representative one
         * under Turbo/HWP without this signal. Capped at
         * kMaxScalingCurFreqCpus - 1 entries (the representative CPU takes
         * slot 0); extra paths beyond that are ignored, not an error.
         */
        std::vector<std::string> cpu_freq_sysfs_paths_extra;
    };

    /**
     * @brief Producer-side telemetry collector.
     *
     * Collector owns one producer pthread and writes Sample objects into a
     * caller-provided SPSC ring. It does not own a consumer and it never writes
     * CSV/JSON. The hot path is restricted to hardware/sysfs reads, try_push,
     * flush_producer, clock_gettime, and absolute sleep.
     */
    class Collector {
        public:
            using Ring = SPSCRing<Sample, RING_CAPACITY>;

            /**
             * @brief Build a collector bound to an existing ring.
             *
             * The ring must outlive the Collector. Only this collector should
             * call producer-side ring methods.
             */
            explicit Collector(CollectorConfig cfg, Ring& ring);
            ~Collector();

            /**
             * @brief Open enabled backends and start the producer thread.
             *
             * Throws on invalid configuration, missing permissions, backend
             * failures, or pthread creation/affinity errors.
             */
            void start();

            /**
             * @brief Request stop, join the producer thread, and close readers.
             */
            void stop();

            /** @return true while the producer thread is expected to be active. */
            bool running() const noexcept {return running_.load();}

            /**
             * @brief Whether this run's node could open stalled-cycles-backend.
             *
             * ARC-50: a per-node capability fact, not a per-sample one -- some
             * kernel/PMU combinations (paccaA100) never map this event.
             * Valid only after start(); false before start() or when
             * enable_perf is false.
             */
            bool has_stalled_cycles_backend() const noexcept {
                return perf_reader_.has_stalled_cycles_backend();
            }

            /**
             * @brief Whether this run's node could open L2_LINES_IN_ALL.
             *
             * ARC-63: same per-node capability semantics as
             * has_stalled_cycles_backend() -- only ever true on the Ice
             * Lake-SP family/model this raw encoding was validated on.
             */
            bool has_l2_lines_in_all() const noexcept {
                return perf_reader_.has_l2_lines_in_all();
            }

            /**
             * @brief Whether this run's node could open FP_ARITH_INST_RETIRED.
             *
             * ARC-97: same per-node capability semantics as
             * has_l2_lines_in_all() -- only ever true on the Ice Lake-SP
             * family/model this raw encoding was validated on.
             */
            bool has_fp_arith() const noexcept {
                return perf_reader_.has_fp_arith();
            }

            /**
             * @brief Whether this run's node could open every uncore_imc box.
             *
             * A per-node/per-permission capability fact, same semantics as
             * has_l2_lines_in_all() -- only true when CAP_PERFMON (or
             * equivalent) and the sysfs event definitions are both present.
             * Valid only after start(); false before start() or when
             * enable_uncore is false.
             */
            bool has_uncore() const noexcept {
                return uncore_reader_.is_open();
            }

            /**
             * @brief Number of failed try_push attempts.
             *
             * A nonzero value indicates that the ring was full at least once.
             * For clean experiments this should normally remain zero.
             */
            uint64_t push_retries() const noexcept { return push_retries_.load(std::memory_order_relaxed); }

        private:
            CollectorConfig cfg_;
            Ring& ring_;
            pthread_t thread_{};
            bool thread_started_{false};
            std::atomic<bool> running_{false};
            std::atomic<bool> stop_flag_{false};
            std::atomic<uint64_t> push_retries_{0};

            PerfReader perf_reader_;
            PerfCgroupReader perf_cgroup_reader_;
            RaplReader rapl_reader_;
            UncoreReader uncore_reader_;
            CpuFreqReader cpu_freq_reader_;
            // ARC-142: one reader per entry in cfg_.cpu_freq_sysfs_paths_extra,
            // same open/close/read lifecycle as cpu_freq_reader_ above.
            std::vector<CpuFreqReader> cpu_freq_readers_extra_;
            NvmlReader nvml_reader_;

            /**
             * ARC-65: CLOCK_MONOTONIC timestamp of the last NVML read, in
             * nanoseconds. 0 means "never sampled yet" (always sample on the
             * first tick). Only touched by the producer thread.
             */
            long long next_gpu_sample_ns_{0};

            static void* thread_entry(void* arg);

            /** Main sampling loop executed by the producer pthread. */
            void run();

            /**
             * @brief Fills sample.scaling_cur_freq_khz(_per_cpu/_count) from
             * cpu_freq_reader_ and cpu_freq_readers_extra_. Shared by both
             * the cgroup and simple-PID CPU sampling branches of run() so
             * the multi-CPU logic exists in exactly one place.
             */
            void sample_cpu_freq(CpuSample& cpu) noexcept;

            /** Legacy relative sleep helper kept for compatibility with tests/debugging. */
            void sleep_ns(long ns) const noexcept;

            /** Close every backend without throwing, used on normal and error paths. */
            void close_readers() noexcept;
    };
    
}
