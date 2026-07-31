#pragma once
#include "metrics.hpp"
#include "spsc_ring.hpp"
#include "perf_reader.hpp"
#include "perf_cgroup_reader.hpp"
#include "rapl_reader.hpp"
#include "nvml_reader.hpp"
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
            NvmlReader nvml_reader_;

            static void* thread_entry(void* arg);

            /** Main sampling loop executed by the producer pthread. */
            void run();

            /** Legacy relative sleep helper kept for compatibility with tests/debugging. */
            void sleep_ns(long ns) const noexcept;

            /** Close every backend without throwing, used on normal and error paths. */
            void close_readers() noexcept;
    };
    
}
