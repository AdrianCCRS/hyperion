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
    // Ring size: power of two, chosen to hold ~10 s of samples at 1 kHz
    // without overflow. Tune based on actual sampling interval.
    static constexpr size_t RING_CAPACITY = 16384;

    struct CollectorConfig {
        int producer_cpu = -1; // -1 = no pinning
        long interval_ns = 1'000'000; // 1 ms default sampling interval
        bool enable_perf = true; // Whether to collect CPU perf_event counters.
        bool enable_gpu = false; //Whether to collect GPU metrics (requires NVML and compatible GPU)
        pid_t target_pid = 0; //PID of the process to monitor (0 = self)
        std::string perf_cgroup_path;
        std::vector<int> perf_cpus;
        std::string rapl_pkg_path;
        std::string rapl_dram_path;
    };

    class Collector {
        public:
            using Ring = SPSCRing<Sample, RING_CAPACITY>;
            explicit Collector(CollectorConfig cfg, Ring& ring);
            ~Collector();

            //Start the producer thread. Throws on failure (e.g. invalid config, thread creation failure).
            void start();

            //Signal the producer thread to stop and join it.
            void stop();

            bool running() const noexcept {return running_.load();}
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
            void run(); //Main loop of the producer thread
            void sleep_ns(long ns) const noexcept; //Helper to sleep for the specified interval with high precision
            void close_readers() noexcept;
    };
    
}
