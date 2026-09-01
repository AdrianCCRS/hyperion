#include "telemetry/collector.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

/**
 * @file telemetry_overhead_bench.cpp
 * @brief Manual synthetic benchmark for collector overhead exploration.
 *
 * This is not a CTest unit test. It returns zero even when the measured
 * overhead is high because the result depends on hardware, permissions, and
 * node load.
 */
namespace {
    volatile double sink = 0.0;

    uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
    }

    void workload(int iterations) {
        double acc = 1.0;
        for(int i = 0; i < iterations; ++i) {
            acc *= 1.0000001;
        }
        sink = acc;
    }

    double mean(const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
    }

    double stddev(const std::vector<double>& values, double m) {
        double variance = 0.0;
        for(double value : values) {
            const double diff = value - m;
            variance += diff * diff;
        }
        return std::sqrt(variance / static_cast<double>(values.size()));
    }

    void print_stats(const char* label, const std::vector<double>& values) {
        const double m = mean(values);
        const double sd = stddev(values, m);
        std::printf("%s: mean=%.2f us sd=%.2f us cv=%.2f%%\n",
                    label,
                    m / 1e3,
                    sd / 1e3,
                    100.0 * sd / m);
    }

    /** Drain the ring so producer backpressure does not dominate the benchmark. */
    void drain_loop(telemetry::Collector::Ring& ring, std::atomic<bool>& stop) {
        while(!stop.load(std::memory_order_relaxed)) {
            while(ring.try_pop()) {}
            ring.flush_consumer();

            struct timespec t{0, 100'000};
            nanosleep(&t, nullptr);
        }
        while(ring.try_pop()) {}
        ring.flush_consumer();
    }
}

int main(int argc, char** argv) {
    int reps = 200;
    int iterations = 1'000'000;
    telemetry::CollectorConfig cfg;
    cfg.interval_ns = 1'000'000;

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if(arg == "--no-perf") {
            cfg.enable_perf = false;
        } else if(arg == "--interval-ns" && i + 1 < argc) {
            cfg.interval_ns = std::stol(argv[++i]);
        } else if(arg == "--reps" && i + 1 < argc) {
            reps = std::stoi(argv[++i]);
        } else if(arg == "--iters" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if(arg == "--rapl-pkg" && i + 1 < argc) {
            cfg.rapl_pkg_path = argv[++i];
        } else if(arg == "--rapl-dram" && i + 1 < argc) {
            cfg.rapl_dram_path = argv[++i];
        }
    }

    std::vector<double> baseline;
    std::vector<double> instrumented;
    baseline.reserve(static_cast<size_t>(reps));
    instrumented.reserve(static_cast<size_t>(reps));

    // Baseline first, then instrumented, mirroring the launcher philosophy but
    // with a synthetic in-process workload.
    for(int r = 0; r < reps; ++r) {
        const uint64_t t0 = now_ns();
        workload(iterations);
        baseline.push_back(static_cast<double>(now_ns() - t0));
    }

    telemetry::Collector::Ring ring;
    telemetry::Collector collector(cfg, ring);
    std::atomic<bool> stop_consumer{false};
    std::thread consumer(drain_loop, std::ref(ring), std::ref(stop_consumer));

    try {
        collector.start();
    } catch (const std::exception& e) {
        stop_consumer.store(true, std::memory_order_relaxed);
        consumer.join();
        std::printf("telemetry_overhead_bench: collector start failed: %s\n", e.what());
        return 0;
    }

    for(int r = 0; r < reps; ++r) {
        const uint64_t t0 = now_ns();
        workload(iterations);
        instrumented.push_back(static_cast<double>(now_ns() - t0));
    }

    collector.stop();
    stop_consumer.store(true, std::memory_order_relaxed);
    consumer.join();

    print_stats("baseline", baseline);
    print_stats("telemetry", instrumented);

    const double mean_baseline = mean(baseline);
    const double mean_instrumented = mean(instrumented);
    const double overhead_pct = 100.0 * (mean_instrumented - mean_baseline) / mean_baseline;
    std::printf("overhead=%.2f%% push_retries=%llu\n",
                overhead_pct,
                static_cast<unsigned long long>(collector.push_retries()));
    return 0;
}
