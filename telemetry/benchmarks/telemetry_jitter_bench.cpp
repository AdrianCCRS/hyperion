#include "telemetry/collector.hpp"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <string>
#include <vector>

namespace {
    uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
    }

    void short_sleep_ns(long ns) {
        struct timespec t;
        t.tv_sec = ns / 1'000'000'000L;
        t.tv_nsec = ns % 1'000'000'000L;
        nanosleep(&t, nullptr);
    }
}

int main(int argc, char** argv) {
    size_t target_samples = 10'000;
    telemetry::CollectorConfig cfg;
    cfg.interval_ns = 1'000'000;

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if(arg == "--samples" && i + 1 < argc) {
            target_samples = static_cast<size_t>(std::stoull(argv[++i]));
        } else if(arg == "--interval-ns" && i + 1 < argc) {
            cfg.interval_ns = std::stol(argv[++i]);
        } else if(arg == "--rapl-pkg" && i + 1 < argc) {
            cfg.rapl_pkg_path = argv[++i];
        } else if(arg == "--rapl-dram" && i + 1 < argc) {
            cfg.rapl_dram_path = argv[++i];
        }
    }

    telemetry::Collector::Ring ring;
    telemetry::Collector collector(cfg, ring);

    try {
        collector.start();
    } catch (const std::exception& e) {
        std::printf("telemetry_jitter_bench: collector start failed: %s\n", e.what());
        return 0;
    }

    std::vector<uint64_t> timestamps;
    timestamps.reserve(target_samples);

    const uint64_t deadline = now_ns() + (target_samples + 1000ULL) *
                              static_cast<uint64_t>(cfg.interval_ns);
    while(timestamps.size() < target_samples && now_ns() < deadline) {
        while(auto sample = ring.try_pop()) {
            if(sample->tag == telemetry::SampleTag::CPU) {
                timestamps.push_back(sample->cpu.timestamp_ns);
                if(timestamps.size() >= target_samples) break;
            }
        }
        ring.flush_consumer();
        short_sleep_ns(100'000);
    }

    collector.stop();

    if(timestamps.size() < 2) {
        std::printf("telemetry_jitter_bench: insufficient samples=%llu push_retries=%llu\n",
                    static_cast<unsigned long long>(timestamps.size()),
                    static_cast<unsigned long long>(collector.push_retries()));
        return 0;
    }

    std::vector<double> intervals;
    intervals.reserve(timestamps.size() - 1);
    for(size_t i = 1; i < timestamps.size(); ++i) {
        intervals.push_back(static_cast<double>(timestamps[i] - timestamps[i - 1]));
    }

    const double mean = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
                        static_cast<double>(intervals.size());
    double variance = 0.0;
    for(double interval : intervals) {
        const double diff = interval - mean;
        variance += diff * diff;
    }
    const double sd = std::sqrt(variance / static_cast<double>(intervals.size()));

    std::printf("samples=%llu interval_mean=%.2f us sd=%.2f us cv=%.2f%% push_retries=%llu\n",
                static_cast<unsigned long long>(timestamps.size()),
                mean / 1e3,
                sd / 1e3,
                100.0 * sd / mean,
                static_cast<unsigned long long>(collector.push_retries()));
    return 0;
}
