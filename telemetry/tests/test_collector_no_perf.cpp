#include "telemetry/collector.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <time.h>
#include <unistd.h>

namespace {
    uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
    }

    void sleep_ns(long ns) {
        struct timespec t;
        t.tv_sec = ns / 1'000'000'000L;
        t.tv_nsec = ns % 1'000'000'000L;
        nanosleep(&t, nullptr);
    }

    bool write_file(const std::filesystem::path& path, const char* value) {
        std::ofstream out(path);
        out << value;
        return static_cast<bool>(out);
    }
}

int main() {
    {
        telemetry::Collector::Ring ring;
        telemetry::CollectorConfig cfg;
        cfg.enable_perf = false;
        cfg.interval_ns = 1'000'000;

        telemetry::Collector collector(cfg, ring);
        collector.start();
        if(!collector.running()) return 1;

        sleep_ns(2'000'000);
        collector.stop();

        if(collector.running()) return 1;
        if(collector.push_retries() != 0) return 1;
        if(ring.try_pop()) return 1;
    }

    {
        const auto base = std::filesystem::temp_directory_path() /
                          ("telemetry-collector-no-perf-" + std::to_string(::getpid()));
        const auto pkg = base / "pkg";
        std::filesystem::create_directories(pkg);

        if(!write_file(pkg / "energy_uj", "1000\n")) return 1;
        if(!write_file(pkg / "max_energy_range_uj", "5000\n")) return 1;

        telemetry::Collector::Ring ring;
        telemetry::CollectorConfig cfg;
        cfg.enable_perf = false;
        cfg.interval_ns = 1;
        cfg.rapl_pkg_path = pkg.string();

        telemetry::Collector collector(cfg, ring);
        collector.start();

        const uint64_t deadline = now_ns() + 2'000'000'000ULL;
        while(collector.push_retries() == 0 && now_ns() < deadline) {
            sleep_ns(1'000'000);
        }
        collector.stop();

        const bool saw_backpressure = collector.push_retries() > 0;
        std::filesystem::remove_all(base);
        if(!saw_backpressure) return 1;
    }

    return 0;
}
