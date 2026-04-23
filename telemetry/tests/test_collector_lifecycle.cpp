#include "telemetry/collector.hpp"

#include <stdexcept>

int main() {
    telemetry::Collector::Ring ring;

    {
        telemetry::CollectorConfig cfg;
        telemetry::Collector collector(cfg, ring);
        if(collector.running()) return 1;
        collector.stop();
        collector.stop();
        if(collector.running()) return 1;
    }

    {
        telemetry::CollectorConfig cfg;
        cfg.interval_ns = 0;
        telemetry::Collector collector(cfg, ring);

        bool threw = false;
        try {
            collector.start();
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if(!threw) return 1;
        if(collector.running()) return 1;
        collector.stop();
    }

    if(!telemetry::NvmlReader::compiled_with_gpu()){
        telemetry::CollectorConfig cfg;
        cfg.enable_gpu = true;
        telemetry::Collector collector(cfg, ring);

        bool threw = false;
        try {
            collector.start();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if(!threw) return 1;
        if(collector.running()) return 1;
        collector.stop();
    }

    return 0;
}
