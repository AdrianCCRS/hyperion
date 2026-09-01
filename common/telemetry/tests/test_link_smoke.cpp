#include "telemetry/collector.hpp"
#include "telemetry/nvml_reader.hpp"
#include "telemetry/perf_reader.hpp"
#include "telemetry/rapl_reader.hpp"

int main() {
    telemetry::PerfReader perf;
    telemetry::RaplReader rapl("/tmp");
    telemetry::NvmlReader nvml;

    telemetry::Collector::Ring ring;
    telemetry::CollectorConfig cfg;
    telemetry::Collector collector(cfg, ring);

    return (perf.is_open() || rapl.is_open() || nvml.is_open() || collector.running()) ? 1 : 0;
}
