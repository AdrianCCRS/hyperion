#include "telemetry/perf_reader.hpp"

int main() {
    telemetry::PerfReader reader;
    telemetry::CpuSample sample{};

    if(reader.is_open()) return 1;
    if(reader.read(sample)) return 1;

    reader.enable();
    reader.disable();
    reader.close();
    if(reader.is_open()) return 1;

    if(telemetry::detail::scale_perf_count(10, 100, 100) != 10) return 1;
    if(telemetry::detail::scale_perf_count(10, 200, 100) != 20) return 1;
    if(telemetry::detail::scale_perf_count(10, 200, 0) != 0) return 1;

    return 0;
}
