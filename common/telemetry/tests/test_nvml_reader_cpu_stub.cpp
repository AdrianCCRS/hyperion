#include "telemetry/nvml_reader.hpp"

#include <stdexcept>

int main() {
    static_assert(!telemetry::NvmlReader::compiled_with_gpu(),
                  "CPU stub test must only be built without TELEMETRY_WITH_GPU");

    telemetry::NvmlReader reader(3);
    if(reader.device_index() != 3) return 1;
    if(reader.is_open()) return 1;

    telemetry::GpuSample sample{};
    if(reader.read(sample)) return 1;

    bool threw = false;
    try {
        reader.open();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if(!threw) return 1;
    if(reader.is_open()) return 1;

    reader.close();
    if(reader.is_open()) return 1;
    return 0;
}
