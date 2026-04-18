#include "telemetry/nvml_reader.hpp"
#include <ctime>
#include <stdexcept>

namespace telemetry {

    NvmlReader::NvmlReader(unsigned int device_index)
        : device_index_(device_index) {}

    NvmlReader::~NvmlReader(){
        close();
    }

#ifdef TELEMETRY_WITH_GPU
    void NvmlReader::open(){
        if(open_) return;

        nvmlReturn_t result = nvmlInit_v2();
        if(result != NVML_SUCCESS){
            throw std::runtime_error(nvmlErrorString(result));
        }

        result = nvmlDeviceGetHandleByIndex_v2(device_index_, &device_);
        if(result != NVML_SUCCESS){
            nvmlShutdown();
            throw std::runtime_error(nvmlErrorString(result));
        }

        open_ = true;
    }

    void NvmlReader::close() noexcept {
        if(open_){
            nvmlShutdown();
            open_ = false;
        }
    }

    bool NvmlReader::read(GpuSample& out) noexcept {
        if(!open_) return false;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        out.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;

        unsigned int power_mw = 0;
        if(nvmlDeviceGetPowerUsage(device_, &power_mw) != NVML_SUCCESS) return false;
        out.power_mw = power_mw;

        nvmlUtilizationRates_t util{};
        if(nvmlDeviceGetUtilizationRates(device_, &util) != NVML_SUCCESS) return false;
        out.util_pct = util.gpu;
        return true;
    }
#else
    void NvmlReader::open(){
        throw std::runtime_error("NVML support was not enabled at build time");
    }

    void NvmlReader::close() noexcept {
        open_ = false;
    }

    bool NvmlReader::read(GpuSample& out) noexcept {
        (void)out;
        return false;
    }
#endif

}
