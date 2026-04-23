#include "telemetry/nvml_reader.hpp"
#include <ctime>
#include <stdexcept>

/**
 * @file nvml_reader.cpp
 * @brief Optional NVML reader implementation.
 *
 * CPU-only builds keep the same public API but fail early if GPU telemetry is
 * requested. This lets tests link the telemetry library without NVIDIA headers
 * or libnvidia-ml.
 */
namespace telemetry {

    NvmlReader::NvmlReader(unsigned int device_index)
        : device_index_(device_index) {}

    NvmlReader::~NvmlReader(){
        close();
    }

#ifdef TELEMETRY_WITH_GPU
    void NvmlReader::open(){
        if(open_) return;

        // NVML initialization is intentionally outside the sampling loop. The
        // producer only performs device queries after the reader is open.
        nvmlReturn_t result = nvmlInit_v2();
        if(result != NVML_SUCCESS){
            throw std::runtime_error(nvmlErrorString(result));
        }

        result = nvmlDeviceGetHandleByIndex_v2(device_index_, &device_);
        if(result != NVML_SUCCESS){
            nvmlShutdown();
            device_ = {};
            throw std::runtime_error(nvmlErrorString(result));
        }

        open_ = true;
    }

    void NvmlReader::close() noexcept {
        if(open_){
            nvmlShutdown();
            open_ = false;
            device_ = {};
        }
    }

    bool NvmlReader::read(GpuSample& out) noexcept {
        if(!open_) return false;

        GpuSample sample{};
        // Timestamp first so CPU, RAPL, and GPU rows share the same monotonic
        // time base even though the underlying APIs differ.
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        sample.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;

        unsigned int power_mw = 0;
        if(nvmlDeviceGetPowerUsage(device_, &power_mw) != NVML_SUCCESS) return false;
        sample.power_mw = power_mw;

        nvmlUtilizationRates_t util{};
        if(nvmlDeviceGetUtilizationRates(device_, &util) != NVML_SUCCESS) return false;
        sample.util_pct = util.gpu;
        out = sample;
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
