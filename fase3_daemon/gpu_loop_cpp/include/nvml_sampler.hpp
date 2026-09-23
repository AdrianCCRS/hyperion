#pragma once
#include <nvml.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "gpu_activity_tracker.hpp"

/**
 * @file
 * @brief Muestreo NVML directo (sin subprocesos `nvidia-smi`), en el mismo
 * proceso que el daemon: las 5 señales que antes traía `query_gpu_features()`
 * (util, util de memoria, potencia, reloj SM, temperatura). Cada lectura
 * cuesta microsegundos; el daemon en Python lanzaba `nvidia-smi` en cada sondeo.
 * Un valor que NVML no devuelve se reporta como ausencia (`read` -> false),
 * nunca como un cero fabricado.
 */
namespace hyperion::gpu_loop {

class NvmlSampler {
public:
    explicit NvmlSampler(unsigned int index) : index_(index) {}
    ~NvmlSampler() { if (open_) nvmlShutdown(); }
    NvmlSampler(const NvmlSampler&) = delete;
    NvmlSampler& operator=(const NvmlSampler&) = delete;

    void open() {
        if (open_) return;
        nvmlReturn_t r = nvmlInit_v2();
        if (r != NVML_SUCCESS) throw std::runtime_error(std::string("nvmlInit: ") + nvmlErrorString(r));
        r = nvmlDeviceGetHandleByIndex_v2(index_, &dev_);
        if (r != NVML_SUCCESS) { nvmlShutdown(); throw std::runtime_error(std::string("nvmlDeviceGetHandleByIndex: ") + nvmlErrorString(r)); }
        open_ = true;
    }

    bool read(GpuSnapshot& out) {
        if (!open_) return false;
        nvmlUtilization_t util{};
        unsigned int power_mw = 0, clock_mhz = 0, temp = 0;
        if (nvmlDeviceGetUtilizationRates(dev_, &util) != NVML_SUCCESS) return false;
        if (nvmlDeviceGetPowerUsage(dev_, &power_mw) != NVML_SUCCESS) return false;
        if (nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_SM, &clock_mhz) != NVML_SUCCESS) return false;
        if (nvmlDeviceGetTemperature(dev_, NVML_TEMPERATURE_GPU, &temp) != NVML_SUCCESS) return false;
        out.util_pct = util.gpu;
        out.mem_util_pct = util.memory;
        out.power_mw = power_mw;
        out.sm_clock_mhz = clock_mhz;
        out.temperature_c = temp;
        return true;
    }

    std::optional<int> sm_clock_mhz() {
        unsigned int c = 0;
        if (!open_ || nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_SM, &c) != NVML_SUCCESS) return std::nullopt;
        return static_cast<int>(c);
    }

    std::optional<int> util_pct() {
        nvmlUtilization_t u{};
        if (!open_ || nvmlDeviceGetUtilizationRates(dev_, &u) != NVML_SUCCESS) return std::nullopt;
        return static_cast<int>(u.gpu);
    }

private:
    unsigned int index_;
    bool open_ = false;
    nvmlDevice_t dev_{};
};

}  // namespace hyperion::gpu_loop
