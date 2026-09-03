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

        // ARC-69: the real NVML type is nvmlUtilization_t, not
        // nvmlUtilizationRates_t -- confirmed against the real header on
        // paccaA100 (CUDA 12.0). The wrong name compiled locally against a
        // hand-written stub that happened to reuse the same (wrong) name,
        // so this was never caught until building with WITH_GPU=ON against
        // a real nvml.h for the first time.
        nvmlUtilization_t util{};
        if(nvmlDeviceGetUtilizationRates(device_, &util) != NVML_SUCCESS) return false;
        sample.util_pct = util.gpu;
        sample.mem_util_pct = util.memory;

        // ARC-94: SM clock, accumulated energy, and temperature are
        // best-effort, unlike power/util above -- some driver/GPU
        // combinations don't support one of these (e.g.
        // nvmlDeviceGetTotalEnergyConsumption needs a fairly recent
        // driver), and a missing optional metric should never invalidate
        // an otherwise-valid power/util reading.
        //
        // F1-GPU-001: check every return code. A failed call leaves the
        // value at 0 AND its *_valid flag at false, so the export path can
        // write an empty cell instead of a 0 that downstream mistakes for
        // a real reading (postprocess.py was fabricating gpu_energy_valid
        // on drivers without energy support: previous==current==0 =>
        // "delta 0, valid"). Same "not measured != real zero" contract as
        // stalled_cycles_mem_any / UncoreSnapshot::interval_valid.
        unsigned int sm_clock_mhz = 0;
        sample.sm_clock_valid =
            nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &sm_clock_mhz) == NVML_SUCCESS;
        sample.sm_clock_mhz = sample.sm_clock_valid ? sm_clock_mhz : 0;

        unsigned long long energy_mj = 0;
        sample.energy_valid =
            nvmlDeviceGetTotalEnergyConsumption(device_, &energy_mj) == NVML_SUCCESS;
        sample.energy_mj = sample.energy_valid ? energy_mj : 0;

        unsigned int temperature_c = 0;
        sample.temperature_valid =
            nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temperature_c) == NVML_SUCCESS;
        sample.temperature_c = sample.temperature_valid ? temperature_c : 0;

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
