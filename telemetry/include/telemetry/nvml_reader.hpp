#pragma once
#include "metrics.hpp"
#include <cstdint>

#ifdef TELEMETRY_WITH_GPU
#include <nvml.h>
#endif

namespace telemetry {
    class NvmlReader
    {
    private:
        unsigned int device_index_;
        bool open_ = false;
        #ifdef TELEMETRY_WITH_GPU
            nvmlDevice_t device_{};
        #endif
    public:
        explicit NvmlReader(unsigned int device_index = 0);
        ~NvmlReader();

        void open(); //nvmlInit + nvmlDeviceGetHandleByIndex
        void close() noexcept;

        bool read(GpuSample& out) noexcept; //nvmlDeviceGetPowerUsage
        bool is_open() const noexcept { return open_; }
    };
}