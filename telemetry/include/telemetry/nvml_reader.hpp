#pragma once
#include "metrics.hpp"
#include <cstdint>

#ifdef TELEMETRY_WITH_GPU
#include <nvml.h>
#endif

namespace telemetry {
    /**
     * @brief Optional NVML device-level telemetry reader.
     *
     * The class compiles in CPU-only builds. In that mode open() throws and
     * read() returns false. With TELEMETRY_WITH_GPU it initializes NVML, obtains
     * one device handle, and reads device power/utilization snapshots.
     */
    class NvmlReader
    {
    private:
        unsigned int device_index_;
        bool open_ = false;
        #ifdef TELEMETRY_WITH_GPU
            nvmlDevice_t device_{};
        #endif
    public:
        /** @param device_index NVML device index to sample. */
        explicit NvmlReader(unsigned int device_index = 0);
        ~NvmlReader();

        /** @brief Initialize NVML and acquire the configured device handle. */
        void open();

        /** @brief Release NVML state owned by this reader. */
        void close() noexcept;

        /**
         * @brief Read power and utilization from the configured device.
         *
         * Returns false if NVML support is disabled, the reader is closed, or
         * NVML cannot return one of the requested values.
         */
        bool read(GpuSample& out) noexcept;

        /** @return true when a device handle is active. */
        bool is_open() const noexcept { return open_; }

        /** @return configured NVML device index. */
        unsigned int device_index() const noexcept { return device_index_; }

        /** @return compile-time indication of NVML support. */
        static constexpr bool compiled_with_gpu() noexcept {
        #ifdef TELEMETRY_WITH_GPU
            return true;
        #else
            return false;
        #endif
        }
    };
}
