#pragma once
#include <cstdint>
#include <ctime>

/**
 * @file
 * @brief Fixed-size telemetry sample definitions.
 *
 * These types are intentionally plain data structures. They cross the SPSC ring
 * from producer to consumer, so they avoid dynamic allocation, ownership, and
 * virtual dispatch. Derived metrics such as IPC, cache-miss ratio, power, or
 * energy deltas should be computed by the consumer/export path, not here.
 */
namespace telemetry {
     
    /**
     * @brief Nanoseconds from CLOCK_MONOTONIC.
     *
     * CLOCK_MONOTONIC avoids wall-clock jumps caused by time synchronization and
     * is therefore appropriate for sampling intervals and elapsed-time analysis.
     */
    using ns_t = uint64_t;

    /**
     * @brief CPU hardware-counter sample.
     *
     * Values are produced by perf_event. Counters may be scaled when perf
     * reports multiplexing through time_enabled/time_running.
     */
    struct CpuSample {
        ns_t     timestamp_ns;
        uint64_t instructions;      /**< PERF_COUNT_HW_INSTRUCTIONS. */
        uint64_t cycles;            /**< PERF_COUNT_HW_CPU_CYCLES. */
        uint64_t cache_references;  /**< PERF_COUNT_HW_CACHE_REFERENCES. */
        uint64_t cache_misses;      /**< PERF_COUNT_HW_CACHE_MISSES. */
        uint64_t stalled_cycles_backend; /**< PERF_COUNT_HW_STALLED_CYCLES_BACKEND. */
        uint64_t l2_lines_in_all;   /**< Raw L2_LINES_IN_ALL (event=0xF1,umask=0x1F), Ice Lake-SP only. */
        uint64_t time_enabled_ns;   /**< Perf enabled time for multiplexing diagnostics. */
        uint64_t time_running_ns;   /**< Perf running time for multiplexing diagnostics. */
    };

    /**
     * @brief Raw Intel RAPL energy snapshot.
     *
     * RAPL exposes cumulative energy counters. Store raw microjoule readings
     * here; compute deltas and wrap-around handling outside the producer.
     */
    struct EnergySnapshot {
        ns_t     timestamp_ns;
        uint64_t pkg_uj;   /**< Package domain, read from sysfs energy_uj. */
        uint64_t dram_uj;  /**< Optional DRAM domain, read from sysfs energy_uj. */
    };

    /**
     * @brief NVML GPU device-level sample.
     *
     * These fields describe device state, not attribution to a specific CUDA
     * kernel. Kernel-level attribution requires a later CUDA/NVTX/CUPTI phase.
     */
    struct GpuSample {
        ns_t     timestamp_ns;
        uint32_t power_mw;  /**< nvmlDeviceGetPowerUsage(), milliwatts. */
        uint32_t util_pct;  /**< nvmlDeviceGetUtilizationRates().gpu, percent. */
    };

    /** @brief Type tag for the active member of Sample. */
    enum class SampleTag : uint8_t { CPU, ENERGY, GPU };

    /**
     * @brief Fixed-size tagged union transported through the telemetry ring.
     *
     * The tag must be checked before reading a union member. Keeping this object
     * fixed-size helps the producer avoid heap allocation in the sampling loop.
     */
    struct Sample
    {
        SampleTag tag;
        union {
            CpuSample cpu;
            EnergySnapshot energy;
            GpuSample gpu;
        };
    };
    
    
    

}
