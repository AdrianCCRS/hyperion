#pragma once
#include <cstdint>
#include <ctime>

/** @file
 * @brief Telemetry metrics definitions.
 *
 * This file defines the data structures used for telemetry metrics collection and storage.
 */
namespace telemetry {
     
    // Timestamp: nanoseconds since CLOCK_MONOTONIC epoch.
    // Use clock_gettime(CLOCK_MONOTONIC, ...) — lower jitter than
    // CLOCK_REALTIME and no discontinuities from NTP adjustments.
    using ns_t = uint64_t;

    // --- CPU hardware counters (perf_event) ---
    struct CpuSample {
    ns_t     timestamp_ns;
    uint64_t instructions;    // PERF_COUNT_HW_INSTRUCTIONS
    uint64_t cycles;          // PERF_COUNT_HW_CPU_CYCLES
    uint64_t cache_references; // PERF_COUNT_HW_CACHE_REFERENCES
    uint64_t cache_misses;    // PERF_COUNT_HW_CACHE_MISSES
    // Derived (compute after capture, not during critical path):
    // double ipc()  const { return (double)instructions / cycles; }
    };

    // --- Intel RAPL energy deltas ---
    // Store raw microjoule readings; compute delta in consumer thread.
    struct EnergySnapshot {
    ns_t     timestamp_ns;
    uint64_t pkg_uj;          // Package domain  (sysfs: energy_uj)
    uint64_t dram_uj;         // DRAM domain     (sysfs: energy_uj)
    };

    // --- NVML GPU sample (optional) ---
    struct GpuSample {
    ns_t     timestamp_ns;
    uint32_t power_mw;        // nvmlDeviceGetPowerUsage() — milliwatts
    uint32_t util_pct;        // nvmlDeviceGetUtilizationRates().gpu
    };

    // Unified telemetry sample with a tag to indicate the type of data.
    enum class SampleTag : uint8_t { CPU, ENERGY, GPU };

    // A union of different sample types, tagged for identification.
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