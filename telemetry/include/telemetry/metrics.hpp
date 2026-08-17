#pragma once
#include <cstddef>
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
     * @brief Upper bound on how many delegated CPUs CpuSample can carry a
     * per-CPU cpufreq reading for.
     *
     * ARC-142: every real manifest in this project delegates 4-8 CPUs; 16
     * leaves comfortable headroom without making CpuSample's fixed layout
     * (see file-level comment: no dynamic allocation across the SPSC ring)
     * unreasonably large. A campaign that delegates more CPUs than this
     * simply stops filling additional slots (collector.cpp caps the loop),
     * it does not corrupt memory or crash.
     */
    constexpr size_t kMaxScalingCurFreqCpus = 16;

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
        /* ARC-97: raw FP_ARITH_INST_RETIRED sub-events (event=0xC7), double
         * precision only, Ice Lake-SP only. Each represents "computations",
         * not instructions -- weight by elements-per-instruction downstream
         * (1/2/4/8) to get a flops total, never sum these raw. Validated
         * empirically against dgemm_bench's analytical 2*iterations*n^3
         * (0.30% error, no multiplexing at the full 10-counter budget) and
         * against a memory-bound NPB kernel (7.48% error, explained by the
         * kernel's own self-timed window excluding its verification phase).
         * See docs/libro/main.tex Fase 1 for the full validation record. */
        uint64_t fp_scalar_double;        /**< Raw FP_ARITH_INST_RETIRED.SCALAR_DOUBLE (0xC7, umask=0x01). 1 flop/count. */
        uint64_t fp_128b_packed_double;   /**< Raw FP_ARITH_INST_RETIRED.128B_PACKED_DOUBLE (0xC7, umask=0x04). 2 flops/count. */
        uint64_t fp_256b_packed_double;   /**< Raw FP_ARITH_INST_RETIRED.256B_PACKED_DOUBLE (0xC7, umask=0x10). 4 flops/count. */
        uint64_t fp_512b_packed_double;   /**< Raw FP_ARITH_INST_RETIRED.512B_PACKED_DOUBLE (0xC7, umask=0x40). 8 flops/count. */
        uint64_t time_enabled_ns;   /**< Perf enabled time for multiplexing diagnostics. */
        uint64_t time_running_ns;   /**< Perf running time for multiplexing diagnostics. */
        /* ARC-135: cpufreq scaling_cur_freq for one representative delegated
         * CPU, read from the SAME producer tick as the counters above --
         * previously this was a single post-hoc Python read taken AFTER the
         * workload process had already exited (orchestrator/campaign.py),
         * which does not confirm the clock actually held during execution
         * (found not to correlate with the requested level at all, e.g. F4
         * -- 0.8GHz floor -- reading above F0's 3.6GHz ceiling in real
         * campaign data). 0 means "not sampled" (reader disabled/unavailable
         * for this run), never a real 0kHz reading. */
        uint64_t scaling_cur_freq_khz;
        /* ARC-142: same reading as scaling_cur_freq_khz above, but for
         * EVERY delegated CPU, not just the representative one -- pacca's
         * cpufreq domain is per-core (unlike felix's per-socket domain), so
         * the other delegated CPUs can diverge from CPU0 under Turbo/HWP
         * without this array, which the scalar field alone could never
         * reveal (confirmed a real risk after ARC-136 found Turbo ignoring
         * a frequency lock at all). scaling_cur_freq_khz_per_cpu[0] always
         * equals scaling_cur_freq_khz -- same reading, not a second sample.
         * count==0 means "not sampled" for every slot (reader disabled),
         * same convention as the scalar field's 0.
         *
         * No in-class initializers here (unlike a normal struct) -- CpuSample
         * is a member of the Sample union below, which requires every member
         * to stay a trivial type; sample_cpu_freq() in collector.cpp always
         * sets these explicitly before a sample is pushed, never relying on
         * a default. */
        uint32_t scaling_cur_freq_khz_count;
        uint64_t scaling_cur_freq_khz_per_cpu[kMaxScalingCurFreqCpus];
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
     * @brief Uncore memory-controller (iMC) sample, ALREADY a per-interval delta.
     *
     * ARC-119: unlike EnergySnapshot (a raw cumulative register that needs
     * differencing downstream), these fields are the counts summed across
     * every uncore_imc PMU box (system-wide/socket-scope events, pid=-1 --
     * see UncoreReader) DURING the `perf stat -I` interval ending at
     * timestamp_ns -- `perf stat -I` documents its own output as "count
     * deltas", not a running total. Never subtract two consecutive
     * UncoreSnapshot readings from each other: each one already stands
     * alone as the traffic for its own interval (including the first one,
     * which covers [run start, first timestamp] -- there is no
     * "first sample has no predecessor" case here, unlike CpuSample/
     * EnergySnapshot). Each count represents one DRAM column-address-strobe
     * transaction (one 64-byte cache line, the standard Intel iMC
     * convention); the bytes conversion itself still happens downstream.
     */
    struct UncoreSnapshot {
        ns_t     timestamp_ns;
        uint64_t cas_count_read_interval;   /**< Sum of CAS_COUNT_READ across all uncore_imc boxes, this interval only. */
        uint64_t cas_count_write_interval;  /**< Sum of CAS_COUNT_WRITE across all uncore_imc boxes, this interval only. */
        /**
         * ARC-120: `perf stat` never exits just because every event it
         * asked for came back "<not counted>"/"<not supported>" (confirmed
         * on pacca: the CAP_PERFMON gap of ARC-117/118 makes every single
         * line invalid, yet the process stays alive indefinitely) -- so
         * is_open() alone cannot tell "genuinely zero traffic this
         * interval" apart from "could not count anything at all". False
         * here means at least one of the terms in this interval failed;
         * the two counts above are the sum of whatever DID parse (0 if
         * none did), never a sentinel -- consumers must check this flag
         * before trusting a 0 as a real zero.
         */
        bool interval_valid;
    };

    /**
     * @brief NVML GPU device-level sample.
     *
     * These fields describe device state, not attribution to a specific CUDA
     * kernel. Kernel-level attribution requires a later CUDA/NVTX/CUPTI phase.
     */
    struct GpuSample {
        ns_t     timestamp_ns;
        uint32_t power_mw;       /**< nvmlDeviceGetPowerUsage(), milliwatts. */
        uint32_t util_pct;       /**< nvmlDeviceGetUtilizationRates().gpu, percent. */
        // ARC-94 (segunda ronda): nvmlUtilization_t trae tanto .gpu como
        // .memory -- solo se conservaba .gpu. Sin utilizacion de memoria,
        // un kernel con actividad puramente de memoria (alto trafico,
        // bajo uso de SM) podia parecer "ocioso" mirando solo util_pct.
        uint32_t mem_util_pct;  /**< nvmlDeviceGetUtilizationRates().memory, percent. */
        // ARC-94: sin estas tres, no era posible confirmar que un nivel
        // DVFS de GPU realmente se mantuvo durante la corrida (sin reloj
        // SM observado), ni calcular EDP de GPU (sin energia acumulada),
        // ni detectar contaminacion termica (sin temperatura). Todas
        // opcionales por diseno (0 si NVML no las reporta en este
        // driver/GPU) -- nunca bloquean una lectura de power/util ya
        // valida.
        uint32_t sm_clock_mhz;   /**< nvmlDeviceGetClockInfo(NVML_CLOCK_SM), MHz. 0 si no disponible. */
        uint64_t energy_mj;      /**< nvmlDeviceGetTotalEnergyConsumption(), mJ acumulados desde que cargo el driver (requiere delta downstream, igual que EnergySnapshot). 0 si no disponible. */
        uint32_t temperature_c;  /**< nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU), grados Celsius. 0 si no disponible. */
    };

    /** @brief Type tag for the active member of Sample. */
    enum class SampleTag : uint8_t { CPU, ENERGY, GPU, UNCORE };

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
            UncoreSnapshot uncore;
        };
    };
    
    
    

}
