# Modular Telemetry Subsystem

This document covers the low-overhead modular telemetry library under
`telemetry/include/telemetry` and `telemetry/src`. The standalone
`telemetry_collection.cpp` collector is kept as legacy/MVP and is not part of
this path.

## Local Unit Tests

```bash
cmake -S telemetry -B /tmp/tg-telemetry-final-build
cmake --build /tmp/tg-telemetry-final-build
ctest --test-dir /tmp/tg-telemetry-final-build --output-on-failure
```

The unit tests are intended to run without requiring RAPL, NVML, or unprivileged
`perf_event` access.

## Manual Measurement Targets

The following targets are built by CMake but are not registered in CTest because
their results depend on hardware, permissions, and system load.

```bash
/tmp/tg-telemetry-final-build/telemetry_overhead_bench
/tmp/tg-telemetry-final-build/telemetry_jitter_bench
```

Useful options:

```bash
/tmp/tg-telemetry-final-build/telemetry_overhead_bench --no-perf
/tmp/tg-telemetry-final-build/telemetry_overhead_bench --rapl-pkg /sys/class/powercap/intel-rapl/intel-rapl:0
/tmp/tg-telemetry-final-build/telemetry_jitter_bench --samples 10000 --interval-ns 1000000
```

`telemetry_jitter_bench` currently records CPU sample timestamps, so it expects
`perf_event` access. Use it on the target experiment node after confirming the
PMU permissions are correct.

Go/No-Go targets for a real experiment node:

- Overhead mean below 2 percent for the workload under study.
- Sampling interval CV below 5 percent.
- `push_retries` equal to 0 during the benchmark.

## GPU Build

Build with NVML only on nodes where both `nvml.h` and `libnvidia-ml` are
available.

```bash
cmake -S telemetry -B /tmp/tg-telemetry-gpu-build -DWITH_GPU=ON
cmake --build /tmp/tg-telemetry-gpu-build
ctest --test-dir /tmp/tg-telemetry-gpu-build --output-on-failure
```

## Hot Path Rule

The producer path must stay narrow: hardware reads, fixed-buffer parsing where
the kernel exposes text, `ring.try_push`, and absolute sleeps. Avoid logging,
file persistence, heap allocation, string construction, mutexes, and derived
metric computation in `Collector::run()` and reader `read()` methods.
