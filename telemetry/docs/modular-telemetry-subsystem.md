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
/tmp/tg-telemetry-final-build/telemetry_kernel_workload
/tmp/tg-telemetry-final-build/telemetry_kernel_launcher
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
- `time_running/time_enabled` close to 1.0 when perf counters are enabled.

## Multithreaded Kernel Runner

The multithreaded experiment path uses a launcher plus a child workload process.
This keeps the collector and consumer outside the measured workload and prepares
the CPU path for thread pools and dynamic worker creation.

```bash
/tmp/tg-telemetry-final-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 100000000 \
  --iterations 20 \
  --warmup 2 \
  --threads 4 \
  --workload-cpus 2,3,4,5 \
  --collector-cpu 6 \
  --consumer-cpu 7 \
  --cgroup-path /sys/fs/cgroup/<delegated-cgroup> \
  --interval-ns 1000000 \
  --output-dir runs \
  --run-id stream_mt_001
```

The launcher runs two child executions with identical parameters:

- baseline: workload child without active collector;
- telemetry: workload child with collector, consumer and export enabled.

The output directory is `--output-dir/--run-id` and contains:

- `samples.csv`: CPU/RAPL/GPU-shaped rows, although GPU is intentionally not
  used in this path yet;
- `metadata.json`: experiment parameters, pinning, overhead, jitter,
  `push_retries` and perf multiplexing ratio;
- `summary.txt`: compact human-readable result.

`--cgroup-path` must point to a pre-created/delegated cgroup. The launcher does
not create global cgroup hierarchy state or assume sudo. `--workload-cpus` is
mandatory when perf is enabled because cgroup `perf_event_open` is per CPU.

For local smoke checks without perf/cgroup access:

```bash
/tmp/tg-telemetry-final-build/telemetry_kernel_launcher \
  --kernel stream_triad \
  --size 10000 \
  --iterations 2 \
  --warmup 1 \
  --threads 2 \
  --workload-cpus 0,1 \
  --collector-cpu -1 \
  --consumer-cpu -1 \
  --no-perf \
  --output-dir /tmp/tg-telemetry-launcher-smoke \
  --run-id smoke_no_perf
```

Important implementation constraint: do not replace cgroup measurement with
`perf_event` inheritance as the main multithreaded strategy. `inherit=1` does
not cover already-existing child tasks and has documented restrictions with
some read formats, including grouped reads. The current cgroup reader keeps that
risk explicit by measuring the delegated cgroup across the requested CPUs.

GPU kernels are deliberately postponed for this runner. NVML is device-level
telemetry and cannot by itself attribute samples to an asynchronous CUDA kernel;
future GPU support should add explicit CUDA synchronization and likely CUDA
events, NVTX or CUPTI around workload phases.

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
