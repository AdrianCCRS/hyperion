## check_node_readiness.py

`hardware/check_node_readiness.py` performs a read-only validation of a Linux node for telemetry and DVFS experiments.

It uses `hardware/schema.json` as the base output contract and adds `gpu.by_vendor` to preserve vendor-grouped discovery details for NVIDIA and Intel GPUs.

### What it collects

- Node metadata: hostname, OS, kernel, architecture.
- CPU inventory: vendor, model, core/thread topology.
- CPU control/telemetry prerequisites:
  - cpufreq driver, governors, min/max/available frequencies.
  - Intel RAPL availability/readability and domains.
- GPU inventory:
  - Vendor buckets with this internal organization:
    - `{'nvidia': [], 'intel': []}`
  - Flattened list in `gpu.devices`.
- NUMA topology via `numactl --hardware` (or sysfs fallback).
- Tool checks: `perf`, `cpupower`, `numactl`, `nvidia-smi`.
- Permission checks: root status, RAPL readability, cpufreq writability, GPU clock-control availability.

### Readiness policy

`readiness.ready_for_telemetry` is `true` when all are satisfied:

- `perf` available.
- Intel RAPL readable.
- At least one GPU detected.

`readiness.ready_for_control` is `true` when telemetry is ready and:

- CPU frequency governor is writable.
- GPU clock control capability is available.

`readiness.blocking_issues` contains unmet hard requirements. `readiness.warnings` contains non-blocking limitations.

### Usage

Run with summary + JSON output:

```bash
python3 hardware/check_node_readiness.py --output-json hardware/node_readiness_report.json
```

Run in JSON-only mode:

```bash
python3 hardware/check_node_readiness.py --output-json hardware/node_readiness_report.json --quiet
```

### Output notes

- Timestamps are UTC ISO-8601 (`generated_at`).
- The script is non-intrusive and does not change kernel/sysfs settings.
- `gpu.by_vendor` is included for easier vendor-specific workflows while keeping schema base fields (`nvidia_present`, `device_count`, `devices`).

### Best practices

- Execute as a normal user first to detect realistic permission gaps.
- Re-run with elevated privileges only when necessary to confirm controllability (`cpufreq_writable`, RAPL access).
- Store each report with experiment artifacts for reproducibility.
- Compare reports across nodes before scheduling multi-node experiments.
