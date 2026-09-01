# Node Readiness Checker — Reference Guide

**Version**: 2.0.0  
**Last updated**: 2026-03-12

This document is the single reference for the node readiness checker tool set located under `Code/hardware/`. It covers the module architecture, data flow, JSON output specification, RAPL domain reference, and usage instructions.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Module Architecture](#2-module-architecture)
   - 2.1 [utils.py](#21-utilspy)
   - 2.2 [probes.py](#22-probespy)
   - 2.3 [capabilities.py](#23-capabilitiespy)
   - 2.4 [check_node_readiness.py](#24-check_node_readinesspy)
3. [Data Flow](#3-data-flow)
4. [Capability Model](#4-capability-model)
   - 4.1 [Three-Level Taxonomy](#41-three-level-taxonomy)
   - 4.2 [Capability Definitions](#42-capability-definitions)
   - 4.3 [Severity Classification](#43-severity-classification)
5. [JSON Output Specification](#5-json-output-specification)
   - 5.1 [Metadata](#51-metadata)
   - 5.2 [node](#52-node)
   - 5.3 [cpu](#53-cpu)
   - 5.4 [gpu](#54-gpu)
   - 5.5 [numa](#55-numa)
   - 5.6 [tools](#56-tools)
   - 5.7 [permissions](#57-permissions)
   - 5.8 [active_probes](#58-active_probes)
   - 5.9 [capabilities](#59-capabilities)
   - 5.10 [backends](#510-backends)
   - 5.11 [experiment_blockers](#511-experiment_blockers)
   - 5.12 [readiness](#512-readiness)
6. [Intel RAPL Domains](#6-intel-rapl-domains)
7. [Usage](#7-usage)
8. [Design Principles](#8-design-principles)

---

## 1. Purpose

The node readiness checker is a **read-only** diagnostic tool that validates a Linux node for telemetry and DVFS (Dynamic Voltage and Frequency Scaling) experiments. It answers the question: *"Is this node ready to collect reliable energy/performance measurements and exercise frequency control?"*

It does not change any system settings. Its output is intended for two audiences:

- **Operator** — a compact human-readable summary printed to stdout.
- **Automation** — a structured JSON report consumed by experiment orchestrators to gate dispatch and configure telemetry pipelines.

---

## 2. Module Architecture

The tool is split into four Python modules that each have a single responsibility. All modules require Python 3.9+ and run on Linux only (sysfs/procfs interfaces).

```
hardware/
├── utils.py                  # Shared constants, sysfs paths, type aliases, hardwareutils class
├── probes.py                 # Active hardware probes (effective validation)
├── capabilities.py           # Capability assessment, backends, blockers, readiness evaluation
└── check_node_readiness.py   # Hardware collectors, report assembly, CLI, human output
```

### 2.1 `utils.py`

Provides shared infrastructure used by all other modules. Nothing here executes on import.

| Component | Description |
|---|---|
| `CRITICAL`, `WARNING`, `INFO` | Severity level string constants |
| `AVAILABLE`, `UNAVAILABLE`, `DEGRADED` | Capability status string constants |
| `CPUFREQ_PATH`, `POWERCAP_PATH`, `INTEL_PSTATE_PATH`, etc. | `pathlib.Path` constants for sysfs/procfs locations |
| `LSPCI_DISPLAY_CLASSES` | PCI device class keywords used to identify display adapters |
| `NUMACTL_NODE_RE` | Compiled regex for parsing `numactl --hardware` output |
| `GpuDevice`, `VendorMap`, `CapabilityResult`, `ResourceTaxonomy` | Type aliases |
| `build_taxonomy(present, observable, controllable, reason)` | Factory for the three-level taxonomy dict |
| `hardwareutils` | Static utility class: `run_cmd`, `read_text`, `as_int`, `parse_cpu_list`, `parse_supported_clocks`, `parse_lscpu_output` |

Key sysfs paths defined in `utils.py`:

| Constant | Path |
|---|---|
| `CPUFREQ_PATH` | `/sys/devices/system/cpu/cpu0/cpufreq` |
| `CPUFREQ_GOVERNOR_PATH` | `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` |
| `POWERCAP_PATH` | `/sys/class/powercap` |
| `INTEL_PSTATE_PATH` | `/sys/devices/system/cpu/intel_pstate` |
| `PERF_PARANOID_PATH` | `/proc/sys/kernel/perf_event_paranoid` |
| `EPP_PATH` | `/sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference` |
| `SYSFS_NODE_PATH` | `/sys/devices/system/node` |

### 2.2 `probes.py`

Contains active probe functions that perform minimal read-only operations to confirm that hardware interfaces are *effectively* accessible — not merely nominally present. Imports only from `utils`.

| Function | What it tests |
|---|---|
| `probe_perf(tools)` | Reads `perf_event_paranoid` and executes `perf stat -e cycles -- true` to confirm perf can collect HW counters under the current user/permission level |
| `probe_cpu_frequency()` | Reads `scaling_cur_freq` from sysfs to confirm the cpufreq interface returns live data |
| `probe_gpu_metrics(gpu)` | Runs `nvidia-smi --query-gpu=temperature.gpu,power.draw` to confirm NVML returns real metrics |
| `collect_active_probes(tools, gpu, cpu)` | Aggregates all probes into a single dict; `rapl_read` is taken directly from the embedded probe stored in `cpu.rapl` |

### 2.3 `capabilities.py`

The assessment engine. Takes the collected hardware data and probe results and produces structured, actionable findings for each of the nine capabilities. Imports constants and types from `utils`.

| Function | Description |
|---|---|
| `_cap(status, severity, evidence, reason, recommendation)` | Private helper that builds a single `CapabilityResult` dict |
| `assess_capabilities(cpu, gpu, numa, tools, permissions, probes)` | Returns a dict of nine `CapabilityResult` entries, one per capability |
| `build_backends(capabilities, cpu, gpu)` | Derives available telemetry/control backend flags from capability results |
| `build_experiment_blockers(capabilities)` | Extracts only the `critical`-severity issues into a flat list for easy gate checks |
| `evaluate_readiness(capabilities)` | Emits eight boolean readiness flags and a severity-sorted flat issue list |

### 2.4 `check_node_readiness.py`

The entry point and orchestrator. Contains all hardware *collectors* (functions that talk to lscpu, sysfs, nvidia-smi, lspci, numactl), assembles the final report dict, and handles CLI arguments and human output.

| Component | Description |
|---|---|
| `collect_node_info()` | hostname, OS, kernel, architecture |
| `collect_cpufreq_info()` | cpufreq driver, governors, frequencies, scaling method |
| `_collect_intel_pstate_info()` | Intel P-state driver status, HWP, EPP, control model, driver limitations |
| `collect_rapl_info()` | RAPL domain discovery, readability check, embedded active probe |
| `collect_cpu_info()` | Full CPU section: topology + cpufreq + RAPL + taxonomy |
| `_collect_nvidia_devices()` | NVIDIA GPU discovery via nvidia-smi with clock tables |
| `_collect_lspci_devices(nvidia_already_found)` | PCI bus scan for display adapters via lspci |
| `collect_gpu_info()` | Full GPU section: vendor-grouped devices + taxonomy |
| `collect_numa_info()` | NUMA topology via numactl or sysfs fallback |
| `collect_tools_info()` | Tool presence checks (perf, cpupower, numactl, nvidia-smi) |
| `collect_permissions_info(cpu, tools, gpu)` | Permission checks: RAPL readability, cpufreq writability, GPU clock control |
| `build_report()` | Calls all collectors then probes → capabilities → backends → blockers → readiness |
| `print_summary(report)` | Human-readable summary with per-capability status, blockers, and readiness flags |
| `main()` | CLI entry point; writes JSON and optionally prints summary |

---

## 3. Data Flow

```
                         check_node_readiness.py
                         ┌────────────────────────────────────────────┐
  Linux kernel/sysfs     │                                            │
  lscpu, lspci, numactl  │  collect_node_info()                       │
  nvidia-smi             │  collect_cpu_info()   ──► cpu dict         │
  /proc, /sys            │    collect_cpufreq_info()                  │
                         │    _collect_intel_pstate_info()            │
                         │    collect_rapl_info()                     │
                         │  collect_gpu_info()   ──► gpu dict         │
                         │    _collect_nvidia_devices()               │
                         │    _collect_lspci_devices()                │
                         │  collect_numa_info()  ──► numa dict        │
                         │  collect_tools_info() ──► tools dict       │
                         │  collect_permissions_info() ──► perms dict │
                         └──────────────────────┬─────────────────────┘
                                                │
                                                ▼
                                          probes.py
                              collect_active_probes(tools, gpu, cpu)
                              ┌───────────────────────────────────┐
                              │  probe_perf()                     │
                              │  probe_cpu_frequency()            │
                              │  probe_gpu_metrics()              │
                              │  [rapl_read from cpu.rapl.probe]  │
                              └──────────────┬────────────────────┘
                                             │
                                             ▼
                                       capabilities.py
                              assess_capabilities(cpu, gpu, numa,
                                                  tools, perms, probes)
                              ┌────────────────────────────────────┐
                              │  → capabilities dict (9 entries)   │
                              │  build_backends()                  │
                              │  build_experiment_blockers()       │
                              │  evaluate_readiness()              │
                              └──────────────┬─────────────────────┘
                                             │
                                             ▼
                              build_report() assembles final JSON
                              print_summary() → stdout
                              _write_report() → .json file
```

---

## 4. Capability Model

### 4.1 Three-Level Taxonomy

Every hardware resource (CPU, GPU) carries a `taxonomy` object with three independent boolean levels:

```json
{
  "present": true,
  "observable": true,
  "controllable": false,
  "reason": "cpufreq governor sysfs not writable by current user"
}
```

| Level | Meaning | Example failure |
|---|---|---|
| `present` | The resource physically exists and was detected | GPU not found by lspci or nvidia-smi |
| `observable` | Metrics can be read from the resource | RAPL domains found but energy_uj not readable |
| `controllable` | Parameters can be modified | cpufreq driver loaded but governor file not writable |

A resource may be present but not observable (GPU detected via lspci but no nvidia-smi), or observable but not controllable (RAPL readable, cpufreq not writable).

### 4.2 Capability Definitions

Nine capabilities are assessed independently:

| Capability | What is checked | Default severity if unavailable |
|---|---|---|
| `cpu_discovery` | `lscpu` returns valid topology with > 0 logical CPUs | critical |
| `cpu_telemetry` | `perf` binary found **and** `perf stat -e cycles -- true` succeeds **and** `scaling_cur_freq` readable | critical |
| `energy_cpu` | RAPL domains found **and** `energy_uj` readable **and** active probe returns a numeric counter value | critical |
| `cpu_control` | cpufreq driver loaded and governor sysfs writable; `degraded` if HWP/driver limitations are present | warning |
| `gpu_discovery` | At least one GPU found via nvidia-smi or lspci | critical |
| `gpu_telemetry` | nvidia-smi available and temperature/power query succeeds | warning / critical |
| `gpu_control` | nvidia-smi supported clocks queryable with no N/A in output | warning |
| `topology_awareness` | NUMA nodes detected; `degraded` when numactl is missing | info / warning |
| `full_pipeline` | All telemetry capabilities `available` **and** all control capabilities `available` | critical / warning |

Each capability result contains:

| Field | Type | Description |
|---|---|---|
| `status` | string | `available`, `unavailable`, or `degraded` |
| `severity` | string | `critical`, `warning`, or `info` |
| `evidence` | string | What was checked and what was found |
| `reason` | string \| null | Why the capability is not fully available |
| `recommendation` | string \| null | Actionable fix for the identified issue |

### 4.3 Severity Classification

| Level | Meaning for experiments |
|---|---|
| `critical` | Prevents experiment execution entirely |
| `warning` | Reduces measurement coverage or precision |
| `info` | Non-blocking observation; recorded for reproducibility |

---

## 5. JSON Output Specification

All fields produced by `build_report()` and written to the output JSON file.

### 5.1 Metadata

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Version of the JSON schema contract — `"2.0.0"` |
| `tool_version` | string | Version of `check_node_readiness.py` — `"2.0.0"` |
| `generated_at` | string | UTC ISO 8601 timestamp, e.g. `"2026-03-12T14:51:00.000000Z"` |

### 5.2 `node`

Basic node identification collected without subprocess calls.

| Field | Type | Source |
|---|---|---|
| `hostname` | string | `socket.gethostname()` |
| `os` | string | `platform.system()` — e.g. `"Linux"` |
| `kernel` | string | `platform.release()` — e.g. `"6.18.16-200.fc43.x86_64"` |
| `architecture` | string | `platform.machine()` — e.g. `"x86_64"` |

### 5.3 `cpu`

#### Main fields

| Field | Type | Source |
|---|---|---|
| `vendor` | string | `lscpu` → `Vendor ID` field |
| `model_name` | string | `lscpu` → `Model name` field |
| `logical_cpus` | integer | `lscpu` → `CPU(s)`, fallback `os.cpu_count()` |
| `physical_cores` | integer | Calculated: `cores_per_socket × sockets` |
| `threads_per_core` | integer | `lscpu` → `Thread(s) per core` |
| `sockets` | integer | `lscpu` → `Socket(s)` |

#### `cpu.taxonomy`

See [§ 4.1 Three-Level Taxonomy](#41-three-level-taxonomy). `controllable` requires the cpufreq governor sysfs file to be writable by the current process.

#### `cpu.cpufreq`

| Field | Type | Source |
|---|---|---|
| `driver` | string | sysfs `scaling_driver` — e.g. `"intel_pstate"`, `"acpi-cpufreq"` |
| `governor_current` | string | sysfs `scaling_governor` — e.g. `"performance"`, `"powersave"` |
| `governors_available` | string[] | sysfs `scaling_available_governors` |
| `min_khz` | string | sysfs `cpuinfo_min_freq` (hardware lower bound) |
| `max_khz` | string | sysfs `cpuinfo_max_freq` (includes turbo) |
| `base_frequency_khz` | string | sysfs `base_frequency` (intel_pstate only) |
| `available_frequencies_khz` | string[] | sysfs `scaling_available_frequencies` (non-empty only on legacy `acpi-cpufreq`) |
| `scaling_method` | string | Derived: `"discrete"`, `"range"`, or `"unknown"` |
| `intel_pstate_status` | string | sysfs `/sys/devices/system/cpu/intel_pstate/status` — `"active"`, `"passive"`, `"off"`, or `""` |
| `hwp_enabled` | boolean | Detected via `hwp_dynamic_boost` existence or `status == "active"` |
| `epp_available` | boolean | `energy_performance_preference` sysfs file exists |
| `control_model` | string | See table below |
| `driver_limitations` | string[] | Human-readable list of architecture-specific constraints |

**Control model values:**

| Value | Meaning |
|---|---|
| `hwp_range` | HWP active — hardware autonomously selects P-state; OS hints are advisory |
| `pstate_range` | `intel_pstate` passive — OS controls within a continuous frequency range |
| `discrete` | Legacy driver with an explicit list of fixed frequency steps |
| `unknown` | Control model cannot be determined |

**Note on HWP:** When `hwp_enabled` is `true`, the CPU microcode autonomously selects the operating frequency within the OS-configured range. The `cpu_control` capability is assessed as `degraded` in this case, and `driver_limitations` explains the constraint. Setting a specific frequency via the governor does not guarantee that frequency will be maintained.

#### `cpu.rapl`

| Field | Type | Source |
|---|---|---|
| `available` | boolean | At least one `intel-rapl*` directory found under `/sys/class/powercap/` |
| `domains` | string[] | Enumerated domain labels, format: `"intel-rapl:0 (package-0)"` |
| `readable` | boolean | At least one `energy_uj` file passes `os.access(R_OK)` |

### 5.4 `gpu`

#### Summary fields

| Field | Type | Description |
|---|---|---|
| `nvidia_present` | boolean | At least one NVIDIA GPU found via any source |
| `device_count` | integer | Total unique GPU count across all vendors |

#### `gpu.taxonomy`

See [§ 4.1 Three-Level Taxonomy](#41-three-level-taxonomy). `observable` requires nvidia-smi to be present and responsive. `controllable` requires supported clocks to be queryable without `N/A`.

#### `gpu.devices[]`

Flat list of all GPU devices, normalized to a uniform shape regardless of detection source.

| Field | Type | Source |
|---|---|---|
| `vendor` | string | `"nvidia"` or `"intel"` |
| `index` | integer \| null | nvidia-smi `--query-gpu=index`; null for lspci-only entries |
| `name` | string | nvidia-smi product name or lspci description |
| `driver_version` | string | nvidia-smi `--query-gpu=driver_version`; empty for non-NVIDIA |
| `vbios_version` | string | nvidia-smi `--query-gpu=vbios_version`; empty for non-NVIDIA |
| `memory_total_mb` | string | nvidia-smi `--query-gpu=memory.total`; empty for non-NVIDIA |
| `supported_graphics_clocks_mhz` | string[] | nvidia-smi `-i <idx> -q -d SUPPORTED_CLOCKS` |
| `supported_memory_clocks_mhz` | string[] | nvidia-smi `-i <idx> -q -d SUPPORTED_CLOCKS` |
| `bus_id` | string | PCI bus address from lspci (may be absent for nvidia-smi-only entries) |
| `source` | string | `"lspci"` for lspci-detected devices; omitted for nvidia-smi sources |

#### `gpu.by_vendor`

Same device records grouped by vendor key: `{"nvidia": [...], "intel": [...]}`.

**Detection strategy:**
1. NVIDIA GPUs are queried first via `nvidia-smi` for rich metadata (clock tables, driver version, memory size).
2. `lspci` is used as a supplementary source to detect NVIDIA GPUs when `nvidia-smi` is absent, and to discover Intel integrated graphics.
3. If `nvidia-smi` already found NVIDIA devices, lspci NVIDIA entries are skipped to prevent duplicates.

### 5.5 `numa`

| Field | Type | Description |
|---|---|---|
| `available` | boolean | At least one NUMA node discovered |
| `nodes` | object | Map of string node IDs (`"0"`, `"1"`) to sorted integer arrays of CPU indices |

**Detection strategy:**
1. Primary: parse `numactl --hardware` with regex `^node\s+(\d+)\s+cpus:\s+(.+)$`
2. Fallback: read `/sys/devices/system/node/node*/cpulist` directly

CPU lists in compact Linux range notation (e.g., `"0-3,8"`) are expanded to integer arrays.

### 5.6 `tools`

Boolean presence check for each required tool via `shutil.which`.

| Field | Required for |
|---|---|
| `perf` | CPU hardware counter telemetry |
| `cpupower` | CPU governor management (optional) |
| `numactl` | NUMA-aware process pinning (optional) |
| `nvidia_smi` | NVIDIA GPU discovery and clock control |

### 5.7 `permissions`

| Field | Type | Source |
|---|---|---|
| `is_root` | boolean | `os.geteuid() == 0` |
| `rapl_readable` | boolean | Derived from `cpu.rapl.readable` |
| `cpufreq_writable` | boolean | `os.access(scaling_governor, W_OK)` |
| `gpu_clock_control_available` | boolean | Derived from `gpu.taxonomy.controllable` |

### 5.8 `active_probes`

Active probes confirm effective hardware access beyond static file/binary presence checks.

#### `active_probes.perf`

| Field | Type | Description |
|---|---|---|
| `event_paranoid_level` | integer \| null | Value of `/proc/sys/kernel/perf_event_paranoid` |
| `accessible` | boolean | `perf stat -e cycles -- true` returned exit code 0 |
| `tested_with` | string | Exact command executed |
| `note` | string | Diagnostic explanation |

#### `active_probes.rapl_read`

| Field | Type | Description |
|---|---|---|
| `success` | boolean | Numeric value successfully read from `energy_uj` |
| `sample_value_uj` | string | Sample energy counter reading in microjoules |
| `domain_tested` | string | RAPL domain directory used (e.g. `"intel-rapl:0"`) |

#### `active_probes.cpu_frequency`

| Field | Type | Description |
|---|---|---|
| `success` | boolean | `scaling_cur_freq` returned a numeric value |
| `current_khz` | string | Current CPU frequency in kHz |

#### `active_probes.gpu_metrics`

| Field | Type | Description |
|---|---|---|
| `success` | boolean | nvidia-smi metric query returned valid data |
| `temperature_c` | string | GPU temperature in Celsius |
| `power_draw_w` | string | GPU power draw in Watts |

### 5.9 `capabilities`

Dict of nine capability results. Each entry has the structure described in [§ 4.2](#42-capability-definitions).

### 5.10 `backends`

Operational metadata for downstream pipeline stages.

#### `backends.telemetry`

| Field | Type | Description |
|---|---|---|
| `cpu_perf` | boolean | perf hardware counters usable |
| `cpu_rapl` | boolean | RAPL energy counters readable |
| `gpu_nvml` | boolean | NVIDIA GPU metrics accessible |

#### `backends.control`

| Field | Type | Description |
|---|---|---|
| `cpu_cpufreq` | boolean | CPU frequency governor controllable |
| `gpu_clocks` | boolean | GPU application clocks settable |

#### `backends.observable_frequencies`

| Field | Type | Description |
|---|---|---|
| `method` | string | `"range"`, `"discrete"`, or `"unknown"` |
| `min_khz` | string | Minimum frequency in kHz |
| `max_khz` | string | Maximum frequency in kHz |
| `base_frequency_khz` | string | Base frequency in kHz (intel_pstate; may be empty) |
| `discrete_steps` | string[] | Discrete frequency steps (non-empty only for `acpi-cpufreq`) |

#### `backends.energy_domains`

String array of accessible RAPL domain labels copied from `cpu.rapl.domains`.

### 5.11 `experiment_blockers`

Array of critical-severity issues that block experiment execution. Each entry:

| Field | Type | Description |
|---|---|---|
| `severity` | string | Always `"critical"` |
| `capability` | string | Capability name that failed |
| `message` | string | Evidence text from the capability result |
| `recommendation` | string | Actionable fix |

An empty array means the node has no blocking issues.

### 5.12 `readiness`

#### Per-capability flags

| Field | Description |
|---|---|
| `cpu_discovery_ready` | CPU topology successfully detected |
| `cpu_telemetry_ready` | perf hardware counters accessible |
| `gpu_discovery_ready` | At least one GPU detected |
| `gpu_telemetry_ready` | GPU metrics queryable |
| `energy_cpu_ready` | RAPL counters readable and verified |
| `cpu_control_ready` | CPU frequency control available |
| `gpu_control_ready` | GPU clock control available |
| `full_pipeline_ready` | All telemetry AND control capabilities available |

#### `readiness.issues`

All non-`available` capability findings, sorted by severity (critical → warning → info):

| Field | Type | Description |
|---|---|---|
| `severity` | string | `critical`, `warning`, or `info` |
| `capability` | string | Capability name |
| `message` | string | Evidence: what was found |
| `reason` | string | Why it is an issue |
| `recommendation` | string | How to fix it |

---

## 6. Intel RAPL Domains

Intel RAPL (Running Average Power Limit) exposes energy counters for CPU subsystems through `/sys/class/powercap/intel-rapl:*`. Each directory represents an energy domain.

### Primary domains

| Domain | Description | Experimental use |
|---|---|---|
| `intel-rapl:0` | **Package** — total energy consumed by the CPU socket (cores, cache, memory controller, uncore) | Main metric for Energy–Delay Product (EDP) |
| `intel-rapl:0:0` | **PP0 / Core** — energy consumed only by the CPU execution cores | Direct DVFS impact on compute-bound workloads |
| `intel-rapl:0:1` | **DRAM** — energy consumed by the memory subsystem | Memory-bound workload analysis |
| `intel-rapl:1` | Second package domain (multi-socket or psys supplement) | Generally irrelevant on single-socket systems |

### MMIO domains

| Domain | Description |
|---|---|
| `intel-rapl-mmio` | Alternative RAPL access via Memory-Mapped I/O instead of MSR registers |
| `intel-rapl-mmio:0` | MMIO sub-domain equivalent to a package domain |

MMIO domains measure the same energy as classic RAPL but through a different kernel access path.

### Reading energy counters

Energy counters are read from:

```
/sys/class/powercap/intel-rapl:*/energy_uj
```

Unit: microjoules (µJ). To compute energy consumed in an interval:

$$E = \frac{energy\_uj_{t_2} - energy\_uj_{t_1}}{10^6} \quad \text{[joules]}$$

**Recommended domains for DVFS energy evaluation:**

- `intel-rapl:0` (package) — total CPU energy
- `intel-rapl:0:1` (DRAM) — memory subsystem energy for memory-bound phase analysis

### Permissions

Reading `energy_uj` requires either root privileges or a udev rule granting read access. The `energy_cpu` capability result reports whether this interface is effectively accessible for the current user.

To grant read access without running as root:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

---

## 7. Usage

### Basic run

Collect a full report and print the human-readable summary:

```bash
python3 hardware/check_node_readiness.py --output-json report.json
```

### JSON-only mode

Suppress the human-readable summary (suitable for pipeline invocation):

```bash
python3 hardware/check_node_readiness.py --output-json report.json --quiet
```

### Run as root for full control assessment

Non-root runs reveal realistic permission gaps and are the recommended first step. Run as root only to confirm controllability:

```bash
# First: check realistic permissions
python3 hardware/check_node_readiness.py --output-json report_user.json

# Then: check full potential
sudo python3 hardware/check_node_readiness.py --output-json report_root.json
```

### Automatic output filename

When `--output-json` is omitted, the report is written to:

```
node_readiness_report_<UTC-timestamp>.json
```

### Example human-readable output

```
=== Node Readiness Summary ===
Host: cluster-node-01
CPU:  GenuineIntel | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
GPU devices: 1 (NVIDIA: 0, Intel: 1)
RAPL readable:     False

--- Capabilities ---
  [i] cpu_discovery: available
  [X] cpu_telemetry: unavailable
      Reason: Hardware performance counter telemetry unavailable
      Fix: Install linux-tools-$(uname -r) or perf package
  [X] energy_cpu: unavailable
      Reason: Insufficient permissions to read RAPL energy counters
      Fix: Grant read access: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
  ...

--- Experiment Blockers ---
  [X] cpu_telemetry: perf binary not found in PATH
      Fix: Install linux-tools-$(uname -r) or perf package

--- Readiness Flags ---
  cpu_discovery_ready: True
  cpu_telemetry_ready: False
  ...
```

### Consuming the report programmatically

```python
import json

report = json.load(open("report.json"))

# Gate experiment dispatch
if report["experiment_blockers"]:
    for b in report["experiment_blockers"]:
        print(f"BLOCKED: {b['capability']} — {b['recommendation']}")
    exit(1)

# Configure telemetry pipeline
backends = report["backends"]["telemetry"]
if backends["cpu_rapl"]:
    rapl_domains = report["backends"]["energy_domains"]

# Check specific capability
cap = report["capabilities"]["cpu_control"]
if cap["status"] == "degraded":
    print(f"HWP active: {cap['reason']}")
```

---

## 8. Design Principles

### Read-only operation

The script never modifies kernel or sysfs state. Active probes execute minimal, non-destructive operations: `perf stat -- true` (measures the `true` binary), reading a single sysfs counter value, or querying a single GPU metric line.

### Static + effective validation

**Static checks** verify that a file path exists or a binary is in PATH. **Active probes** confirm that the interface actually works under the current user and permission context. This eliminates false positives where, for example, `perf` is installed but blocked by `perf_event_paranoid=3`.

### Granular over binary

Readiness is not a single boolean. Each capability is assessed independently so that a missing GPU does not obscure a perfectly functioning CPU telemetry stack, and vice versa.

### Architecture-aware detection

Intel HWP (Hardware P-states / Speed Shift) is detected and surfaced as a `degraded` status for `cpu_control` with an explicit recommendation. The absence of `scaling_available_frequencies` on `intel_pstate` systems is recognized as range-based control, not a failure.

### Separation of concerns across modules

| Concern | Module |
|---|---|
| Shared infrastructure | `utils.py` |
| Effective validation | `probes.py` |
| Capability assessment and scoring | `capabilities.py` |
| Hardware collection, assembly, output | `check_node_readiness.py` |

### Best practices for experiment workflows

- Run as a regular user first to capture realistic permission gaps.
- Re-run with `sudo` only to confirm the full controllability potential.
- Store the JSON report alongside experiment artifacts for reproducibility.
- Compare reports across nodes before scheduling multi-node experiments.
- Use `experiment_blockers` to gate automated experiment dispatch.
- Use `backends` to auto-configure telemetry and control pipelines.
