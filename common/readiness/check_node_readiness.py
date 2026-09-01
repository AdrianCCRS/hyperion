#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Node readiness checker for telemetry and DVFS experiments.

- Hyperion

This script is intentionally read-only: it detects node capabilities,
permissions, and tooling without changing system settings.

The output models readiness across eight granular capabilities with a
three-level taxonomy (presence / observability / controllability) for
every hardware resource.  Each finding includes evidence, failure cause,
severity (critical / warning / info), and an actionable recommendation.

Output is emitted as JSON following ``hardware/schema.json`` as the base
contract.

Typical usage::

    python3 hardware/check_node_readiness.py \\
        --output-json hardware/node_readiness_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Any, Dict, List

from capabilities import (
    assess_capabilities,
    build_backends,
    build_experiment_blockers,
    evaluate_readiness,
)
from probes import collect_active_probes
from utils import (
    CPUFREQ_GOVERNOR_PATH,
    CPUFREQ_PATH,
    CRITICAL,
    EPP_PATH,
    INTEL_PSTATE_PATH,
    LSPCI_DISPLAY_CLASSES,
    NUMACTL_NODE_RE,
    POWERCAP_PATH,
    SYSFS_NODE_PATH,
    WARNING,
    GpuDevice,
    VendorMap,
    build_taxonomy,
    hardwareutils,
)

__version__ = "2.0.0"
__schema_version__ = "2.0.0"

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default path written by main() when --output-json is not supplied.
_DEFAULT_OUTPUT_NAME = "node_readiness_report"

#: Severity symbol map for human-readable output.
_SEVERITY_SYMBOL = {CRITICAL: "[X]", WARNING: "[!]", "info": "[i]"}


# ---------------------------------------------------------------------------
# Node identification
# ---------------------------------------------------------------------------


def collect_node_info() -> Dict[str, str]:
    """Collect basic node identification fields.

    Returns:
        ``hostname``, ``os``, ``kernel``, ``architecture``.
    """
    return {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
    }


# ---------------------------------------------------------------------------
# CPU subsystem
# ---------------------------------------------------------------------------


def _collect_intel_pstate_info() -> Dict[str, Any]:
    """Detect Intel P-state driver status and HWP / EPP availability.

    Returns:
        A dictionary with keys ``intel_pstate_status``, ``hwp_enabled``,
        ``epp_available``, ``control_model``, and ``driver_limitations``.
    """
    info: Dict[str, Any] = {
        "intel_pstate_status": "",
        "hwp_enabled": False,
        "epp_available": False,
        "control_model": "unknown",
        "driver_limitations": [],
    }

    if not INTEL_PSTATE_PATH.exists():
        return info

    status = hardwareutils.read_text(INTEL_PSTATE_PATH / "status")
    info["intel_pstate_status"] = status

    hwp_dynamic = INTEL_PSTATE_PATH / "hwp_dynamic_boost"
    info["hwp_enabled"] = hwp_dynamic.exists() or status == "active"

    info["epp_available"] = EPP_PATH.exists()

    if info["hwp_enabled"]:
        info["control_model"] = "hwp_range"
        info["driver_limitations"].append(
            "HWP enabled: hardware autonomously selects P-state within "
            "configured range; OS hints are advisory, not deterministic"
        )
        if not info["epp_available"]:
            info["driver_limitations"].append(
                "EPP interface unavailable: energy_performance_preference "
                "cannot be tuned per-policy"
            )
    elif status == "passive":
        info["control_model"] = "pstate_range"
        info["driver_limitations"].append(
            "intel_pstate passive mode: behaves like acpi-cpufreq with "
            "range-based frequency targets"
        )
    elif status == "off":
        info["driver_limitations"].append(
            "intel_pstate driver disabled: fallback driver in use"
        )

    return info


def collect_cpufreq_info() -> Dict[str, Any]:
    """Collect CPU frequency-scaling metadata from the cpufreq sysfs interface.

    Extends the base cpufreq data with architecture-specific intel_pstate /
    HWP detection and driver limitation annotations.
    """
    info: Dict[str, Any] = {
        "driver": "",
        "governor_current": "",
        "governors_available": [],
        "min_khz": "",
        "max_khz": "",
        "base_frequency_khz": "",
        "available_frequencies_khz": [],
        "scaling_method": "unknown",
    }

    if not CPUFREQ_PATH.exists():
        pstate = _collect_intel_pstate_info()
        info.update(pstate)
        return info

    info["driver"] = hardwareutils.read_text(CPUFREQ_PATH / "scaling_driver")
    info["governor_current"] = hardwareutils.read_text(
        CPUFREQ_PATH / "scaling_governor"
    )

    available_governors = hardwareutils.read_text(
        CPUFREQ_PATH / "scaling_available_governors"
    )
    if available_governors:
        info["governors_available"] = available_governors.split()

    info["min_khz"] = hardwareutils.read_text(CPUFREQ_PATH / "cpuinfo_min_freq")
    info["max_khz"] = hardwareutils.read_text(CPUFREQ_PATH / "cpuinfo_max_freq")

    available_frequencies = hardwareutils.read_text(
        CPUFREQ_PATH / "scaling_available_frequencies"
    )
    if available_frequencies:
        info["available_frequencies_khz"] = available_frequencies.split()
        info["scaling_method"] = "discrete"
    else:
        base_freq = hardwareutils.read_text(CPUFREQ_PATH / "base_frequency")
        if base_freq:
            info["base_frequency_khz"] = base_freq
            info["scaling_method"] = "range"
        elif info["min_khz"] and info["max_khz"]:
            info["scaling_method"] = "range"

    # Architecture-specific intel_pstate / HWP detection.
    pstate = _collect_intel_pstate_info()
    info.update(pstate)

    # Override control_model when discrete frequencies are available.
    if info["scaling_method"] == "discrete":
        info["control_model"] = "discrete"

    return info


def collect_rapl_info() -> Dict[str, Any]:
    """Collect Intel RAPL power-capping domain availability and readability.

    Includes an active probe that attempts to read a numeric ``energy_uj``
    value from the first readable domain.
    """
    rapl: Dict[str, Any] = {
        "available": False,
        "domains": [],
        "readable": False,
    }

    if not POWERCAP_PATH.exists():
        return rapl

    domains: List[str] = []
    readable = False

    for domain_dir in sorted(POWERCAP_PATH.glob("intel-rapl*")):
        if not domain_dir.is_dir():
            continue

        friendly_name = hardwareutils.read_text(domain_dir / "name")
        label = f"{domain_dir.name} ({friendly_name})" if friendly_name else domain_dir.name
        domains.append(label)

        energy_path = domain_dir / "energy_uj"
        if energy_path.exists() and os.access(energy_path, os.R_OK):
            readable = True

    rapl["available"] = bool(domains)
    rapl["domains"] = domains
    rapl["readable"] = readable
    return rapl


def collect_cpu_info() -> Dict[str, Any]:
    """Collect CPU inventory together with cpufreq, RAPL, and taxonomy."""
    result = hardwareutils.run_cmd(["lscpu"])
    parsed = hardwareutils.parse_lscpu_output(result.stdout) if result.returncode == 0 else {}

    logical_cpus = hardwareutils.as_int(parsed.get("CPU(s)", "")) or (os.cpu_count() or 0)
    sockets = hardwareutils.as_int(parsed.get("Socket(s)", "")) or 0
    cores_per_socket = hardwareutils.as_int(parsed.get("Core(s) per socket", "")) or 0
    threads_per_core = hardwareutils.as_int(parsed.get("Thread(s) per core", "")) or 0
    physical_cores = cores_per_socket * sockets if cores_per_socket and sockets else 0

    cpufreq = collect_cpufreq_info()
    rapl = collect_rapl_info()

    # Build CPU taxonomy.
    cpu_present = logical_cpus > 0
    cpu_observable = cpu_present and rapl.get("readable", False)
    cpu_controllable = cpu_observable and CPUFREQ_GOVERNOR_PATH.exists() and os.access(
        CPUFREQ_GOVERNOR_PATH, os.W_OK
    )

    return {
        "vendor": parsed.get("Vendor ID", "unknown"),
        "model_name": parsed.get("Model name", "unknown"),
        "logical_cpus": logical_cpus,
        "physical_cores": physical_cores,
        "threads_per_core": threads_per_core,
        "sockets": sockets,
        "cpufreq": cpufreq,
        "rapl": rapl,
        "taxonomy": build_taxonomy(
            present=cpu_present,
            observable=cpu_observable,
            controllable=cpu_controllable,
            reason=None if cpu_controllable else (
                "cpufreq governor not writable" if not cpu_controllable and cpu_observable
                else "RAPL not readable" if not cpu_observable and cpu_present
                else "No CPUs detected" if not cpu_present else None
            ),
        ),
    }


# ---------------------------------------------------------------------------
# GPU subsystem
# ---------------------------------------------------------------------------


def _collect_nvidia_devices() -> List[GpuDevice]:
    """Query NVIDIA GPUs via ``nvidia-smi`` and return a device list."""
    if not which("nvidia-smi"):
        return []

    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,vbios_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = hardwareutils.run_cmd(query)
    if result.returncode != 0 or not result.stdout:
        return []

    devices: List[GpuDevice] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue

        index, name, driver, vbios, memory_total = parts
        clocks_out = hardwareutils.run_cmd(["nvidia-smi", "-i", index, "-q", "-d", "SUPPORTED_CLOCKS"])
        clocks = (
            hardwareutils.parse_supported_clocks(clocks_out.stdout)
            if clocks_out.returncode == 0
            else {"graphics": [], "memory": []}
        )

        devices.append(
            {
                "index": hardwareutils.as_int(index) if index.isdigit() else index,
                "name": name,
                "driver_version": driver,
                "vbios_version": vbios,
                "memory_total_mb": memory_total,
                "supported_graphics_clocks_mhz": [str(v) for v in clocks["graphics"]],
                "supported_memory_clocks_mhz": [str(v) for v in clocks["memory"]],
            }
        )

    return devices


def _collect_lspci_devices(nvidia_already_found: bool) -> VendorMap:
    """Scan PCI bus via ``lspci`` and return display adapters grouped by vendor."""
    fallback: VendorMap = {"nvidia": [], "intel": []}

    if not which("lspci"):
        return fallback

    result = hardwareutils.run_cmd(["lspci"])
    if result.returncode != 0 or not result.stdout:
        return fallback

    for line in result.stdout.splitlines():
        lower = line.lower()
        if not any(cls in lower for cls in LSPCI_DISPLAY_CLASSES):
            continue

        entry: GpuDevice = {
            "bus_id": line.split(" ", 1)[0],
            "name": line.split(":", 2)[-1].strip() if ":" in line else line.strip(),
            "source": "lspci",
        }

        if "nvidia" in lower and not nvidia_already_found:
            fallback["nvidia"].append(entry)
        elif "intel" in lower:
            fallback["intel"].append(entry)

    return fallback


#: Schema-required defaults for GPU device records.
_GPU_DEVICE_DEFAULTS: GpuDevice = {
    "index": None,
    "name": "",
    "driver_version": "",
    "vbios_version": "",
    "memory_total_mb": "",
    "supported_graphics_clocks_mhz": [],
    "supported_memory_clocks_mhz": [],
}


def _normalize_device(device: GpuDevice, vendor: str) -> GpuDevice:
    """Ensure *device* contains every field required by schema.json."""
    return {**_GPU_DEVICE_DEFAULTS, **device, "vendor": vendor}


def collect_gpu_info() -> Dict[str, Any]:
    """Collect GPU inventory grouped by vendor, with taxonomy."""
    nvidia_devices = _collect_nvidia_devices()
    lspci_buckets = _collect_lspci_devices(nvidia_already_found=bool(nvidia_devices))

    by_vendor: VendorMap = {
        "nvidia": nvidia_devices + lspci_buckets["nvidia"],
        "intel": lspci_buckets["intel"],
    }

    devices: List[GpuDevice] = [
        _normalize_device(device, vendor)
        for vendor, vendor_devices in by_vendor.items()
        for device in vendor_devices
    ]

    gpu_present = len(devices) > 0
    has_nvidia_smi = bool(which("nvidia-smi"))
    gpu_observable = gpu_present and has_nvidia_smi
    gpu_controllable = False
    gpu_control_reason = None

    if gpu_observable:
        # Check if clock control is available.
        nv_devs = by_vendor.get("nvidia", [])
        if nv_devs:
            check = hardwareutils.run_cmd(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"])
            if check.returncode == 0 and check.stdout and "N/A" not in check.stdout:
                gpu_controllable = True
            else:
                gpu_control_reason = "nvidia-smi supported clocks not available or N/A"
        else:
            gpu_control_reason = "No NVIDIA devices found; only NVIDIA clock control is supported"
    elif gpu_present:
        gpu_control_reason = "nvidia-smi not available; GPU metrics and control unavailable"
    else:
        gpu_control_reason = "No GPU detected"

    return {
        "nvidia_present": bool(by_vendor["nvidia"]),
        "device_count": len(devices),
        "devices": devices,
        "by_vendor": by_vendor,
        "taxonomy": build_taxonomy(
            present=gpu_present,
            observable=gpu_observable,
            controllable=gpu_controllable,
            reason=gpu_control_reason,
        ),
    }


# ---------------------------------------------------------------------------
# NUMA topology
# ---------------------------------------------------------------------------


def collect_numa_info() -> Dict[str, Any]:
    """Collect NUMA availability and per-node CPU mapping."""
    nodes: Dict[str, List[int]] = {}

    if which("numactl"):
        result = hardwareutils.run_cmd(["numactl", "--hardware"])
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = NUMACTL_NODE_RE.match(line.strip())
                if not match:
                    continue
                node_id, cpu_list_str = match.groups()
                nodes[node_id] = hardwareutils.parse_cpu_list(cpu_list_str)

    # Fallback when numactl is missing or fails.
    if not nodes:
        for node_dir in sorted(SYSFS_NODE_PATH.glob("node*")):
            node_id = node_dir.name.replace("node", "")
            cpu_list = hardwareutils.read_text(node_dir / "cpulist")
            if cpu_list:
                nodes[node_id] = hardwareutils.parse_cpu_list(cpu_list)

    return {
        "available": bool(nodes),
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------
# Tooling presence
# ---------------------------------------------------------------------------


def collect_tools_info() -> Dict[str, bool]:
    """Report which system tools required by the experiment stack are present."""
    return {
        "perf": bool(which("perf")),
        "cpupower": bool(which("cpupower")),
        "numactl": bool(which("numactl")),
        "nvidia_smi": bool(which("nvidia-smi")),
    }


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def collect_permissions_info(
    cpu: Dict[str, Any],
    tools: Dict[str, bool],
    gpu: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine what the current process is permitted to read or write."""
    cpufreq_writable = CPUFREQ_GOVERNOR_PATH.exists() and os.access(
        CPUFREQ_GOVERNOR_PATH, os.W_OK
    )

    gpu_clock_control_available = gpu.get("taxonomy", {}).get("controllable", False)

    return {
        "is_root": os.geteuid() == 0,
        "rapl_readable": bool(cpu.get("rapl", {}).get("readable", False)),
        "cpufreq_writable": cpufreq_writable,
        "gpu_clock_control_available": gpu_clock_control_available,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> Dict[str, Any]:
    """Assemble the full node readiness report conforming to schema.json.

    Calls every collector, then runs active probes and the granular
    capability assessment.
    """
    node = collect_node_info()
    cpu = collect_cpu_info()
    gpu = collect_gpu_info()
    numa = collect_numa_info()
    tools = collect_tools_info()
    permissions = collect_permissions_info(cpu, tools, gpu)

    # Active probes (effective validation).
    probes = collect_active_probes(tools, gpu, cpu)

    # Granular capability assessment.
    capabilities = assess_capabilities(cpu, gpu, numa, tools, permissions, probes)
    backends = build_backends(capabilities, cpu, gpu)
    blockers = build_experiment_blockers(capabilities)
    readiness = evaluate_readiness(capabilities)

    return {
        "schema_version": __schema_version__,
        "tool_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "node": node,
        "cpu": {
            "vendor": cpu["vendor"],
            "model_name": cpu["model_name"],
            "logical_cpus": cpu["logical_cpus"],
            "physical_cores": cpu["physical_cores"],
            "threads_per_core": cpu["threads_per_core"],
            "sockets": cpu["sockets"],
            "cpufreq": cpu["cpufreq"],
            "rapl": cpu["rapl"],
            "taxonomy": cpu["taxonomy"],
        },
        "gpu": {
            "nvidia_present": gpu["nvidia_present"],
            "device_count": gpu["device_count"],
            "devices": gpu["devices"],
            "by_vendor": gpu["by_vendor"],
            "taxonomy": gpu["taxonomy"],
        },
        "numa": numa,
        "tools": tools,
        "permissions": permissions,
        "active_probes": probes,
        "capabilities": capabilities,
        "backends": backends,
        "experiment_blockers": blockers,
        "readiness": readiness,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_summary(report: Dict[str, Any]) -> None:
    """Print a compact human-readable readiness summary to stdout."""
    print("=== Node Readiness Summary ===")
    print(f"Host: {report['node']['hostname']}")
    print(f"CPU:  {report['cpu']['vendor']} | {report['cpu']['model_name']}")
    print(
        f"GPU devices: {report['gpu']['device_count']} "
        f"(NVIDIA: {len(report['gpu']['by_vendor']['nvidia'])}, "
        f"Intel: {len(report['gpu']['by_vendor']['intel'])})"
    )
    print(f"RAPL readable:     {report['permissions']['rapl_readable']}")
    print()

    # Per-capability status.
    print("--- Capabilities ---")
    capabilities = report.get("capabilities", {})
    for cap_name, cap_data in capabilities.items():
        status = cap_data.get("status", "unknown")
        severity = cap_data.get("severity", "info")
        symbol = _SEVERITY_SYMBOL.get(severity, "[?]")
        print(f"  {symbol} {cap_name}: {status}")
        if cap_data.get("reason"):
            print(f"      Reason: {cap_data['reason']}")
        if cap_data.get("recommendation"):
            print(f"      Fix: {cap_data['recommendation']}")

    # Experiment blockers.
    blockers = report.get("experiment_blockers", [])
    if blockers:
        print()
        print("--- Experiment Blockers ---")
        for b in blockers:
            print(f"  [X] {b['capability']}: {b['message']}")
            if b.get("recommendation"):
                print(f"      Fix: {b['recommendation']}")

    # Readiness flags.
    readiness = report.get("readiness", {})
    print()
    print("--- Readiness Flags ---")
    for key, value in readiness.items():
        if key == "issues":
            continue
        print(f"  {key}: {value}")

    issues = readiness.get("issues", [])
    if issues:
        print()
        print(f"--- Issues ({len(issues)}) ---")
        for issue in issues:
            symbol = _SEVERITY_SYMBOL.get(issue.get("severity", "info"), "[?]")
            print(f"  {symbol} {issue['capability']}: {issue['message']}")


def _write_report(report: Dict[str, Any], output_path: Path) -> None:
    """Serialise *report* to *output_path* as indented JSON (UTF-8)."""
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the node readiness CLI."""
    parser = argparse.ArgumentParser(
        description="Check node readiness for telemetry/DVFS workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to save the JSON report. (default: node_readiness_report_<timestamp>.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary; write only the JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    report = build_report()

    if args.output_json is None:
        timestamp = report["generated_at"].replace(":", "-").replace(".", "-")
        args.output_json = f"{_DEFAULT_OUTPUT_NAME}_{timestamp}.json"

    output_path = Path(args.output_json)
    _write_report(report, output_path)

    if not args.quiet:
        print_summary(report)
        print(f"\nJSON report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
