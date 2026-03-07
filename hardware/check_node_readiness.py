#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Node readiness checker for telemetry and DVFS experiments.

This script is intentionally read-only: it detects node capabilities,
permissions, and tooling without changing system settings.

Output is emitted as JSON following ``hardware/schema.json`` as the base
contract, with one additive field under ``gpu`` (``by_vendor``) to preserve
vendor-grouped discovery details for NVIDIA and Intel GPUs.

Typical usage::

    python3 hardware/check_node_readiness.py \\
        --output-json hardware/node_readiness_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Any, Dict, List

from utils import (
    as_int,
    hardwareutils,
    parse_cpu_list,
    parse_supported_clocks,
    read_text,
    run_cmd,
)

__version__ = "1.0.0"
__schema_version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default path written by main() when --output-json is not supplied.
_DEFAULT_OUTPUT_JSON = "node_readiness_report.json"

# sysfs paths referenced by multiple functions.
_CPUFREQ_PATH: Path = Path("/sys/devices/system/cpu/cpu0/cpufreq")
_CPUFREQ_GOVERNOR_PATH: Path = _CPUFREQ_PATH / "scaling_governor"
_POWERCAP_PATH: Path = Path("/sys/class/powercap")
_SYSFS_NODE_PATH: Path = Path("/sys/devices/system/node")

# Compiled regex for ``numactl --hardware`` node-CPU lines.
_NUMACTL_NODE_RE: re.Pattern[str] = re.compile(r"^node\s+(\d+)\s+cpus:\s+(.+)$")

# GPU device classes reported by lspci that indicate a display adapter.
_LSPCI_DISPLAY_CLASSES: tuple[str, ...] = ("vga", "3d", "display")

# ---------------------------------------------------------------------------
# Internal type aliases
# ---------------------------------------------------------------------------

#: A single GPU device record as stored in ``by_vendor`` buckets.
_GpuDevice = Dict[str, Any]

#: Vendor-keyed map of GPU device lists — ``{"nvidia": [...], "intel": [...]}``.
_VendorMap = Dict[str, List[_GpuDevice]]


# ---------------------------------------------------------------------------
# Node identification
# ---------------------------------------------------------------------------


def collect_node_info() -> Dict[str, str]:
    """Collect basic node identification fields.

    Reads the hostname, OS name, kernel release and machine architecture
    using stdlib facilities — no sub-processes are spawned.

    Returns:
        A dictionary with the following string keys:

        - ``hostname`` – network name of the machine.
        - ``os`` – operating system name (e.g. ``"Linux"``).
        - ``kernel`` – kernel release string (e.g. ``"5.15.0-91-generic"``).
        - ``architecture`` – CPU instruction-set architecture (e.g. ``"x86_64"``).
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


def collect_cpufreq_info() -> Dict[str, Any]:
    """Collect CPU frequency-scaling metadata from the cpufreq sysfs interface.

    All values are read from ``/sys/devices/system/cpu/cpu0/cpufreq``.  If
    that directory does not exist (e.g. on systems where the cpufreq driver
    is not loaded), all fields are returned with empty/default values so the
    caller can unconditionally unpack the result.

    Returns:
        A dictionary with the following keys:

        - ``driver`` (*str*) – active scaling driver (e.g. ``"intel_pstate"``).
        - ``governor_current`` (*str*) – active governor (e.g. ``"powersave"``).
        - ``governors_available`` (*List[str]*) – governors supported by the driver.
        - ``min_khz`` (*str*) – hardware minimum frequency in kHz.
        - ``max_khz`` (*str*) – hardware maximum frequency in kHz.
        - ``available_frequencies_khz`` (*List[str]*) – discrete frequency steps
          available for the ``userspace`` governor, or an empty list if the
          driver does not expose a fixed table.
    """
    info: Dict[str, Any] = {
        "driver": "",
        "governor_current": "",
        "governors_available": [],
        "min_khz": "",
        "max_khz": "",
        "available_frequencies_khz": [],
    }

    if not _CPUFREQ_PATH.exists():
        return info

    info["driver"] = read_text(_CPUFREQ_PATH / "scaling_driver")
    info["governor_current"] = read_text(_CPUFREQ_PATH / "scaling_governor")

    available_governors = read_text(_CPUFREQ_PATH / "scaling_available_governors")
    if available_governors:
        info["governors_available"] = available_governors.split()

    info["min_khz"] = read_text(_CPUFREQ_PATH / "cpuinfo_min_freq")
    info["max_khz"] = read_text(_CPUFREQ_PATH / "cpuinfo_max_freq")

    available_frequencies = read_text(_CPUFREQ_PATH / "scaling_available_frequencies")
    if available_frequencies:
        info["available_frequencies_khz"] = available_frequencies.split()

    return info


def collect_rapl_info() -> Dict[str, Any]:
    """Collect Intel RAPL power-capping domain availability and readability.

    Scans ``/sys/class/powercap/intel-rapl*`` directories.  For each
    domain, the human-friendly name is appended in parentheses when
    available (e.g. ``"intel-rapl:0 (package-0)"``).  Readability is
    confirmed by checking read permissions on ``energy_uj``.

    Returns:
        A dictionary with the following keys:

        - ``available`` (*bool*) – ``True`` when at least one RAPL domain
          directory was found under ``/sys/class/powercap``.
        - ``domains`` (*List[str]*) – list of discovered domain labels.
        - ``readable`` (*bool*) – ``True`` when at least one ``energy_uj``
          counter is readable by the current process.
    """
    rapl: Dict[str, Any] = {
        "available": False,
        "domains": [],
        "readable": False,
    }

    if not _POWERCAP_PATH.exists():
        return rapl

    domains: List[str] = []
    readable = False

    for domain_dir in sorted(_POWERCAP_PATH.glob("intel-rapl*")):
        if not domain_dir.is_dir():
            continue

        friendly_name = read_text(domain_dir / "name")
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
    """Collect CPU inventory together with cpufreq and RAPL sub-sections.

    Invokes ``lscpu`` to obtain vendor, model and topology fields.  Falls
    back to :func:`os.cpu_count` for the logical CPU count when ``lscpu``
    is unavailable or returns a non-zero exit code.

    Returns:
        A dictionary with the following keys:

        - ``vendor`` (*str*) – CPU vendor string from ``lscpu`` (``"unknown"``
          when not available).
        - ``model_name`` (*str*) – CPU model name (``"unknown"`` when not
          available).
        - ``logical_cpus`` (*int*) – total logical CPU count (hardware threads).
        - ``physical_cores`` (*int*) – total physical core count across all sockets;
          ``0`` when the topology cannot be determined.
        - ``threads_per_core`` (*int*) – simultaneous multi-threading (SMT) factor;
          ``0`` when unknown.
        - ``sockets`` (*int*) – number of physical CPU packages; ``0`` when unknown.
        - ``cpufreq`` (*dict*) – result of :func:`collect_cpufreq_info`.
        - ``rapl`` (*dict*) – result of :func:`collect_rapl_info`.
    """
    result = run_cmd(["lscpu"])
    parsed = hardwareutils.parse_lscpu_output(result.stdout) if result.returncode == 0 else {}

    logical_cpus = as_int(parsed.get("CPU(s)", "")) or (os.cpu_count() or 0)
    sockets = as_int(parsed.get("Socket(s)", "")) or 0
    cores_per_socket = as_int(parsed.get("Core(s) per socket", "")) or 0
    threads_per_core = as_int(parsed.get("Thread(s) per core", "")) or 0
    physical_cores = cores_per_socket * sockets if cores_per_socket and sockets else 0

    return {
        "vendor": parsed.get("Vendor ID", "unknown"),
        "model_name": parsed.get("Model name", "unknown"),
        "logical_cpus": logical_cpus,
        "physical_cores": physical_cores,
        "threads_per_core": threads_per_core,
        "sockets": sockets,
        "cpufreq": collect_cpufreq_info(),
        "rapl": collect_rapl_info(),
    }


# ---------------------------------------------------------------------------
# GPU subsystem — private helpers
# ---------------------------------------------------------------------------


def _collect_nvidia_devices() -> List[_GpuDevice]:
    """Query NVIDIA GPUs via ``nvidia-smi`` and return a device list.

    For each GPU the supported clock frequencies are fetched with a
    secondary ``nvidia-smi -i <index> -q -d SUPPORTED_CLOCKS`` call.

    Returns:
        A list of device dictionaries.  Each entry contains:

        - ``index`` (*int | str*) – zero-based GPU index reported by nvidia-smi.
        - ``name`` (*str*) – GPU product name.
        - ``driver_version`` (*str*) – installed NVIDIA driver version.
        - ``vbios_version`` (*str*) – VBIOS firmware version.
        - ``memory_total_mb`` (*str*) – total frame-buffer size in MiB.
        - ``supported_graphics_clocks_mhz`` (*List[str]*) – available graphics
          clocks sorted in ascending order.
        - ``supported_memory_clocks_mhz`` (*List[str]*) – available memory
          clocks sorted in ascending order.

        Returns an empty list when ``nvidia-smi`` is not in ``PATH`` or
        produces no output.
    """
    if not which("nvidia-smi"):
        return []

    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,vbios_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = run_cmd(query)
    if result.returncode != 0 or not result.stdout:
        return []

    devices: List[_GpuDevice] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue

        index, name, driver, vbios, memory_total = parts
        clocks_out = run_cmd(["nvidia-smi", "-i", index, "-q", "-d", "SUPPORTED_CLOCKS"])
        clocks = (
            parse_supported_clocks(clocks_out.stdout)
            if clocks_out.returncode == 0
            else {"graphics": [], "memory": []}
        )

        devices.append(
            {
                "index": as_int(index) if index.isdigit() else index,
                "name": name,
                "driver_version": driver,
                "vbios_version": vbios,
                "memory_total_mb": memory_total,
                "supported_graphics_clocks_mhz": [str(v) for v in clocks["graphics"]],
                "supported_memory_clocks_mhz": [str(v) for v in clocks["memory"]],
            }
        )

    return devices


def _collect_lspci_devices(nvidia_already_found: bool) -> _VendorMap:
    """Scan PCI bus via ``lspci`` and return display adapters grouped by vendor.

    Only lines whose device class contains ``vga``, ``3d``, or ``display``
    (case-insensitive) are considered.  NVIDIA entries are skipped when
    ``nvidia_already_found`` is ``True`` to avoid duplicating devices
    already discovered through ``nvidia-smi``.

    Args:
        nvidia_already_found: When ``True``, NVIDIA devices from ``lspci``
            are ignored because they were already captured by
            :func:`_collect_nvidia_devices`.

    Returns:
        A :data:`_VendorMap` (``{"nvidia": [...], "intel": [...]}``) with
        entries discovered by ``lspci``.  Each entry contains:

        - ``bus_id`` (*str*) – PCI bus address (e.g. ``"01:00.0"``).
        - ``name`` (*str*) – device description from ``lspci``.
        - ``source`` (*str*) – always ``"lspci"``.

        Returns empty lists for both vendors when ``lspci`` is unavailable
        or fails.
    """
    fallback: _VendorMap = {"nvidia": [], "intel": []}

    if not which("lspci"):
        return fallback

    result = run_cmd(["lspci"])
    if result.returncode != 0 or not result.stdout:
        return fallback

    for line in result.stdout.splitlines():
        lower = line.lower()
        if not any(cls in lower for cls in _LSPCI_DISPLAY_CLASSES):
            continue

        entry: _GpuDevice = {
            "bus_id": line.split(" ", 1)[0],
            "name": line.split(":", 2)[-1].strip() if ":" in line else line.strip(),
            "source": "lspci",
        }

        if "nvidia" in lower and not nvidia_already_found:
            fallback["nvidia"].append(entry)
        elif "intel" in lower:
            fallback["intel"].append(entry)

    return fallback


# ---------------------------------------------------------------------------
# GPU subsystem — normalisation
# ---------------------------------------------------------------------------

#: Schema-required fields for every entry in ``gpu.devices``, with their
#: typed default values used when a device was discovered via ``lspci``
#: rather than ``nvidia-smi``.
_GPU_DEVICE_DEFAULTS: _GpuDevice = {
    "index": None,
    "name": "",
    "driver_version": "",
    "vbios_version": "",
    "memory_total_mb": "",
    "supported_graphics_clocks_mhz": [],
    "supported_memory_clocks_mhz": [],
}


def _normalize_device(device: _GpuDevice, vendor: str) -> _GpuDevice:
    """Ensure *device* contains every field required by ``hardware/schema.json``.

    Devices discovered via ``lspci`` carry only ``bus_id``, ``name``, and
    ``source``.  This function merges them with :data:`_GPU_DEVICE_DEFAULTS`
    so that the flattened ``gpu.devices`` list has a uniform shape regardless
    of the detection source.  Fields already present in *device* are never
    overwritten.

    Args:
        device: Raw device record from :func:`_collect_nvidia_devices` or
            :func:`_collect_lspci_devices`.
        vendor: Vendor key (e.g. ``"nvidia"`` or ``"intel"``) to attach as
            the ``vendor`` field.

    Returns:
        A new dictionary that contains at minimum all keys from
        :data:`_GPU_DEVICE_DEFAULTS` plus ``vendor``, ``bus_id`` (when
        present), and ``source`` (when present).
    """
    normalized: _GpuDevice = {**_GPU_DEVICE_DEFAULTS, **device, "vendor": vendor}
    return normalized


# ---------------------------------------------------------------------------
# GPU subsystem — public collector
# ---------------------------------------------------------------------------


def collect_gpu_info() -> Dict[str, Any]:
    """Collect GPU inventory grouped by vendor, with a flattened device list.

    NVIDIA GPUs are discovered first via ``nvidia-smi`` (rich metadata
    including clock tables).  ``lspci`` is used as a supplementary source
    to catch NVIDIA GPUs when ``nvidia-smi`` is absent and to discover
    Intel integrated graphics.

    Returns:
        A dictionary with the following keys:

        - ``nvidia_present`` (*bool*) – ``True`` when at least one NVIDIA
          device was found through any source.
        - ``device_count`` (*int*) – total number of unique GPU devices across
          all vendors.
        - ``devices`` (*List[dict]*) – flat list of all device records, each
          enriched with a ``vendor`` field.
        - ``by_vendor`` (*dict*) – vendor-keyed map:
          ``{"nvidia": [...], "intel": [...]}``.
    """
    nvidia_devices = _collect_nvidia_devices()
    lspci_buckets = _collect_lspci_devices(nvidia_already_found=bool(nvidia_devices))

    by_vendor: _VendorMap = {
        "nvidia": nvidia_devices + lspci_buckets["nvidia"],
        "intel": lspci_buckets["intel"],
    }

    devices: List[_GpuDevice] = [
        _normalize_device(device, vendor)
        for vendor, vendor_devices in by_vendor.items()
        for device in vendor_devices
    ]

    return {
        "nvidia_present": bool(by_vendor["nvidia"]),
        "device_count": len(devices),
        "devices": devices,
        "by_vendor": by_vendor,
    }


# ---------------------------------------------------------------------------
# NUMA topology
# ---------------------------------------------------------------------------


def collect_numa_info() -> Dict[str, Any]:
    """Collect NUMA availability and per-node CPU mapping.

    Attempts to parse ``numactl --hardware`` output first.  Falls back to
    reading ``/sys/devices/system/node/node*/cpulist`` when ``numactl`` is
    absent or returns a non-zero exit code.

    Returns:
        A dictionary with the following keys:

        - ``available`` (*bool*) – ``True`` when at least one NUMA node was
          discovered through either method.
        - ``nodes`` (*Dict[str, List[int]]*) – mapping of string node IDs
          (e.g. ``"0"``, ``"1"``) to sorted integer lists of CPU indices
          belonging to that node.
    """
    nodes: Dict[str, List[int]] = {}

    if which("numactl"):
        result = run_cmd(["numactl", "--hardware"])
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = _NUMACTL_NODE_RE.match(line.strip())
                if not match:
                    continue
                node_id, cpu_list_str = match.groups()
                nodes[node_id] = parse_cpu_list(cpu_list_str)

    # Fallback when numactl is missing or fails.
    if not nodes:
        for node_dir in sorted(_SYSFS_NODE_PATH.glob("node*")):
            node_id = node_dir.name.replace("node", "")
            cpu_list = read_text(node_dir / "cpulist")
            if cpu_list:
                nodes[node_id] = parse_cpu_list(cpu_list)

    return {
        "available": bool(nodes),
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------
# Tooling presence
# ---------------------------------------------------------------------------


def collect_tools_info() -> Dict[str, bool]:
    """Report which system tools required by the experiment stack are present.

    Each value is ``True`` when the corresponding executable is found in
    ``PATH`` via :func:`shutil.which`.

    Returns:
        A dictionary with boolean values for the following keys:

        - ``perf`` – Linux ``perf`` profiler (required for telemetry).
        - ``cpupower`` – ``cpupower`` utility (needed to change CPU governors).
        - ``numactl`` – ``numactl`` (needed for NUMA-aware process pinning).
        - ``nvidia_smi`` – ``nvidia-smi`` (needed for NVIDIA GPU discovery and
          clock control).
    """
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
    """Determine what the current process is permitted to read or write.

    Checks sysfs write access for CPU frequency control and RAPL read access
    for energy counters.  GPU clock control availability is probed by running
    ``nvidia-smi -q -d SUPPORTED_CLOCKS`` and checking for a valid response.

    Args:
        cpu: CPU section produced by :func:`collect_cpu_info`.  Used to
            derive ``rapl_readable`` from the nested ``rapl.readable`` flag.
        tools: Tools section produced by :func:`collect_tools_info`.  Used
            to gate the ``nvidia-smi`` probe.
        gpu: GPU section produced by :func:`collect_gpu_info`.  Used to
            confirm at least one NVIDIA device is present before probing.

    Returns:
        A dictionary with the following keys:

        - ``is_root`` (*bool*) – ``True`` when the process runs as UID 0.
        - ``rapl_readable`` (*bool*) – ``True`` when at least one RAPL
          ``energy_uj`` file is readable.
        - ``cpufreq_writable`` (*bool*) – ``True`` when the cpufreq governor
          sysfs file is writable by the current process.
        - ``gpu_clock_control_available`` (*bool*) – ``True`` when
          ``nvidia-smi`` confirms supported clocks are queryable (a proxy
          for ``nvidia-smi -ac`` controllability).
    """
    cpufreq_writable = _CPUFREQ_GOVERNOR_PATH.exists() and os.access(
        _CPUFREQ_GOVERNOR_PATH, os.W_OK
    )

    gpu_clock_control_available = False
    if tools.get("nvidia_smi") and gpu.get("by_vendor", {}).get("nvidia"):
        result = run_cmd(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"])
        if result.returncode == 0 and result.stdout and "N/A" not in result.stdout:
            gpu_clock_control_available = True

    return {
        "is_root": os.geteuid() == 0,
        "rapl_readable": bool(cpu.get("rapl", {}).get("readable", False)),
        "cpufreq_writable": cpufreq_writable,
        "gpu_clock_control_available": gpu_clock_control_available,
    }


# ---------------------------------------------------------------------------
# Readiness evaluation
# ---------------------------------------------------------------------------


def evaluate_readiness(
    cpu: Dict[str, Any],
    gpu: Dict[str, Any],
    tools: Dict[str, bool],
    permissions: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate whether the node meets the requirements for each workflow tier.

    Two tiers are assessed:

    **Telemetry** (hard requirements — all must pass):

    - ``perf`` is present in ``PATH``.
    - Intel RAPL energy counters are readable.
    - At least one GPU was detected.

    **Control** (hard requirements — requires telemetry + all must pass):

    - CPU frequency governor sysfs file is writable.
    - NVIDIA GPU clock control is available.

    Soft requirements generate entries in ``warnings`` but do not block
    either tier.

    Args:
        cpu: CPU section produced by :func:`collect_cpu_info`.
        gpu: GPU section produced by :func:`collect_gpu_info`.
        tools: Tools section produced by :func:`collect_tools_info`.
        permissions: Permissions section produced by
            :func:`collect_permissions_info`.

    Returns:
        A dictionary with the following keys:

        - ``ready_for_telemetry`` (*bool*) – all telemetry requirements met.
        - ``ready_for_control`` (*bool*) – all control requirements met.
        - ``blocking_issues`` (*List[str]*) – human-readable descriptions of
          unmet hard requirements.
        - ``warnings`` (*List[str]*) – non-blocking limitations that may
          degrade experiment quality.
    """
    blocking_issues: List[str] = []
    warnings: List[str] = []

    # --- Hard requirements for telemetry ---
    if not tools.get("perf", False):
        blocking_issues.append("Missing required tool: perf")

    if not permissions.get("rapl_readable", False):
        blocking_issues.append("Intel RAPL counters are not readable")

    if gpu.get("device_count", 0) == 0:
        blocking_issues.append("No GPU detected (NVIDIA/Intel)")

    # --- Soft requirements (warnings) ---
    if not tools.get("numactl", False):
        warnings.append("numactl not found: NUMA-aware pinning will be limited")

    if not tools.get("cpupower", False):
        warnings.append("cpupower not found: CPU governor tooling is limited")

    if gpu.get("by_vendor", {}).get("nvidia") and not tools.get("nvidia_smi", False):
        warnings.append("NVIDIA GPU detected without nvidia-smi in PATH")

    ready_for_telemetry = len(blocking_issues) == 0
    ready_for_control = (
        ready_for_telemetry
        and permissions.get("cpufreq_writable", False)
        and permissions.get("gpu_clock_control_available", False)
    )

    if ready_for_telemetry and not permissions.get("cpufreq_writable", False):
        warnings.append("Telemetry ready, but CPU frequency control is not writable")

    if ready_for_telemetry and not permissions.get("gpu_clock_control_available", False):
        warnings.append("Telemetry ready, but GPU clock control capability is unavailable")

    return {
        "ready_for_telemetry": ready_for_telemetry,
        "ready_for_control": ready_for_control,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> Dict[str, Any]:
    """Assemble the full node readiness report conforming to ``hardware/schema.json``.

    Calls every ``collect_*`` function in dependency order and combines the
    results into a single serialisable dictionary.  The ``gpu.by_vendor``
    field is an additive extension beyond the base schema contract.

    Returns:
        A dictionary ready for JSON serialisation.  Top-level keys are:

        ``schema_version``, ``tool_version``, ``generated_at``,
        ``node``, ``cpu``, ``gpu``, ``numa``, ``tools``,
        ``permissions``, ``readiness``.
    """
    node = collect_node_info()
    cpu = collect_cpu_info()
    gpu = collect_gpu_info()
    numa = collect_numa_info()
    tools = collect_tools_info()
    permissions = collect_permissions_info(cpu, tools, gpu)
    readiness = evaluate_readiness(cpu, gpu, tools, permissions)

    return {
        "schema_version": __schema_version__,
        "tool_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "node": node,
        "cpu": cpu,
        "gpu": {
            "nvidia_present": gpu["nvidia_present"],
            "device_count": gpu["device_count"],
            "devices": gpu["devices"],
            "by_vendor": gpu["by_vendor"],
        },
        "numa": numa,
        "tools": tools,
        "permissions": permissions,
        "readiness": readiness,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_summary(report: Dict[str, Any]) -> None:
    """Print a compact human-readable readiness summary to stdout.

    Outputs the hostname, CPU model, GPU count, key permission flags
    and the final telemetry/control readiness verdicts.  Blocking issues
    and warnings are printed as bullet lists when present.

    Args:
        report: The full report dictionary produced by :func:`build_report`.
    """
    print("=== Node Readiness Summary ===")
    print(f"Host: {report['node']['hostname']}")
    print(f"CPU:  {report['cpu']['vendor']} | {report['cpu']['model_name']}")
    print(
        f"GPU devices: {report['gpu']['device_count']} "
        f"(NVIDIA: {len(report['gpu']['by_vendor']['nvidia'])}, "
        f"Intel: {len(report['gpu']['by_vendor']['intel'])})"
    )
    print(f"RAPL readable:     {report['permissions']['rapl_readable']}")
    print(f"Telemetry ready:   {report['readiness']['ready_for_telemetry']}")
    print(f"Control ready:     {report['readiness']['ready_for_control']}")

    if report["readiness"]["blocking_issues"]:
        print("Blocking issues:")
        for issue in report["readiness"]["blocking_issues"]:
            print(f"  - {issue}")

    if report["readiness"]["warnings"]:
        print("Warnings:")
        for warning in report["readiness"]["warnings"]:
            print(f"  - {warning}")


def _write_report(report: Dict[str, Any], output_path: Path) -> None:
    """Serialise *report* to *output_path* as indented JSON (UTF-8).

    Extracted from :func:`main` to allow tests to call :func:`build_report`
    and verify the serialisation path independently.

    Args:
        report: The full report dictionary produced by :func:`build_report`.
        output_path: Destination file path.  Parent directories must already
            exist; :func:`Path.write_text` is used directly.
    """
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the node readiness CLI.

    Returns:
        An :class:`argparse.Namespace` with the following attributes:

        - ``output_json`` (*str*) – destination path for the JSON report.
        - ``quiet`` (*bool*) – when ``True``, the human-readable summary is
          suppressed and only the JSON file is written.
    """
    parser = argparse.ArgumentParser(
        description="Check node readiness for telemetry/DVFS workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-json",
        default=_DEFAULT_OUTPUT_JSON,
        help="Path to save the JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary; write only the JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point.

    Builds the full node readiness report, writes it to disk, and
    optionally prints a human-readable summary.

    Returns:
        ``0`` on success.
    """
    args = parse_args()
    report = build_report()

    output_path = Path(args.output_json)
    _write_report(report, output_path)

    if not args.quiet:
        print_summary(report)
        print(f"\nJSON report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
