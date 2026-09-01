"""Capability assessment, backend derivation, and readiness evaluation.

- Hyperion

This module evaluates each experimental capability independently, derives
available telemetry/control backends, extracts experiment blockers, and
produces the granular per-capability readiness verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils import (
    AVAILABLE,
    CPUFREQ_GOVERNOR_PATH,
    CRITICAL,
    DEGRADED,
    INFO,
    UNAVAILABLE,
    WARNING,
    CapabilityResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap(
    status: str,
    severity: str,
    evidence: str,
    reason: Optional[str] = None,
    recommendation: Optional[str] = None,
) -> CapabilityResult:
    """Build a single capability check result."""
    return {
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "reason": reason,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Capability assessment
# ---------------------------------------------------------------------------


def assess_capabilities(
    cpu: Dict[str, Any],
    gpu: Dict[str, Any],
    numa: Dict[str, Any],
    tools: Dict[str, bool],
    permissions: Dict[str, Any],
    probes: Dict[str, Any],
) -> Dict[str, CapabilityResult]:
    """Evaluate each experimental capability independently.

    Returns a dict keyed by capability name with structured results
    including status, severity, evidence, reason, and recommendation.
    """
    caps: Dict[str, CapabilityResult] = {}

    # --- cpu_discovery ---
    if cpu.get("logical_cpus", 0) > 0:
        caps["cpu_discovery"] = _cap(
            AVAILABLE,
            INFO,
            f"lscpu detected {cpu['logical_cpus']} logical CPUs, "
            f"{cpu['physical_cores']} physical cores, "
            f"{cpu['sockets']} socket(s)",
        )
    else:
        caps["cpu_discovery"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            "lscpu failed or returned 0 CPUs",
            reason="Cannot determine CPU topology",
            recommendation="Verify lscpu is installed and /proc/cpuinfo is readable",
        )

    # --- cpu_telemetry ---
    perf_probe = probes.get("perf", {})
    freq_probe = probes.get("cpu_frequency", {})
    if perf_probe.get("accessible") and freq_probe.get("success"):
        caps["cpu_telemetry"] = _cap(
            AVAILABLE,
            INFO,
            f"perf accessible (paranoid={perf_probe.get('event_paranoid_level')}); "
            f"current freq={freq_probe.get('current_khz', '?')} kHz",
        )
    elif perf_probe.get("accessible"):
        caps["cpu_telemetry"] = _cap(
            DEGRADED,
            WARNING,
            "perf accessible but CPU frequency read failed",
            reason="scaling_cur_freq unreadable; frequency telemetry unavailable",
            recommendation=(
                "Check cpufreq driver is loaded and "
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq is readable"
            ),
        )
    elif tools.get("perf"):
        paranoid = perf_probe.get("event_paranoid_level")
        caps["cpu_telemetry"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            f"perf binary found but stat probe failed "
            f"(paranoid={paranoid})",
            reason=perf_probe.get("note", "perf stat returned non-zero"),
            recommendation=(
                "Set /proc/sys/kernel/perf_event_paranoid to 1 or lower, "
                "or run as root: sudo sysctl kernel.perf_event_paranoid=1"
            ),
        )
    else:
        caps["cpu_telemetry"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            "perf binary not found in PATH",
            reason="Hardware performance counter telemetry unavailable",
            recommendation="Install linux-tools-$(uname -r) or perf package",
        )

    # --- energy_cpu ---
    rapl = cpu.get("rapl", {})
    rapl_probe = probes.get("rapl_read", {})
    if rapl.get("readable") and rapl_probe.get("success"):
        caps["energy_cpu"] = _cap(
            AVAILABLE,
            INFO,
            f"RAPL energy_uj readable; sample read from "
            f"{rapl_probe.get('domain_tested', '?')} returned "
            f"{rapl_probe.get('sample_value_uj', '?')} uJ; "
            f"{len(rapl.get('domains', []))} domain(s) found",
        )
    elif rapl.get("readable"):
        caps["energy_cpu"] = _cap(
            DEGRADED,
            WARNING,
            f"RAPL domains found ({len(rapl.get('domains', []))}) and "
            f"permissions OK but active read probe failed",
            reason="energy_uj file readable but did not return a numeric value",
            recommendation=(
                "Verify /sys/class/powercap/intel-rapl:0/energy_uj "
                "returns a valid integer; the counter may have overflowed"
            ),
        )
    elif rapl.get("available"):
        caps["energy_cpu"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            f"RAPL domains detected ({len(rapl.get('domains', []))}) "
            f"but energy_uj not readable",
            reason="Insufficient permissions to read RAPL energy counters",
            recommendation=(
                "Grant read access: sudo chmod a+r "
                "/sys/class/powercap/intel-rapl:*/energy_uj, or add user "
                "to the appropriate group, or configure a udev rule"
            ),
        )
    else:
        caps["energy_cpu"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            "No RAPL domains found under /sys/class/powercap/",
            reason="Intel RAPL interface absent (non-Intel CPU or module not loaded)",
            recommendation=(
                "Load the intel_rapl_common module: sudo modprobe intel_rapl_common; "
                "on non-Intel platforms RAPL is not available"
            ),
        )

    # --- cpu_control ---
    cpufreq = cpu.get("cpufreq", {})
    if permissions.get("cpufreq_writable"):
        limitations = cpufreq.get("driver_limitations", [])
        if limitations:
            caps["cpu_control"] = _cap(
                DEGRADED,
                WARNING,
                f"cpufreq governor writable (driver={cpufreq.get('driver', '?')}, "
                f"control_model={cpufreq.get('control_model', '?')})",
                reason="; ".join(limitations),
                recommendation=(
                    "For deterministic DVFS, consider disabling HWP if possible "
                    "(intel_pstate=passive kernel parameter) or account for "
                    "hardware-autonomous frequency selection in measurements"
                ),
            )
        else:
            caps["cpu_control"] = _cap(
                AVAILABLE,
                INFO,
                f"cpufreq governor writable; driver={cpufreq.get('driver', '?')}, "
                f"method={cpufreq.get('scaling_method', '?')}, "
                f"control_model={cpufreq.get('control_model', '?')}",
            )
    elif cpufreq.get("driver"):
        caps["cpu_control"] = _cap(
            UNAVAILABLE,
            WARNING,
            f"cpufreq driver loaded ({cpufreq['driver']}) but governor "
            f"not writable",
            reason=(
                f"Write permission denied on {CPUFREQ_GOVERNOR_PATH}"
            ),
            recommendation=(
                "Run with elevated privileges, or configure write access "
                "via udev rule or sudo for cpupower"
            ),
        )
    else:
        caps["cpu_control"] = _cap(
            UNAVAILABLE,
            WARNING,
            "cpufreq interface not available",
            reason="No cpufreq driver loaded; CPU frequency control impossible",
            recommendation=(
                "Load acpi-cpufreq or intel_pstate driver; check kernel config"
            ),
        )

    # --- gpu_discovery ---
    gpu_tax = gpu.get("taxonomy", {})
    if gpu_tax.get("present"):
        nv_count = len(gpu.get("by_vendor", {}).get("nvidia", []))
        intel_count = len(gpu.get("by_vendor", {}).get("intel", []))
        caps["gpu_discovery"] = _cap(
            AVAILABLE,
            INFO,
            f"{gpu.get('device_count', 0)} GPU(s) detected "
            f"(NVIDIA: {nv_count}, Intel: {intel_count})",
        )
    else:
        caps["gpu_discovery"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            "No GPU detected via nvidia-smi or lspci",
            reason="No NVIDIA or Intel display adapter found on PCI bus",
            recommendation=(
                "Verify GPU is physically installed and detected by lspci; "
                "install lspci (pciutils) if missing"
            ),
        )

    # --- gpu_telemetry ---
    gpu_metrics = probes.get("gpu_metrics", {})
    if gpu_tax.get("observable") and gpu_metrics.get("success"):
        caps["gpu_telemetry"] = _cap(
            AVAILABLE,
            INFO,
            f"nvidia-smi metrics accessible: "
            f"temp={gpu_metrics.get('temperature_c', '?')}C, "
            f"power={gpu_metrics.get('power_draw_w', '?')}W",
        )
    elif gpu_tax.get("observable"):
        caps["gpu_telemetry"] = _cap(
            DEGRADED,
            WARNING,
            "nvidia-smi reachable but metric query returned no data",
            reason="GPU metrics probe failed; telemetry may be intermittent",
            recommendation=(
                "Check nvidia-smi --query-gpu=temperature.gpu,power.draw "
                "manually; ensure NVIDIA driver is correctly loaded"
            ),
        )
    elif gpu_tax.get("present"):
        caps["gpu_telemetry"] = _cap(
            UNAVAILABLE,
            WARNING,
            "GPU detected via lspci but nvidia-smi is unavailable",
            reason="Without nvidia-smi/NVML, GPU telemetry is not possible",
            recommendation=(
                "Install the NVIDIA driver and nvidia-smi, or use NVML "
                "bindings directly"
            ),
        )
    else:
        caps["gpu_telemetry"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            "No GPU present; telemetry not applicable",
            reason="No GPU hardware detected",
            recommendation="Install a supported GPU",
        )

    # --- gpu_control ---
    if gpu_tax.get("controllable"):
        caps["gpu_control"] = _cap(
            AVAILABLE,
            INFO,
            "nvidia-smi supported clocks queryable; application clock "
            "control available",
        )
    elif gpu_tax.get("observable"):
        caps["gpu_control"] = _cap(
            UNAVAILABLE,
            WARNING,
            "GPU observable but clock control unavailable",
            reason=gpu_tax.get("reason", "clock control not available"),
            recommendation=(
                "Run nvidia-smi as root, enable persistence mode "
                "(nvidia-smi -pm 1), or verify GPU supports -ac clocks"
            ),
        )
    elif gpu_tax.get("present"):
        caps["gpu_control"] = _cap(
            UNAVAILABLE,
            WARNING,
            "GPU present but not observable; control impossible",
            reason=gpu_tax.get("reason", "nvidia-smi unavailable"),
            recommendation="Install NVIDIA driver and nvidia-smi",
        )
    else:
        caps["gpu_control"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            "No GPU present; control not applicable",
            reason="No GPU hardware detected",
            recommendation="Install a supported GPU",
        )

    # --- topology_awareness ---
    if numa.get("available") and tools.get("numactl"):
        node_count = len(numa.get("nodes", {}))
        caps["topology_awareness"] = _cap(
            AVAILABLE,
            INFO,
            f"{node_count} NUMA node(s) detected; numactl available "
            f"for pinning",
        )
    elif numa.get("available"):
        caps["topology_awareness"] = _cap(
            DEGRADED,
            WARNING,
            f"NUMA topology detected ({len(numa.get('nodes', {}))}) node(s) "
            f"but numactl not in PATH",
            reason="NUMA-aware process pinning requires numactl",
            recommendation="Install numactl: sudo apt install numactl",
        )
    else:
        caps["topology_awareness"] = _cap(
            UNAVAILABLE,
            INFO,
            "Single NUMA node or NUMA information unavailable",
            reason="System may be UMA or NUMA info inaccessible",
            recommendation=(
                "Not blocking; single-socket systems typically do not "
                "require NUMA awareness"
            ),
        )

    # --- full_pipeline ---
    critical_caps = [
        "cpu_discovery",
        "cpu_telemetry",
        "energy_cpu",
        "gpu_discovery",
        "gpu_telemetry",
    ]
    control_caps = ["cpu_control", "gpu_control"]

    all_critical_ok = all(
        caps.get(c, {}).get("status") == AVAILABLE for c in critical_caps
    )
    all_control_ok = all(
        caps.get(c, {}).get("status") == AVAILABLE for c in control_caps
    )

    if all_critical_ok and all_control_ok:
        caps["full_pipeline"] = _cap(
            AVAILABLE,
            INFO,
            "All telemetry and control capabilities fully available",
        )
    elif all_critical_ok:
        degraded_names = [
            c
            for c in control_caps
            if caps.get(c, {}).get("status") != AVAILABLE
        ]
        caps["full_pipeline"] = _cap(
            DEGRADED,
            WARNING,
            f"Telemetry available; control capabilities missing: "
            f"{', '.join(degraded_names)}",
            reason="Full DVFS experiment pipeline requires control capabilities",
            recommendation=(
                "Elevate privileges for full DVFS control capability"
            ),
        )
    else:
        missing = [
            c
            for c in critical_caps
            if caps.get(c, {}).get("status") != AVAILABLE
        ]
        caps["full_pipeline"] = _cap(
            UNAVAILABLE,
            CRITICAL,
            f"Telemetry baseline not met; missing: {', '.join(missing)}",
            reason="Core telemetry capabilities unavailable",
            recommendation="Resolve critical issues in listed capabilities first",
        )

    return caps


# ---------------------------------------------------------------------------
# Backends and experiment blockers
# ---------------------------------------------------------------------------


def build_backends(
    capabilities: Dict[str, CapabilityResult],
    cpu: Dict[str, Any],
    gpu: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive available telemetry and control backend flags."""
    cpufreq = cpu.get("cpufreq", {})
    return {
        "telemetry": {
            "cpu_perf": capabilities.get("cpu_telemetry", {}).get("status")
            == AVAILABLE,
            "cpu_rapl": capabilities.get("energy_cpu", {}).get("status")
            in (AVAILABLE, DEGRADED),
            "gpu_nvml": capabilities.get("gpu_telemetry", {}).get("status")
            == AVAILABLE,
        },
        "control": {
            "cpu_cpufreq": capabilities.get("cpu_control", {}).get("status")
            in (AVAILABLE, DEGRADED),
            "gpu_clocks": capabilities.get("gpu_control", {}).get("status")
            == AVAILABLE,
        },
        "observable_frequencies": {
            "method": cpufreq.get("scaling_method", "unknown"),
            "min_khz": cpufreq.get("min_khz", ""),
            "max_khz": cpufreq.get("max_khz", ""),
            "base_frequency_khz": cpufreq.get("base_frequency_khz", ""),
            "discrete_steps": cpufreq.get("available_frequencies_khz", []),
        },
        "energy_domains": cpu.get("rapl", {}).get("domains", []),
    }


def build_experiment_blockers(
    capabilities: Dict[str, CapabilityResult],
) -> List[Dict[str, str]]:
    """Extract experiment-blocking issues from capability results."""
    blockers: List[Dict[str, str]] = []
    for cap_name, cap_data in capabilities.items():
        if cap_data.get("severity") == CRITICAL and cap_data.get("status") != AVAILABLE:
            blockers.append(
                {
                    "severity": CRITICAL,
                    "capability": cap_name,
                    "message": cap_data.get("evidence", ""),
                    "recommendation": cap_data.get("recommendation", ""),
                }
            )
    return blockers


# ---------------------------------------------------------------------------
# Granular readiness evaluation
# ---------------------------------------------------------------------------


def evaluate_readiness(
    capabilities: Dict[str, CapabilityResult],
) -> Dict[str, Any]:
    """Evaluate per-capability readiness with classified issues.

    Returns eight granular readiness flags and a structured issue list
    classified by severity.
    """
    cap_ready_map = {
        "cpu_discovery_ready": "cpu_discovery",
        "cpu_telemetry_ready": "cpu_telemetry",
        "gpu_discovery_ready": "gpu_discovery",
        "gpu_telemetry_ready": "gpu_telemetry",
        "energy_cpu_ready": "energy_cpu",
        "cpu_control_ready": "cpu_control",
        "gpu_control_ready": "gpu_control",
        "full_pipeline_ready": "full_pipeline",
    }

    readiness: Dict[str, Any] = {}
    for ready_key, cap_key in cap_ready_map.items():
        status = capabilities.get(cap_key, {}).get("status", UNAVAILABLE)
        readiness[ready_key] = status == AVAILABLE

    # Collect all issues classified by severity.
    issues: List[Dict[str, Any]] = []
    for cap_name, cap_data in capabilities.items():
        if cap_data.get("status") == AVAILABLE:
            continue
        issues.append(
            {
                "severity": cap_data.get("severity", INFO),
                "capability": cap_name,
                "message": cap_data.get("evidence", ""),
                "reason": cap_data.get("reason", ""),
                "recommendation": cap_data.get("recommendation", ""),
            }
        )

    # Sort: critical first, then warning, then info.
    severity_order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

    readiness["issues"] = issues
    return readiness
