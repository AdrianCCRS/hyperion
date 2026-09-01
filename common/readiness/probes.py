"""Active probes for effective validation of hardware interfaces.

- Hyperion

Each probe performs a minimal, read-only operation to confirm that a
hardware interface is not only nominally present but effectively
accessible by the current process.
"""

from __future__ import annotations

from shutil import which
from typing import Any, Dict

from utils import (
    CPUFREQ_PATH,
    PERF_PARANOID_PATH,
    hardwareutils,
)


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def probe_perf(tools: Dict[str, bool]) -> Dict[str, Any]:
    """Probe effective ``perf`` accessibility.

    Reads ``perf_event_paranoid`` and attempts a minimal ``perf stat``
    invocation to confirm the binary can actually collect HW counters.
    """
    probe: Dict[str, Any] = {
        "event_paranoid_level": None,
        "accessible": False,
        "tested_with": "",
        "note": "",
    }

    paranoid_raw = hardwareutils.read_text(PERF_PARANOID_PATH)
    paranoid = hardwareutils.as_int(paranoid_raw)
    probe["event_paranoid_level"] = paranoid

    if not tools.get("perf"):
        probe["note"] = "perf binary not found in PATH"
        return probe

    result = hardwareutils.run_cmd(
        ["perf", "stat", "-e", "cycles", "--", "true"], timeout=10
    )
    probe["tested_with"] = "perf stat -e cycles -- true"

    if result.returncode == 0:
        probe["accessible"] = True
        if paranoid is not None and paranoid >= 2:
            probe["note"] = (
                f"perf_event_paranoid={paranoid}: kernel profiling restricted "
                "for non-root users but basic HW counters accessible"
            )
        else:
            probe["note"] = "perf accessible without restrictions"
    else:
        probe["note"] = (
            f"perf stat failed (rc={result.returncode}): "
            f"{result.stderr or 'unknown error'}"
        )
        if paranoid is not None and paranoid >= 3:
            probe["note"] += (
                f"; perf_event_paranoid={paranoid} blocks all non-root access"
            )

    return probe


def probe_cpu_frequency() -> Dict[str, Any]:
    """Read the current CPU frequency from sysfs as an active probe."""
    probe: Dict[str, Any] = {
        "success": False,
        "current_khz": "",
    }
    cur_freq = hardwareutils.read_text(CPUFREQ_PATH / "scaling_cur_freq")
    if cur_freq and cur_freq.isdigit():
        probe["success"] = True
        probe["current_khz"] = cur_freq
    return probe


def probe_gpu_metrics(gpu: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to read minimal GPU metrics via nvidia-smi."""
    probe: Dict[str, Any] = {
        "success": False,
        "temperature_c": "",
        "power_draw_w": "",
    }

    if not gpu.get("nvidia_present") or not which("nvidia-smi"):
        return probe

    result = hardwareutils.run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0 or not result.stdout:
        return probe

    parts = [p.strip() for p in result.stdout.splitlines()[0].split(",")]
    if len(parts) >= 2:
        probe["success"] = True
        probe["temperature_c"] = parts[0]
        probe["power_draw_w"] = parts[1]

    return probe


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def collect_active_probes(
    tools: Dict[str, bool],
    gpu: Dict[str, Any],
    cpu: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all active probes and return aggregated results."""
    return {
        "perf": probe_perf(tools),
        "rapl_read": cpu.get("rapl", {}).get("probe", {}),
        "cpu_frequency": probe_cpu_frequency(),
        "gpu_metrics": probe_gpu_metrics(gpu),
    }
