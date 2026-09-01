from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmdResult:
    """Immutable result of a subprocess invocation.

    Attributes:
        returncode: Exit code returned by the process. ``-1`` indicates that
            the command timed out or raised an unexpected exception.
        stdout: Captured standard output (stripped of leading/trailing
            whitespace).
        stderr: Captured standard error (stripped of leading/trailing
            whitespace).
    """

    returncode: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Severity levels used across modules.
CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

# Capability status constants.
AVAILABLE = "available"
UNAVAILABLE = "unavailable"
DEGRADED = "degraded"

# Common sysfs paths.
CPUFREQ_PATH: Path = Path("/sys/devices/system/cpu/cpu0/cpufreq")
CPUFREQ_GOVERNOR_PATH: Path = CPUFREQ_PATH / "scaling_governor"
POWERCAP_PATH: Path = Path("/sys/class/powercap")
SYSFS_NODE_PATH: Path = Path("/sys/devices/system/node")
INTEL_PSTATE_PATH: Path = Path("/sys/devices/system/cpu/intel_pstate")
PERF_PARANOID_PATH: Path = Path("/proc/sys/kernel/perf_event_paranoid")
EPP_PATH: Path = Path(
    "/sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference"
)

# GPU device classes reported by lspci that indicate a display adapter.
LSPCI_DISPLAY_CLASSES: tuple[str, ...] = ("vga", "3d", "display")

# Compiled regex for ``numactl --hardware`` node-CPU lines.
NUMACTL_NODE_RE: re.Pattern[str] = re.compile(r"^node\s+(\d+)\s+cpus:\s+(.+)$")

# ---------------------------------------------------------------------------
# Type aliases shared across modules
# ---------------------------------------------------------------------------

#: A single GPU device record.
GpuDevice = Dict[str, Any]

#: Vendor-keyed map of GPU device lists.
VendorMap = Dict[str, List[GpuDevice]]

#: Structured capability check result.
CapabilityResult = Dict[str, Any]

#: Taxonomy describing a resource's three access levels.
ResourceTaxonomy = Dict[str, Any]


# ---------------------------------------------------------------------------
# Taxonomy builder
# ---------------------------------------------------------------------------


def build_taxonomy(
    present: bool,
    observable: bool,
    controllable: bool,
    reason: Optional[str] = None,
) -> ResourceTaxonomy:
    """Build a presence / observability / controllability taxonomy dict."""
    return {
        "present": present,
        "observable": observable,
        "controllable": controllable,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# General hardware utilities
# ---------------------------------------------------------------------------


class hardwareutils:
    """General hardware utility helpers – Hyperion."""

    @staticmethod
    def run_cmd(cmd: List[str], timeout: int = 10) -> CmdResult:
        """Execute a subprocess command and capture its output.

        The command is run without a shell to avoid injection risks.  If the
        process does not finish within *timeout* seconds, or if an unexpected
        exception is raised, a :class:`CmdResult` with ``returncode=-1`` is
        returned instead of propagating the exception.

        Args:
            cmd: Argument list passed directly to :func:`subprocess.run`
                (e.g. ``["lscpu"]`` or ``["nvidia-smi", "--query-gpu=name"]``).
            timeout: Maximum number of seconds to wait for the process to
                complete.  Defaults to ``10``.

        Returns:
            A :class:`CmdResult` whose fields contain the process exit code
            and the stripped *stdout* / *stderr* strings.
        """
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CmdResult(
                returncode=result.returncode,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return CmdResult(-1, "", f"Command timed out after {timeout}s")
        except Exception as exc:  # pragma: no cover - defensive fallback
            return CmdResult(-1, "", f"Command execution failed: {exc}")

    @staticmethod
    def read_text(path: Path) -> str:
        """Read a text file and return its contents stripped of whitespace.

        Any :exc:`OSError` or related exception (e.g. permission denied, file
        not found) is silently swallowed and an empty string is returned, making
        the function safe to call against sysfs/procfs paths that may or may not
        exist.

        Args:
            path: Absolute or relative :class:`~pathlib.Path` to the file.

        Returns:
            The file contents with leading and trailing whitespace removed, or
            an empty string if the file cannot be read.
        """
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    @staticmethod
    def as_int(value: str) -> Optional[int]:
        """Convert a string to an integer without raising on failure.

        Args:
            value: The string to convert (e.g. ``"42"`` or ``"N/A"``).

        Returns:
            The integer representation of *value*, or ``None`` if the conversion
            is not possible.
        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def parse_cpu_list(cpu_list: str) -> List[int]:
        """Parse a Linux CPU-list string into a sorted list of CPU indices.

        Linux exposes CPU sets in a compact range notation such as
        ``"0-3,8,10-11"``.  This function expands every range and individual
        entry into a flat list of integers.

        Args:
            cpu_list: A CPU-list string as found in sysfs files like
                ``/sys/devices/system/node/node0/cpulist`` or in the output of
                ``numactl --hardware`` (e.g. ``"0-3,8,10-11"``).

        Returns:
            A list of integer CPU indices in the order they were encountered.
            Malformed chunks are silently skipped.

        Examples:
            >>> hardwareutils.parse_cpu_list("0-3,8,10-11")
            [0, 1, 2, 3, 8, 10, 11]
            >>> hardwareutils.parse_cpu_list("0")
            [0]
        """
        cpus: List[int] = []
        for chunk in cpu_list.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                try:
                    start, end = chunk.split("-", 1)
                    cpus.extend(range(int(start), int(end) + 1))
                except ValueError:
                    continue
            else:
                try:
                    cpus.append(int(chunk))
                except (TypeError, ValueError):
                    continue
        return cpus

    @staticmethod
    def parse_supported_clocks(output: str) -> Dict[str, List[int]]:
        """Parse the ``nvidia-smi -q -d SUPPORTED_CLOCKS`` text output.

        The command produces blocks like::

            Supported Clocks
                Memory                      : 9251 MHz
                Graphics                    : 1980 MHz
                Graphics                    : 1965 MHz
                ...

        This function collects all unique memory and graphics frequencies from
        the entire output (across all GPUs if multiple are present) and returns
        them as sorted lists.

        Args:
            output: Raw text produced by
                ``nvidia-smi -q -d SUPPORTED_CLOCKS`` (or a per-GPU variant
                with ``-i <index>``).

        Returns:
            A dictionary with two keys:

            - ``"memory"`` – sorted list of unique memory clock values in MHz.
            - ``"graphics"`` – sorted list of unique graphics clock values in MHz.
        """
        memory_clocks: List[int] = []
        graphics_clocks: List[int] = []

        for line in output.splitlines():
            line = line.strip()
            mem_match = re.match(r"^Memory\s*:\s*(\d+)\s*MHz", line)
            gr_match = re.match(r"^Graphics\s*:\s*(\d+)\s*MHz", line)
            if mem_match:
                memory_clocks.append(int(mem_match.group(1)))
            elif gr_match:
                graphics_clocks.append(int(gr_match.group(1)))

        return {
            "memory": sorted(set(memory_clocks)),
            "graphics": sorted(set(graphics_clocks)),
        }

    @staticmethod
    def parse_lscpu_output(output: str) -> Dict[str, str]:
        """Parse the text output of ``lscpu`` into a key/value dictionary.

        ``lscpu`` emits lines of the form ``Key:   Value``.  Lines that do
        not contain a colon are ignored.

        Args:
            output: The raw string captured from ``lscpu`` stdout.

        Returns:
            A dictionary mapping each field name (left of the first ``:``)
            to its value (right of the first ``:``) with surrounding
            whitespace stripped from both sides.

        Examples:
            >>> hardwareutils.parse_lscpu_output("Architecture:    x86_64\\nCPU(s):    8")
            {'Architecture': 'x86_64', 'CPU(s)': '8'}
        """
        data: Dict[str, str] = {}

        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

        return data