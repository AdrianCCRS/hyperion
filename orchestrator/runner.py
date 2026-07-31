from __future__ import annotations

from dataclasses import dataclass
import contextlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping

from .catalog import KernelEntry, resolve_exec_command, verify_binary
from .config import HarnessConfig, load_config
from .metadata_schema import merge_metadata

logger = logging.getLogger(__name__)

# RUN-03: the launcher's own timeout must never be the only guard. This is a
# generous multiple of the catalog's expected runtime so normal variance
# never triggers a false timeout kill.
SAFETY_MARGIN = 3.0

# RUN-04: after killing the process group, poll a few times before declaring
# the cleanup failed instead of giving up on the first check.
_GROUP_GONE_ATTEMPTS = 20
_GROUP_GONE_INTERVAL_SECONDS = 0.05


class RunTimeoutError(RuntimeError):
    """Raised when the measured process group could not be cleaned up."""


@dataclass(frozen=True)
class RunResult:
    """Outcome of one telemetry_kernel_launcher invocation (RUN-01..RUN-08)."""

    run_id: str
    kernel_ref: str
    freq_level_id: str
    repetition_index: int
    command: tuple[str, ...]
    exit_code: int
    timed_out: bool
    success: bool
    elapsed_seconds: float
    run_dir: Path
    stdout_path: Path
    stderr_path: Path
    metadata: Mapping[str, Any]


def build_run_id(campaign_id: str, kernel_ref: str, freq_level_id: str, repetition_index: int) -> str:
    """RUN-02: deterministic run_id, reproducible from the manifest alone."""
    return f"{campaign_id}__{kernel_ref}__{freq_level_id}__rep{repetition_index:02d}"


def _format_cpu_list(cpus: Iterable[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def build_command(
    entry: KernelEntry, manifest: Any, run_id: str, harness: HarnessConfig
) -> list[str]:
    """RUN-01: the launcher argv is always derived from the catalog entry and
    the manifest, never hardcoded for a specific kernel or campaign."""
    exec_command = resolve_exec_command(entry, harness)
    cores = manifest.cores
    command = [
        harness.binary_path,
        *exec_command,
        "--perf-cpus", _format_cpu_list(cores.delegated_cpus),
        "--collector-cpu", str(cores.collector_cpu),
        "--consumer-cpu", str(cores.consumer_cpu),
        "--interval-ns", str(manifest.interval_ns),
        "--output-dir", str(manifest.output_dir),
        "--run-id", run_id,
    ]
    # --cgroup-path is optional isolation only (CPP-05); it is never required
    # for perf to attach correctly.
    if manifest.cgroup_path:
        command += ["--cgroup-path", manifest.cgroup_path]
    if not manifest.perf_enabled:
        command.append("--no-perf")
    return command


def _resolve_timeout_seconds(entry: KernelEntry, manifest: Any) -> float:
    """RUN-03: expected_runtime_seconds x SAFETY_MARGIN, or the manifest's
    generic run timeout for entries that do not declare a runtime (e.g. some
    calibration kernels)."""
    if entry.expected_runtime_seconds is not None:
        return float(entry.expected_runtime_seconds) * SAFETY_MARGIN
    return float(manifest.timeouts_seconds.run)


def _terminate_process_group(pgid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_process_group_gone(
    pgid: int, attempts: int = _GROUP_GONE_ATTEMPTS, interval: float = _GROUP_GONE_INTERVAL_SECONDS
) -> bool:
    for _ in range(attempts):
        if not _process_group_alive(pgid):
            return True
        time.sleep(interval)
    return not _process_group_alive(pgid)


def _check_success(entry: KernelEntry, exit_code: int, stdout_path: Path) -> bool:
    """RUN-05: apply entry.success_check against the real result. catalog.py
    already validated the check's shape (CAT-03/C03) when the catalog loaded."""
    check = entry.success_check
    check_type = check.get("type")
    if check_type == "exit_code":
        return exit_code == check.get("expected", 0)
    if check_type == "stdout_regex":
        pattern = check.get("pattern")
        text = stdout_path.read_text(errors="replace")
        return re.search(pattern, text) is not None
    return False


def _read_launcher_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            return json.load(metadata_file)
    except FileNotFoundError:
        # The launcher only writes metadata.json on a clean exit; a killed or
        # crashed run legitimately has none. success is already False by the
        # time this matters.
        return {}


def _merge_metadata(
    launcher_metadata: Mapping[str, Any],
    entry: KernelEntry,
    manifest: Any,
    kernel_ref: str,
    freq_level_id: str,
    repetition_index: int,
    node_id: str | None,
    calibration_refs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """RUN-06: merge launcher metadata (samples_collected, push_retries,
    perf_attach_mode, measured_pids, ...) with orchestrator-level metadata,
    via metadata_schema.merge_metadata() (MET-01: never {**a, **b})."""
    orchestrator_fields: dict[str, Any] = {
        "campaign_id": manifest.campaign_id,
        "kernel_ref": kernel_ref,
        "kernel_suite": entry.suite,
        "kernel_role": entry.role,
        "freq_level_id": freq_level_id,
        "repetition_index": repetition_index,
        "node_id": node_id,
        "binary_checksum": entry.binary_checksum,
    }
    if calibration_refs:
        orchestrator_fields = merge_metadata(orchestrator_fields, calibration_refs, context="RUN-06")

    return merge_metadata(launcher_metadata, orchestrator_fields, context="RUN-06")


def run_single(
    entry: KernelEntry,
    manifest: Any,
    kernel_ref: str,
    freq_level_id: str,
    repetition_index: int,
    *,
    environment_profile: Any = None,
    harness: HarnessConfig | None = None,
    node_id: str | None = None,
    calibration_refs: Mapping[str, Any] | None = None,
    apply_frequency: Callable[[Any, Any, Any], None] | None = None,
) -> RunResult:
    """Run one telemetry_kernel_launcher invocation and collect its result.

    `apply_frequency`, if given, is only ever called when
    environment_profile.frequency_write_capable is True (RUN-08); freqctl.py
    is the caller's concern, run_single just enforces the gate.
    """
    harness = harness or load_config().harness

    # CAT-07: re-verify the binary on disk immediately before every run, not
    # only once during preflight.
    if not verify_binary(entry):
        raise ValueError(f"C02: checksum de {entry.exec_path!r} no coincide antes de ejecutar")

    if apply_frequency is not None:
        if environment_profile is not None and getattr(environment_profile, "frequency_write_capable", False):
            apply_frequency(manifest.cores.delegated_cpus, freq_level_id, environment_profile)
        else:
            logger.debug(
                "RUN-08: frequency_write_capable=False, omitiendo apply_frequency para %s/%s",
                kernel_ref,
                freq_level_id,
            )

    run_id = build_run_id(manifest.campaign_id, kernel_ref, freq_level_id, repetition_index)
    run_dir = Path(manifest.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(entry, manifest, run_id, harness)
    timeout_seconds = _resolve_timeout_seconds(entry, manifest)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    start = time.monotonic()
    timed_out = False
    # start_new_session=True makes the child its own process group leader, so
    # os.killpg(child.pid, ...) also reaches everything it forks (RUN-03/04).
    with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
        process = subprocess.Popen(
            command, stdout=stdout_file, stderr=stderr_file, start_new_session=True
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process.pid)
            exit_code = process.wait()
    elapsed = time.monotonic() - start

    # RUN-04: the next combination must not start with leftover processes
    # from this one, whether it timed out or exited on its own.
    group_gone = _wait_process_group_gone(process.pid)
    if not group_gone:
        _terminate_process_group(process.pid)
        group_gone = _wait_process_group_gone(process.pid, attempts=10)
    if not group_gone:
        raise RunTimeoutError(
            f"RUN-04: quedan procesos vivos en el grupo de {run_id} (pgid={process.pid})"
        )

    success = (not timed_out) and _check_success(entry, exit_code, stdout_path)

    launcher_metadata = _read_launcher_metadata(run_dir)
    metadata = _merge_metadata(
        launcher_metadata,
        entry,
        manifest,
        kernel_ref,
        freq_level_id,
        repetition_index,
        node_id,
        calibration_refs,
    )

    return RunResult(
        run_id=run_id,
        kernel_ref=kernel_ref,
        freq_level_id=freq_level_id,
        repetition_index=repetition_index,
        command=tuple(command),
        exit_code=exit_code,
        timed_out=timed_out,
        success=success,
        elapsed_seconds=elapsed,
        run_dir=run_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        metadata=metadata,
    )
