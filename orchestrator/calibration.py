from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from . import runner
from .catalog import KernelEntry
from .runner import RunResult

logger = logging.getLogger(__name__)

# D03/CAL-04: +/-40% around the manifest's declared datasheet peak values.
D03_TOLERANCE_FRACTION = 0.40

# CAL-10: default coefficient-of-variation threshold for reference stability.
DEFAULT_CV_THRESHOLD_PCT = 5.0

# CAL-09: the reference kernel must be repeated at least this many times to
# make a P95/CV% estimate meaningful.
MIN_REFERENCE_REPETITIONS = 5

_CALIBRATION_FREQ_LEVEL_ID = "F0_calibration"


class CalibrationError(RuntimeError):
    """Roofline/reference calibration could not be trusted (CAL-04/06/09)."""


@dataclass(frozen=True)
class RooflineCalibration:
    campaign_id: str
    timestamp: str
    delegated_cpus: str
    bw_pico_bytes_per_s: float
    p_pico_flops_per_s: float
    i_ridge_flops_per_byte: float
    stream_raw_output: str
    ert_raw_output: str
    plausibility_check_passed: bool
    plausibility_message: str = ""


@dataclass(frozen=True)
class CalibrationReferences:
    node_id: str
    ipc_p95: float
    ips_p95: float
    mpki_p95: float
    miss_rate_p95: float
    repetitions: int
    cv_pct: float
    accepted: bool


def _format_cpu_list(cpus: Sequence[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def _extract_metric(pattern: str, text: str, *, label: str) -> float:
    match = re.search(pattern, text)
    if not match or not match.groups():
        raise CalibrationError(
            f"CAL-02/CAL-03: no se pudo extraer {label} del stdout con el patrón {pattern!r}"
        )
    try:
        return float(match.group(1))
    except ValueError as error:
        raise CalibrationError(f"CAL-02/CAL-03: {label} no es numérico: {match.group(1)!r}") from error


def _check_plausibility(
    bw_pico: float, p_pico: float, datasheet: Mapping[str, float] | None
) -> tuple[bool, str]:
    """CAL-04/D03. An undeclared datasheet is a failed check, not a skipped
    one (ARC-20: never approve a check due to missing data)."""
    if not datasheet or "bw_pico_bytes_per_s" not in datasheet or "p_pico_flops_per_s" not in datasheet:
        return False, "D03: manifest.hardware_datasheet no declara bw_pico_bytes_per_s/p_pico_flops_per_s"

    def within_range(observed: float, expected: float) -> bool:
        return expected > 0 and abs(observed - expected) / expected <= D03_TOLERANCE_FRACTION

    bw_ok = within_range(bw_pico, datasheet["bw_pico_bytes_per_s"])
    p_ok = within_range(p_pico, datasheet["p_pico_flops_per_s"])
    if bw_ok and p_ok:
        return True, ""
    reasons = []
    if not bw_ok:
        reasons.append(
            f"BW_pico observado={bw_pico:.3e} fuera de ±{D03_TOLERANCE_FRACTION:.0%} "
            f"de {datasheet['bw_pico_bytes_per_s']:.3e}"
        )
    if not p_ok:
        reasons.append(
            f"P_pico observado={p_pico:.3e} fuera de ±{D03_TOLERANCE_FRACTION:.0%} "
            f"de {datasheet['p_pico_flops_per_s']:.3e}"
        )
    return False, "D03: " + "; ".join(reasons)


def write_calibration(calibration: RooflineCalibration, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "roofline_calibration.json"
    with path.open("w", encoding="utf-8") as calibration_file:
        json.dump(asdict(calibration), calibration_file, indent=2, sort_keys=True)
        calibration_file.write("\n")
    return path


def load_calibration(output_dir: str | Path) -> RooflineCalibration:
    """CAL-06: refuses (raises) a calibration that failed its plausibility
    check. postprocess.py must never label a window against an unverified
    I_ridge."""
    path = Path(output_dir) / "roofline_calibration.json"
    with path.open(encoding="utf-8") as calibration_file:
        data = json.load(calibration_file)
    calibration = RooflineCalibration(**data)
    if not calibration.plausibility_check_passed:
        raise CalibrationError(
            f"CAL-06: {path} tiene plausibility_check_passed=False: {calibration.plausibility_message}"
        )
    return calibration


def run_calibration(
    manifest: Any,
    catalog: Mapping[str, KernelEntry],
    *,
    environment_profile: Any = None,
    node_id: str | None = None,
    run_single: Callable[..., RunResult] = runner.run_single,
) -> RooflineCalibration:
    """CAL-01..05: run STREAM (bandwidth) and ERT (FLOPs) once each, extract
    their peaks from stdout (never PMU counters), compute I_ridge, and check
    D03 in this same function so an implausible calibration can never
    silently reach windows.csv (CAL-04: D03 failing is a blocking exception).

    Runs at the node's native/current state deliberately: on felix REF is
    already the max frequency (governor=performance), so CAL-01 holds without
    invoking freqctl. Wiring an explicit F0 pin here for tiers where
    frequency_write_capable=True is FRQ-07, still pending (see freqctl.py).
    """
    bandwidth_entry: tuple[str, KernelEntry] | None = None
    flops_entry: tuple[str, KernelEntry] | None = None
    for kernel_ref in manifest.calibration:
        entry = catalog[kernel_ref]
        if entry.reports_bandwidth_stdout:
            bandwidth_entry = (kernel_ref, entry)
        if entry.reports_flops_stdout:
            flops_entry = (kernel_ref, entry)
    if bandwidth_entry is None or flops_entry is None:
        raise CalibrationError(
            "CAL-02/CAL-03: manifest.calibration debe referenciar una entrada con "
            "reports_bandwidth_stdout y otra con reports_flops_stdout"
        )

    stream_ref, stream_kernel = bandwidth_entry
    ert_ref, ert_kernel = flops_entry

    stream_result = run_single(
        stream_kernel, manifest, stream_ref, _CALIBRATION_FREQ_LEVEL_ID, 0,
        environment_profile=environment_profile, node_id=node_id,
    )
    if not stream_result.success:
        raise CalibrationError(f"CAL-02: la calibración de ancho de banda ({stream_ref}) no tuvo éxito")

    ert_result = run_single(
        ert_kernel, manifest, ert_ref, _CALIBRATION_FREQ_LEVEL_ID, 0,
        environment_profile=environment_profile, node_id=node_id,
    )
    if not ert_result.success:
        raise CalibrationError(f"CAL-03: la calibración de FLOPs ({ert_ref}) no tuvo éxito")

    stream_raw = stream_result.stdout_path.read_text(errors="replace")
    ert_raw = ert_result.stdout_path.read_text(errors="replace")

    bw_pico = _extract_metric(stream_kernel.bandwidth_stdout_pattern, stream_raw, label="BW_pico")
    p_pico = _extract_metric(ert_kernel.flops_stdout_pattern, ert_raw, label="P_pico")
    if bw_pico <= 0:
        raise CalibrationError(f"CAL-04: BW_pico debe ser positivo, se obtuvo {bw_pico}")

    i_ridge = p_pico / bw_pico
    passed, message = _check_plausibility(bw_pico, p_pico, manifest.hardware_datasheet)

    calibration = RooflineCalibration(
        campaign_id=manifest.campaign_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        delegated_cpus=_format_cpu_list(manifest.cores.delegated_cpus),
        bw_pico_bytes_per_s=bw_pico,
        p_pico_flops_per_s=p_pico,
        i_ridge_flops_per_byte=i_ridge,
        stream_raw_output=stream_raw,
        ert_raw_output=ert_raw,
        plausibility_check_passed=passed,
        plausibility_message=message,
    )
    # Always persisted, even when the check fails: the artifact is the
    # evidence needed to investigate D03, load_calibration() is what refuses
    # to use it (CAL-06).
    write_calibration(calibration, manifest.output_dir)

    if not passed:
        raise CalibrationError(message)
    return calibration


def _read_final_cpu_counters(run_dir: Path) -> tuple[int, int, int, int] | None:
    """Last CPU row of samples.csv: perf counters are cumulative since
    IOC_RESET, so the last sample approximates the whole run's totals."""
    samples_path = run_dir / "samples.csv"
    try:
        with samples_path.open(newline="", encoding="utf-8") as samples_file:
            reader = csv.DictReader(samples_file)
            last_cpu_row = None
            for row in reader:
                if row.get("tag") == "CPU":
                    last_cpu_row = row
            if last_cpu_row is None:
                return None
            return (
                int(last_cpu_row["instructions"]),
                int(last_cpu_row["cycles"]),
                int(last_cpu_row["cache_references"]),
                int(last_cpu_row["cache_misses"]),
            )
    except (OSError, KeyError, ValueError):
        return None


def _run_metrics(result: RunResult) -> dict[str, float] | None:
    counters = _read_final_cpu_counters(result.run_dir)
    if counters is None:
        return None
    instructions, cycles, cache_references, cache_misses = counters
    if instructions <= 0 or cycles <= 0 or result.elapsed_seconds <= 0:
        return None
    return {
        "ipc": instructions / cycles,
        "ips": instructions / result.elapsed_seconds,
        "mpki": (cache_misses / instructions) * 1000.0,
        "miss_rate": (cache_misses / cache_references) if cache_references > 0 else 0.0,
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _cv_pct(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance ** 0.5 / mean) * 100.0


def build_calibration_references(
    calibration_runs: Sequence[RunResult],
    node_id: str,
    *,
    cv_threshold_pct: float = DEFAULT_CV_THRESHOLD_PCT,
) -> CalibrationReferences:
    """CAL-09/CAL-10: >=5 repetitions of a reference kernel -> P95 of
    IPC/IPS/MPKI/MissRate plus a stability check.

    The dataclass carries a single cv_pct/accepted pair for four metrics;
    cv_pct here is the worst (max) of the four per-metric coefficients of
    variation, so instability in any one signal disqualifies the reference
    instead of being averaged away.
    """
    if len(calibration_runs) < MIN_REFERENCE_REPETITIONS:
        raise CalibrationError(
            f"CAL-09: build_calibration_references requiere >={MIN_REFERENCE_REPETITIONS} "
            f"repeticiones, se dieron {len(calibration_runs)}"
        )

    metrics_by_key: dict[str, list[float]] = {"ipc": [], "ips": [], "mpki": [], "miss_rate": []}
    for result in calibration_runs:
        if not result.success:
            raise CalibrationError(f"CAL-09: la repetición {result.run_id} de referencia no tuvo éxito")
        metrics = _run_metrics(result)
        if metrics is None:
            raise CalibrationError(
                f"CAL-09: la repetición {result.run_id} no tiene contadores CPU utilizables en samples.csv"
            )
        for key, value in metrics.items():
            metrics_by_key[key].append(value)

    cv_pct = max(_cv_pct(values) for values in metrics_by_key.values())
    return CalibrationReferences(
        node_id=node_id,
        ipc_p95=_percentile(metrics_by_key["ipc"], 95),
        ips_p95=_percentile(metrics_by_key["ips"], 95),
        mpki_p95=_percentile(metrics_by_key["mpki"], 95),
        miss_rate_p95=_percentile(metrics_by_key["miss_rate"], 95),
        repetitions=len(calibration_runs),
        cv_pct=cv_pct,
        accepted=cv_pct <= cv_threshold_pct,
    )


def run_calibration_references(
    entry: KernelEntry,
    manifest: Any,
    kernel_ref: str,
    *,
    node_id: str,
    repetitions: int = MIN_REFERENCE_REPETITIONS,
    environment_profile: Any = None,
    cv_threshold_pct: float = DEFAULT_CV_THRESHOLD_PCT,
    run_single: Callable[..., RunResult] = runner.run_single,
) -> CalibrationReferences:
    """Runs the reference kernel `repetitions` times and delegates to
    build_calibration_references(). Persists calibration_references.json
    regardless of `accepted` (CAL-10/D04 is a warning, not a hard stop)."""
    if repetitions < MIN_REFERENCE_REPETITIONS:
        raise ValueError(f"CAL-09: repetitions debe ser >={MIN_REFERENCE_REPETITIONS}")

    runs = [
        run_single(
            entry, manifest, kernel_ref, _CALIBRATION_FREQ_LEVEL_ID, repetition,
            environment_profile=environment_profile, node_id=node_id,
        )
        for repetition in range(1, repetitions + 1)
    ]
    references = build_calibration_references(runs, node_id, cv_threshold_pct=cv_threshold_pct)
    write_calibration_references(references, manifest.output_dir)
    if not references.accepted:
        logger.warning(
            "CAL-10/D04: calibration_references cv_pct=%.2f%% supera el umbral %.2f%% (node_id=%s)",
            references.cv_pct, cv_threshold_pct, node_id,
        )
    return references


def write_calibration_references(references: CalibrationReferences, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "calibration_references.json"
    with path.open("w", encoding="utf-8") as references_file:
        json.dump(asdict(references), references_file, indent=2, sort_keys=True)
        references_file.write("\n")
    return path


def load_calibration_references(output_dir: str | Path) -> CalibrationReferences:
    path = Path(output_dir) / "calibration_references.json"
    with path.open(encoding="utf-8") as references_file:
        return CalibrationReferences(**json.load(references_file))
