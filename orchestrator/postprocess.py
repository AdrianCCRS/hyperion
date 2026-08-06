from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Sequence

from . import calibration as calibration_module
from . import node_profile as node_profile_module

# POST-XX ids below refer to docs/retoma/Guia_Maestra_Fase1_DVFS.md section
# 12.9. samples.csv columns come from telemetry_kernel_launcher.cpp's
# write_samples_csv(): perf counters are CUMULATIVE since IOC_RESET, so every
# window here is a delta between two consecutive same-tag rows, never a raw
# reading on its own.

REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "run_id", "repetition", "kernel_ref", "node_id", "phase_label_hint", "phase_label_train",
    "freq_level_id", "freq_khz_requested", "freq_khz_applied", "freq_khz_observed",
    "window_index", "t_start_ns", "t_end_ns", "delta_t_ns",
    "delta_instructions", "delta_cycles", "delta_cache_references", "delta_cache_misses",
    "delta_stalled_cycles_backend", "stall_backend_ratio",
    "delta_l2_lines_in_all", "bytes_moved_l2_proxy",
    "ipc", "llc_miss_rate", "mpki", "ips",
    "ipc_relative", "mpki_relative", "miss_rate_relative",
    "delta_running_ns", "delta_enabled_ns", "running_ratio",
    "pkg_delta_uj", "dram_delta_uj", "power_w", "energy_valid",
    "flops_window_estimate", "bytes_moved_window", "operational_intensity",
    "i_ridge_used", "roofline_calibration_ref", "node_profile_ref", "calibration_ref",
    "binary_checksum", "quality_status",
    # ARC-70: filas GPU (tag=GPU en samples.csv) -- ver build_windows() y el
    # comentario de quality_status="gpu_telemetry" abajo. Vacías en toda fila
    # derivada de CPU/ENERGY.
    "gpu_power_mw", "gpu_util_pct",
)

VALID_QUALITY_STATUSES = frozenset({
    "ok", "first_sample_no_delta", "warmup_excluded", "pmu_degraded",
    "energy_invalid", "no_freq_reading", "intensity_undefined",
    # ARC-70: una muestra NVML cruda (potencia/utilización), no una ventana
    # de CPU -- ninguno de los campos de Roofline/PMU de CPU aplica. Ver
    # docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md sección 3/4: estas
    # filas son las *features* de entrenamiento del futuro clasificador de
    # GPU (Fase 2), no se clasifican en compute/memory-bound aquí -- eso
    # requiere el modelo, todavía sin entrenar.
    "gpu_telemetry",
})

# Priority order used when more than one condition applies to the same
# window (only one quality_status column exists per row). Not specified as
# an explicit table anywhere in the guides; documented here as our own
# interpretation (see ARC changelog) — most fundamental data problem wins.
_QUALITY_PRIORITY: tuple[str, ...] = (
    "first_sample_no_delta",
    "pmu_degraded",
    "warmup_excluded",
    "intensity_undefined",
    "energy_invalid",
    "no_freq_reading",
)


@dataclass(frozen=True)
class WindowContext:
    """Everything about a run that is constant across all of its windows."""

    run_id: str
    repetition: int
    kernel_ref: str
    node_id: str
    phase_label_hint: str | None
    freq_level_id: str
    freq_khz_requested: int | None
    freq_khz_applied: int | None
    freq_khz_observed: int | None
    binary_checksum: str
    roofline_calibration_ref: str
    node_profile_ref: str
    calibration_ref: str
    i_ridge_flops_per_byte: float
    llc_line_size_bytes: int
    run_flops_total: float | None
    warmup_seconds: float
    running_ratio_min: float
    rapl_enabled: bool
    calibration_references: Any = None  # calibration.CalibrationReferences | None


def _read_rows(samples_csv_path: str | Path) -> list[dict[str, str]]:
    with Path(samples_csv_path).open(newline="", encoding="utf-8") as samples_file:
        return list(csv.DictReader(samples_file))


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _delta(cur: int | None, prev: int | None) -> int | None:
    if cur is None or prev is None:
        return None
    return cur - prev


# ARC-48: runner.py nunca pasa --repetitions al launcher, así que
# telemetry_kernel_launcher.cpp siempre usa su default (opt.repetitions=1):
# CADA invocación es un proceso nuevo cuyo bucle interno de repetición
# corre exactamente una vez, y por eso samples.csv SIEMPRE tiene "1" en su
# propia columna "repetition" -- sin importar si esta corrida es la
# repetición 1, 2 o 3 a nivel de campaña (campaign.py). Filtrar
# samples.csv por el repetition_index de la campaña (como hacía este
# código antes) solo encontraba filas cuando repetition_index==1; para
# repetition_index>=2 el filtro nunca matcheaba nada y windows.csv salía
# vacío en silencio -- encontrado en la primera campaña real de 3
# repeticiones (F4.4 extendido), afectaba el 100% de las repeticiones 2 y
# 3 de los 7 kernels (14 de 21 corridas).
_LAUNCHER_INTERNAL_REPETITION = 1


def _split_by_repetition_and_tag(
    rows: Sequence[dict[str, str]], repetition: int = _LAUNCHER_INTERNAL_REPETITION
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cpu_rows = [r for r in rows if r.get("tag") == "CPU" and _to_int(r.get("repetition")) == repetition]
    energy_rows = [r for r in rows if r.get("tag") == "ENERGY" and _to_int(r.get("repetition")) == repetition]
    cpu_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    energy_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    return cpu_rows, energy_rows


def _match_energy_windows(
    cpu_rows: Sequence[dict[str, str]], energy_rows: Sequence[dict[str, str]]
) -> list[tuple[dict[str, str], int | None] | None]:
    """One entry per window (cpu_rows[1:]): the ENERGY row whose own
    (already-computed) delta best represents that window, matched by a
    single forward pass since both lists are timestamp-sorted, paired with
    the REAL elapsed time that delta actually spans (this matched row's own
    timestamp minus the previous ENERGY row's timestamp -- RAPL's own
    sampling cadence, not the CPU window's).

    ARC-56: pkg_delta_uj/dram_delta_uj are computed by the launcher between
    two consecutive ENERGY samples, whose cadence does not have to line up
    with the CPU window it gets matched into. A CPU window can be
    anomalously short (sampling jitter) while the matched ENERGY delta still
    spans RAPL's normal ~1ms interval -- dividing that delta by the CPU
    window's own (tiny) delta_t_ns produced power_w spikes into the tens of
    kilowatts, physically impossible for this hardware. The second element
    of each tuple is exactly the denominator power_w must use instead.
    """
    matches: list[tuple[dict[str, str], int | None] | None] = []
    energy_idx = 0
    for i in range(1, len(cpu_rows)):
        window_start = int(cpu_rows[i - 1]["timestamp_ns"])
        window_end = int(cpu_rows[i]["timestamp_ns"])
        match: dict[str, str] | None = None
        match_idx: int | None = None
        while energy_idx < len(energy_rows) and int(energy_rows[energy_idx]["timestamp_ns"]) <= window_end:
            if int(energy_rows[energy_idx]["timestamp_ns"]) > window_start:
                match = energy_rows[energy_idx]
                match_idx = energy_idx
            energy_idx += 1
        if match is None:
            matches.append(None)
            continue
        own_delta_ns = None
        if match_idx is not None and match_idx > 0:
            own_delta_ns = int(match["timestamp_ns"]) - int(energy_rows[match_idx - 1]["timestamp_ns"])
        matches.append((match, own_delta_ns))
    return matches


def _resolve_quality_status(flags: dict[str, bool]) -> str:
    for status in _QUALITY_PRIORITY:
        if flags.get(status):
            return status
    return "ok"


def _relative(value: float | None, reference: float | None) -> float | None:
    # POST-12/13: always computed when possible, never clipped to [0, 1].
    if value is None or reference is None or reference == 0:
        return None
    return value / reference


def build_windows(samples_csv_path: str | Path, context: WindowContext) -> list[dict[str, Any]]:
    """samples.csv -> list of window rows (POST-01..16). One dict per row of
    windows.csv, in REQUIRED_OUTPUT_COLUMNS order-compatible keys.
    """
    rows = _read_rows(samples_csv_path)
    # ARC-48: NUNCA context.repetition aquí -- ver el comentario de
    # _split_by_repetition_and_tag. context.repetition es la repetición a
    # nivel de campaña (para etiquetar windows.csv), no la numeración
    # interna del launcher dentro de ESTE samples.csv.
    cpu_rows, energy_rows = _split_by_repetition_and_tag(rows)
    if not cpu_rows:
        return []

    # ARC-50: stalled_cycles_backend is a per-NODE capability, not a
    # per-window one -- some kernel/PMU combinations (paccaA100, confirmed
    # empirically) never map this event, and the launcher writes it as an
    # empty column for every row of the run when that happens (never "0",
    # which would look like a real reading). Treat that uniform absence as
    # "not measured here", never as pmu_degraded; a genuinely missing value
    # on a node that DOES support the counter elsewhere in the same run is
    # still a real anomaly and keeps triggering pmu_degraded below.
    stall_backend_supported = any(
        r.get("stalled_cycles_backend") not in (None, "") for r in cpu_rows
    )
    # ARC-63: same per-node-capability rule as stalled_cycles_backend above --
    # L2_LINES_IN_ALL is a raw event only opened on Ice Lake-SP, empty (not
    # "0") for every row when unsupported.
    l2_lines_in_all_supported = any(
        r.get("l2_lines_in_all") not in (None, "") for r in cpu_rows
    )

    run_total_instructions = _to_int(cpu_rows[-1].get("instructions"))
    run_start_ns = int(cpu_rows[0]["timestamp_ns"])
    warmup_end_ns = run_start_ns + int(context.warmup_seconds * 1_000_000_000)
    energy_matches = _match_energy_windows(cpu_rows, energy_rows)

    windows: list[dict[str, Any]] = []

    # POST-01: the first sample has no predecessor to delta against. It is
    # still emitted (traceability, MET-07) with every delta/derived field
    # left unset rather than an imputed zero.
    first_row: dict[str, Any] = _base_row(context, window_index=0)
    first_row["t_start_ns"] = None
    first_row["t_end_ns"] = int(cpu_rows[0]["timestamp_ns"])
    first_row["delta_t_ns"] = None
    first_row["quality_status"] = "first_sample_no_delta"
    windows.append(first_row)

    for i in range(1, len(cpu_rows)):
        prev, cur = cpu_rows[i - 1], cpu_rows[i]
        row = _base_row(context, window_index=i)

        t_start_ns = int(prev["timestamp_ns"])
        t_end_ns = int(cur["timestamp_ns"])
        delta_t_ns = t_end_ns - t_start_ns
        row["t_start_ns"] = t_start_ns
        row["t_end_ns"] = t_end_ns
        row["delta_t_ns"] = delta_t_ns

        delta_instructions = _delta(_to_int(cur.get("instructions")), _to_int(prev.get("instructions")))
        delta_cycles = _delta(_to_int(cur.get("cycles")), _to_int(prev.get("cycles")))
        delta_cache_references = _delta(_to_int(cur.get("cache_references")), _to_int(prev.get("cache_references")))
        delta_cache_misses = _delta(_to_int(cur.get("cache_misses")), _to_int(prev.get("cache_misses")))
        delta_stalled_cycles_backend = (
            _delta(_to_int(cur.get("stalled_cycles_backend")), _to_int(prev.get("stalled_cycles_backend")))
            if stall_backend_supported else None
        )
        delta_l2_lines_in_all = (
            _delta(_to_int(cur.get("l2_lines_in_all")), _to_int(prev.get("l2_lines_in_all")))
            if l2_lines_in_all_supported else None
        )
        delta_running_ns = _delta(_to_int(cur.get("time_running_ns")), _to_int(prev.get("time_running_ns")))
        delta_enabled_ns = _delta(_to_int(cur.get("time_enabled_ns")), _to_int(prev.get("time_enabled_ns")))

        # POST-02: a negative delta means the counter wrapped or was reset
        # mid-window (no wrap-correction is attempted for perf counters,
        # unlike RAPL which the launcher already corrects). A missing field
        # is treated the same way: the window is kept but flagged
        # pmu_degraded, never silently fixed up or imputed. stalled_cycles_backend
        # only participates in this gate on nodes that support it (ARC-50) --
        # otherwise every window on an unsupported node would be flagged
        # degraded for a counter that was never going to exist.
        core_deltas = [delta_instructions, delta_cycles, delta_cache_references, delta_cache_misses]
        if stall_backend_supported:
            core_deltas.append(delta_stalled_cycles_backend)
        if l2_lines_in_all_supported:
            core_deltas.append(delta_l2_lines_in_all)
        counters_negative = any(value is not None and value < 0 for value in core_deltas)
        counters_missing = any(value is None for value in core_deltas)

        running_ratio = (
            delta_running_ns / delta_enabled_ns
            if delta_enabled_ns and delta_enabled_ns > 0 and delta_running_ns is not None
            else 0.0
        )
        pmu_degraded = counters_negative or counters_missing or running_ratio < context.running_ratio_min

        row["delta_instructions"] = delta_instructions
        row["delta_cycles"] = delta_cycles
        row["delta_cache_references"] = delta_cache_references
        row["delta_cache_misses"] = delta_cache_misses
        row["delta_stalled_cycles_backend"] = delta_stalled_cycles_backend
        row["delta_l2_lines_in_all"] = delta_l2_lines_in_all
        row["delta_running_ns"] = delta_running_ns
        row["delta_enabled_ns"] = delta_enabled_ns
        row["running_ratio"] = running_ratio

        valid_counters = not (counters_negative or counters_missing)
        # POST-04: rates always use the real measured delta_t_ns, never the
        # nominal --interval-ns.
        if valid_counters and delta_t_ns > 0:
            row["ips"] = delta_instructions / (delta_t_ns / 1_000_000_000)
            row["ipc"] = (delta_instructions / delta_cycles) if delta_cycles else None
            row["mpki"] = (delta_cache_misses / delta_instructions * 1000.0) if delta_instructions else None
            row["llc_miss_rate"] = (
                delta_cache_misses / delta_cache_references if delta_cache_references else None
            )
            # Fracción de ciclos parados en el backend (recursos de ejecución
            # ocupados, típicamente espera de memoria) -- la señal directa que
            # ARC-27 identificó como la que phase_label_train necesitaba y
            # que hasta ahora se rodeaba indirectamente via cache_misses.
            row["stall_backend_ratio"] = (
                delta_stalled_cycles_backend / delta_cycles
                if delta_cycles and delta_stalled_cycles_backend is not None
                else None
            )
        else:
            row["ips"] = row["ipc"] = row["mpki"] = row["llc_miss_rate"] = None
            row["stall_backend_ratio"] = None

        if context.calibration_references is not None:
            row["ipc_relative"] = _relative(row["ipc"], context.calibration_references.ipc_p95)
            row["mpki_relative"] = _relative(row["mpki"], context.calibration_references.mpki_p95)
            row["miss_rate_relative"] = _relative(row["llc_miss_rate"], context.calibration_references.miss_rate_p95)
        else:
            row["ipc_relative"] = row["mpki_relative"] = row["miss_rate_relative"] = None

        # POST-05/POST-06: the launcher already computed pkg_delta_uj with
        # wrap correction and its own validity bit; postprocess only
        # propagates it, it never treats an invalid/zero reading as real.
        energy_match = energy_matches[i - 1]
        energy_row = energy_match[0] if energy_match is not None else None
        energy_own_delta_ns = energy_match[1] if energy_match is not None else None
        energy_valid = False
        pkg_delta_uj = dram_delta_uj = power_w = None
        if context.rapl_enabled and energy_row is not None and energy_row.get("energy_delta_valid") == "1":
            pkg_delta_uj = _to_int(energy_row.get("pkg_delta_uj"))
            dram_delta_uj = _to_int(energy_row.get("dram_delta_uj"))
            # ARC-56: power_w usa el intervalo REAL que pkg_delta_uj abarca
            # (entre las dos muestras RAPL consecutivas que lo produjeron),
            # nunca delta_t_ns de la ventana CPU -- son cadencias de
            # muestreo independientes, y dividir por la ventana CPU cuando
            # es anómalamente corta producía potencias de decenas de kW,
            # físicamente imposibles. Sin ese intervalo propio (RAPL
            # arrancó en esta misma ventana, primera lectura), power_w
            # queda sin definir en vez de usar un denominador que no le
            # corresponde a esta medición.
            if pkg_delta_uj is not None and energy_own_delta_ns is not None and energy_own_delta_ns > 0:
                power_w = (pkg_delta_uj / 1_000_000.0) / (energy_own_delta_ns / 1_000_000_000.0)
            energy_valid = True
        row["pkg_delta_uj"] = pkg_delta_uj
        row["dram_delta_uj"] = dram_delta_uj
        row["power_w"] = power_w
        row["energy_valid"] = energy_valid
        energy_invalid = context.rapl_enabled and not energy_valid

        # POST-07: warmup windows are excluded from training but never
        # dropped from windows.csv.
        warmup_excluded = t_start_ns < warmup_end_ns

        # POST-08/POST-09/POST-10: FLOPs from the binary's own stdout,
        # prorated across windows proportionally to delta_instructions (a
        # declared approximation, not a PMU measurement — see module
        # docstring of run this came from in calibration.py/campaign.py).
        # bytes_moved_window uses the node_profile's real LLC line size.
        flops_window_estimate = None
        if (
            valid_counters
            and context.run_flops_total is not None
            and run_total_instructions
            and delta_instructions is not None
        ):
            flops_window_estimate = context.run_flops_total * (delta_instructions / run_total_instructions)

        bytes_moved_window = (
            delta_cache_misses * context.llc_line_size_bytes
            if valid_counters and delta_cache_misses is not None
            else None
        )
        row["flops_window_estimate"] = flops_window_estimate
        row["bytes_moved_window"] = bytes_moved_window

        # ARC-63: independent cross-check for bytes_moved_window's bias
        # (F3.4/ARC-33, ARC-60) -- same line-size multiplier convention, so
        # it is directly comparable to bytes_moved_window per window. Still
        # a core-level (L2) proxy, not real DRAM bytes (uncore stays
        # blocked, ARC-59); never used in operational_intensity/
        # phase_label_train, purely a reported cross-check column.
        row["bytes_moved_l2_proxy"] = (
            delta_l2_lines_in_all * context.llc_line_size_bytes
            if l2_lines_in_all_supported and delta_l2_lines_in_all is not None
            else None
        )

        intensity_undefined = (
            bytes_moved_window is None or bytes_moved_window == 0 or flops_window_estimate is None
        )
        if intensity_undefined:
            row["operational_intensity"] = float("nan")
            row["phase_label_train"] = None
        else:
            operational_intensity = flops_window_estimate / bytes_moved_window
            row["operational_intensity"] = operational_intensity
            # POST-11: always derived from Roofline, never copied from
            # phase_label_hint.
            row["phase_label_train"] = (
                "memory_bound" if operational_intensity < context.i_ridge_flops_per_byte else "compute_bound"
            )

        no_freq_reading = context.freq_khz_observed is None

        row["quality_status"] = _resolve_quality_status({
            "pmu_degraded": pmu_degraded,
            "warmup_excluded": warmup_excluded,
            "intensity_undefined": intensity_undefined,
            "energy_invalid": energy_invalid,
            "no_freq_reading": no_freq_reading,
        })
        windows.append(row)

    # ARC-70: filas GPU (tag=GPU, muestras NVML crudas) -- deliberadamente
    # NO se ventanean contra los límites de las ventanas de CPU de arriba.
    # A diferencia de CPU, no hay una intensidad operacional que calcular
    # aquí en vivo (ver Diseno_Politica_DVFS_CPU_GPU.md sección 3): NVML
    # solo expone potencia/utilización, nunca FLOPs ni bytes. Cada muestra
    # es un passthrough con el contexto de la corrida ya adjunto -- son las
    # *features* de entrenamiento del futuro clasificador de GPU (Fase 2),
    # no windows.csv "de GPU" en el mismo sentido que las de CPU.
    gpu_rows = [
        r for r in rows
        if r.get("tag") == "GPU" and _to_int(r.get("repetition")) == _LAUNCHER_INTERNAL_REPETITION
    ]
    gpu_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    for gpu_index, gpu_row in enumerate(gpu_rows):
        row = _base_row(context, window_index=gpu_index)
        row["t_start_ns"] = None
        row["t_end_ns"] = int(gpu_row["timestamp_ns"])
        row["delta_t_ns"] = None
        row["gpu_power_mw"] = _to_int(gpu_row.get("gpu_power_mw"))
        row["gpu_util_pct"] = _to_int(gpu_row.get("gpu_util_pct"))
        row["quality_status"] = "gpu_telemetry"
        windows.append(row)

    return windows


def _base_row(context: WindowContext, *, window_index: int) -> dict[str, Any]:
    # POST-14/MLT-01: traceability columns present on every single row.
    return {
        "run_id": context.run_id,
        "repetition": context.repetition,
        "kernel_ref": context.kernel_ref,
        "node_id": context.node_id,
        "phase_label_hint": context.phase_label_hint,
        "phase_label_train": None,
        "freq_level_id": context.freq_level_id,
        "freq_khz_requested": context.freq_khz_requested,
        "freq_khz_applied": context.freq_khz_applied,
        "freq_khz_observed": context.freq_khz_observed,
        "window_index": window_index,
        "delta_instructions": None,
        "delta_cycles": None,
        "delta_cache_references": None,
        "delta_cache_misses": None,
        "delta_stalled_cycles_backend": None,
        "stall_backend_ratio": None,
        "delta_l2_lines_in_all": None,
        "bytes_moved_l2_proxy": None,
        "ipc": None,
        "llc_miss_rate": None,
        "mpki": None,
        "ips": None,
        "ipc_relative": None,
        "mpki_relative": None,
        "miss_rate_relative": None,
        "delta_running_ns": None,
        "delta_enabled_ns": None,
        "running_ratio": None,
        "pkg_delta_uj": None,
        "dram_delta_uj": None,
        "power_w": None,
        "energy_valid": False,
        "flops_window_estimate": None,
        "bytes_moved_window": None,
        "operational_intensity": None,
        "i_ridge_used": context.i_ridge_flops_per_byte,
        "roofline_calibration_ref": context.roofline_calibration_ref,
        "node_profile_ref": context.node_profile_ref,
        "calibration_ref": context.calibration_ref,
        "binary_checksum": context.binary_checksum,
        "quality_status": "ok",
        "gpu_power_mw": None,
        "gpu_util_pct": None,
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def write_windows_csv(windows: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    """POST-16: every REQUIRED_OUTPUT_COLUMNS column is written for every
    row, so absolute and relative features are always both present."""
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as windows_file:
        writer = csv.writer(windows_file)
        writer.writerow(REQUIRED_OUTPUT_COLUMNS)
        for row in windows:
            status = row.get("quality_status", "ok")
            if status not in VALID_QUALITY_STATUSES:
                raise ValueError(f"quality_status inválido: {status!r}")
            writer.writerow([_format_cell(row.get(column)) for column in REQUIRED_OUTPUT_COLUMNS])
    return path


def _extract_stdout_number(pattern: str | None, stdout_text: str) -> float | None:
    if not pattern:
        return None
    match = re.search(pattern, stdout_text)
    if not match or not match.groups():
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_run_flops_total(kernel_entry: Any, stdout_text: str) -> float | None:
    """POST-09: total FLOPs the kernel itself reported on stdout, never a PMU
    counter. Returns None (never raises) when the catalog entry has no
    flops_total_stdout_pattern or the pattern does not match, so the caller
    ends up with quality_status="intensity_undefined" windows instead of a
    hard failure.

    F3.2: NPB (confirmed on felix) never prints an absolute FLOP total, only
    a rate ("Mop/s total") and the run duration ("Time in seconds"). When
    flops_total_stdout_pattern is absent, fall back to
    flops_rate_stdout_pattern (Mop/s) x 1e6 x runtime_seconds_stdout_pattern
    (seconds) — both regexes read from the kernel's own stdout, still never
    a PMU counter.
    """
    total = _extract_stdout_number(getattr(kernel_entry, "flops_total_stdout_pattern", None), stdout_text)
    if total is not None:
        return total
    rate = _extract_stdout_number(getattr(kernel_entry, "flops_rate_stdout_pattern", None), stdout_text)
    runtime = _extract_stdout_number(getattr(kernel_entry, "runtime_seconds_stdout_pattern", None), stdout_text)
    if rate is None or runtime is None:
        return None
    return rate * 1e6 * runtime


def run_postprocess(
    run_dir: str | Path,
    *,
    run_id: str,
    repetition: int,
    kernel_ref: str,
    kernel_entry: Any,
    node_id: str,
    freq_level_id: str,
    calibration_dir: str | Path,
    freq_khz_requested: int | None = None,
    freq_khz_applied: int | None = None,
    freq_khz_observed: int | None = None,
    warmup_seconds: float = 0.0,
    running_ratio_min: float = 0.9,
    rapl_enabled: bool = False,
    calibration_references: Any = None,
) -> Path:
    """Orchestrates one run's samples.csv -> windows.csv.

    POST-15: calibration.load_calibration() refuses (raises) a calibration
    whose D03 plausibility check failed, so an unverified I_ridge can never
    reach this far. POST-10: the LLC line size always comes from
    node_profile.json, never a hardcoded constant.
    """
    roofline = calibration_module.load_calibration(calibration_dir)
    profile = node_profile_module.load_node_profile(calibration_dir)

    run_dir = Path(run_dir)
    stdout_text = (run_dir / "stdout.txt").read_text(errors="replace") if (run_dir / "stdout.txt").exists() else ""
    run_flops_total = extract_run_flops_total(kernel_entry, stdout_text)

    context = WindowContext(
        run_id=run_id,
        repetition=repetition,
        kernel_ref=kernel_ref,
        node_id=node_id,
        phase_label_hint=getattr(kernel_entry, "phase_label_hint", None),
        freq_level_id=freq_level_id,
        freq_khz_requested=freq_khz_requested,
        freq_khz_applied=freq_khz_applied,
        freq_khz_observed=freq_khz_observed,
        binary_checksum=kernel_entry.binary_checksum,
        roofline_calibration_ref=str(Path(calibration_dir) / "roofline_calibration.json"),
        node_profile_ref=str(Path(calibration_dir) / "node_profile.json"),
        calibration_ref=str(Path(calibration_dir) / "calibration_references.json"),
        i_ridge_flops_per_byte=roofline.i_ridge_flops_per_byte,
        llc_line_size_bytes=profile.cache_line_size_bytes,
        run_flops_total=run_flops_total,
        warmup_seconds=warmup_seconds,
        running_ratio_min=running_ratio_min,
        rapl_enabled=rapl_enabled,
        calibration_references=calibration_references,
    )
    windows = build_windows(run_dir / "samples.csv", context)
    return write_windows_csv(windows, run_dir / "windows.csv")
