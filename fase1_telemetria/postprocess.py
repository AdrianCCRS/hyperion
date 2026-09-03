from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from . import calibration as calibration_module
from common.hpc import node_profile as node_profile_module
from . import validation as validation_module

# POST-XX ids below refer to docs/retoma/Guia_Maestra_Fase1_DVFS.md section
# 12.9. samples.csv columns come from telemetry_kernel_launcher.cpp's
# write_samples_csv(): perf counters are CUMULATIVE since IOC_RESET, so every
# window here is a delta between two consecutive same-tag rows, never a raw
# reading on its own.

REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "run_id", "repetition", "kernel_ref", "node_id", "phase_label_hint", "phase_label_train",
    "freq_level_id", "gpu_freq_level_id", "freq_khz_requested", "freq_khz_applied", "freq_khz_observed",
    # ARC-142: max-min entre los CPUs delegados que reportaron lectura --
    # ver _observed_freq_spread(). None cuando hay menos de 2 lecturas
    # válidas esa ventana, nunca un 0 fabricado.
    "freq_khz_observed_spread",
    # ARC-174: clasificación de frecuencia POR VENTANA (reemplaza el gate
    # agregado de tolerancia que existía en validate_cpu_frequency_trace()
    # -- ver validation.classify_frequency_window()). Vacías para filas GPU
    # (tag=GPU no tiene per-CPU scaling_cur_freq_khz_all que clasificar).
    "frequency_quality_status", "frequency_outlier_cpu_count",
    "frequency_min_khz", "frequency_max_khz", "frequency_max_relative_error",
    "window_index", "t_start_ns", "t_end_ns", "delta_t_ns",
    "delta_instructions", "delta_cycles", "delta_cache_references", "delta_cache_misses",
    "delta_stalled_cycles_mem_any", "stall_mem_ratio",
    "delta_l2_lines_in_all", "bytes_moved_l2_proxy",
    "ipc", "llc_miss_rate", "mpki", "ips",
    "ipc_relative", "mpki_relative", "miss_rate_relative",
    "delta_running_ns", "delta_enabled_ns", "running_ratio",
    "pkg_delta_uj", "dram_delta_uj", "power_w", "energy_valid",
    # ARC-97/100: FLOPs measured directly by hardware (FP_ARITH_INST_RETIRED,
    # Ice Lake-SP only), sole source of flops_measured_window -- no
    # instruction-prorated fallback (see build_windows()).
    "flops_measured_window",
    "bytes_moved_window", "operational_intensity",
    # ARC-119/123: uncore_imc CAS_COUNT_READ/WRITE (bytes reales de DRAM,
    # ámbito sistema/socket -- ver telemetry/src/uncore_reader.cpp). A
    # diferencia de bytes_moved_window, cada lectura de uncore ya es un
    # delta de un intervalo de `perf stat -I` (~10ms o más), más ancho que
    # una ventana de CPU (~1ms) -- NO se puede atribuir a una sola ventana
    # sin agregar. Estas tres columnas se calculan a la granularidad real
    # del intervalo de perf (sumando flops_measured_window de todas las
    # ventanas de CPU cubiertas) y se difunden sin cambios a cada una de
    # esas ventanas. ARC-123: operational_intensity/phase_label_train
    # arriba se deciden EXCLUSIVAMENTE con esto -- bytes_moved_window (el
    # proxy) nunca las alimenta, ni siquiera como respaldo: una ventana sin
    # cobertura real de uncore queda quality_status="intensity_undefined"
    # en vez de mezclar una medición real con una sesgada por no ver
    # tráfico de prefetch. Ver _finalize_operational_intensity().
    "uncore_cas_count_read_interval", "uncore_cas_count_write_interval",
    "bytes_moved_uncore_real", "operational_intensity_uncore_real", "phase_label_uncore_real",
    # F1-CPU-002: límites del intervalo físico que originó los CAS. Se
    # conservan también en la traza fina para que el CSV de entrenamiento
    # pueda reagrupar sin reconstruir ni adivinar los límites temporales.
    "uncore_interval_id", "uncore_t_start_ns", "uncore_t_end_ns", "uncore_delta_t_ns",
    "i_ridge_used", "roofline_calibration_ref", "node_profile_ref", "calibration_ref",
    "binary_checksum", "quality_status",
    # ARC-70: filas GPU (tag=GPU en samples.csv) -- ver build_windows() y el
    # comentario de quality_status="gpu_telemetry" abajo. Vacías en toda fila
    # derivada de CPU/ENERGY.
    "gpu_power_mw", "gpu_util_pct", "gpu_mem_util_pct",
    # ARC-94: reloj SM observado (confirma que un nivel DVFS de GPU se
    # mantuvo durante la corrida), energía acumulada (insumo real para EDP
    # de GPU, antes inexistente) y temperatura (detecta contaminación
    # térmica) -- las tres opcionales, None si el launcher no las reportó
    # (driver/GPU sin soporte, o corrida generada antes de este cambio).
    "gpu_sm_clock_mhz", "gpu_energy_mj", "gpu_temperature_c",
    # ARC-95: delta de energía GPU entre esta ventana y la anterior (mJ),
    # con su propio bit de validez -- gpu_energy_mj por sí solo es un
    # acumulado crudo, insuficiente para EDP de GPU sin este cálculo.
    "gpu_energy_delta_mj", "gpu_energy_valid",
)

# F1-CPU-002: artefacto de entrenamiento, separado de windows.csv.  Una fila
# representa exactamente un intervalo de perf stat -I del IMC; las filas no
# usables se mantienen con training_quality_status para poder auditarlas, pero
# Fase 2 solo acepta las marcadas como ok.
TRAINING_CPU_INTERVALS_FILENAME = "training_cpu_intervals.csv"
TRAINING_CPU_INTERVAL_COLUMNS: tuple[str, ...] = (
    "run_id", "repetition", "kernel_ref", "node_id", "freq_level_id",
    "uncore_interval_id", "uncore_t_start_ns", "uncore_t_end_ns", "uncore_delta_t_ns",
    "cpu_window_count", "training_quality_status", "training_quality_reason",
    "frequency_quality_status", "phase_label_train",
    "delta_instructions", "delta_cycles", "delta_cache_references", "delta_cache_misses",
    "delta_stalled_cycles_mem_any", "delta_running_ns", "delta_enabled_ns",
    "ipc", "mpki", "llc_miss_rate", "stall_mem_ratio", "ips", "running_ratio",
    "freq_khz_observed", "freq_khz_observed_spread",
    # Solo trazabilidad/verdad Roofline: el entrenador no las lee (fuga).
    "flops_measured_interval", "uncore_cas_count_read_interval",
    "uncore_cas_count_write_interval", "bytes_moved_uncore_real",
    "operational_intensity_uncore_real", "i_ridge_used",
)

VALID_QUALITY_STATUSES = frozenset({
    "ok", "first_sample_no_delta", "warmup_excluded", "pmu_degraded",
    "energy_invalid", "no_freq_reading", "intensity_undefined",
    # ARC-70: una muestra NVML cruda (potencia/utilización), no una ventana
    # de CPU -- ninguno de los campos de Roofline/PMU de CPU aplica (nunca
    # tiene delta_t_ns/ipc/etc, ver build_windows()). Sí tiene su propio
    # phase_label_train (ARC-80, calculado con la intensidad medida offline
    # con ncu y el ridge de GPU calibrado por precisión/nivel de frecuencia)
    # cuando esa calibración está disponible -- "gpu_telemetry" describe que
    # la fila es un passthrough NVML, no que carezca de etiqueta.
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

# ARC-97: FP_ARITH_INST_RETIRED double-precision sub-events report
# "computations", not instructions -- each count already represents this
# many double-precision flops (1 for scalar, 2/4/8 for 128B/256B/512B
# packed, per Intel's own counter definition and LIKWID's FLOPS_DP group).
# Weights confirmed empirically on pacca (Ice Lake-SP): the weighted sum
# tracked dgemm_bench's analytical 2*iterations*n^3 within 0.29-0.30%
# (single-threaded and 6-core-pinned) and NPB MG's self-reported total
# within 7.48% (explained by MG's own timer excluding its verification
# phase, not by these weights). See telemetry/src/perf_reader.cpp for the
# raw event encoding this multiplies.
_FP_ARITH_DOUBLES_PER_EVENT: dict[str, int] = {
    "fp_scalar_double": 1,
    "fp_128b_packed_double": 2,
    "fp_256b_packed_double": 4,
    "fp_512b_packed_double": 8,
}

# ARC-116: one CAS_COUNT_READ/WRITE transaction is one DDR column-address-
# strobe burst, the standard Intel iMC convention -- one 64-byte cache line
# per count, same physical unit as context.llc_line_size_bytes on every
# node this project has measured, but this is a memory-controller
# architectural constant (documented by Intel's own uncore performance
# monitoring reference manuals), not derived from node_profile.
_UNCORE_CAS_LINE_BYTES = 64


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
    warmup_seconds: float
    running_ratio_min: float
    rapl_enabled: bool
    calibration_references: Any = None  # calibration.CalibrationReferences | None
    # ARC-80: intensidad operacional (constante, medida offline con `ncu`,
    # ver KernelEntry.operational_intensity_flops_per_byte) y el i_ridge_gpu
    # (calibrado por nivel de frecuencia y precisión, run_gpu_calibration)
    # de ESTE kernel -- None cuando el kernel no es GPU, o cuando no hay
    # calibración GPU disponible en este calibration_dir (campañas previas a
    # ARC-80, o manifest.gpu sin "calibration" declarado). Nunca se reusa
    # i_ridge_flops_per_byte (ese es el de CPU) para una fila de GPU.
    gpu_operational_intensity: float | None = None
    gpu_i_ridge_flops_per_byte: float | None = None
    # ARC-94 (segunda ronda): antes de este campo, las filas GPU heredaban
    # roofline_calibration_ref de _base_row() (el archivo de calibración de
    # CPU) aunque su phase_label_train se calculó con gpu_i_ridge_flops_per_byte
    # -- la columna de trazabilidad apuntaba al archivo equivocado.
    gpu_roofline_calibration_ref: str | None = None
    # ARC-129: nivel de GPU real de esta corrida (producto cartesiano CPU x
    # GPU) -- None para kernels de CPU, o de GPU sin gpu_frequency_levels
    # declarado (acoplado a freq_level_id, como siempre).
    gpu_freq_level_id: str | None = None
    # ARC-174: insumos para validation.classify_frequency_window() por
    # ventana de CPU -- freq_is_native_governor viene EXPLÍCITAMENTE del
    # modo del nivel de frecuencia (manifest.frequency_levels[].mode), no
    # se infiere de freq_khz_applied is None (ese campo también es None
    # cuando la actuación de frecuencia está desactivada por completo,
    # frequency_write_capable=False -- un caso distinto de REF que nunca
    # debe clasificarse como "not_applicable_native"). Defaults preservan
    # "sin clasificar" (fail-closed) para callers que no los pasan.
    freq_tolerance_fraction: float | None = None
    freq_expected_cpu_count: int | None = None
    freq_grace_seconds: float = 0.0
    freq_tail_grace_seconds: float = 0.0
    freq_is_native_governor: bool = False


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


def _observed_freq_spread(raw: str | None) -> int | None:
    """ARC-142: parses telemetry_kernel_launcher.cpp's ';'-separated
    scaling_cur_freq_khz_all column (one reading per delegated CPU, 0 for a
    CPU whose individual read failed that tick, same "not sampled"
    convention as every other optional column) and returns max-min across
    the CPUs that DID report a nonzero reading. None (never 0) when fewer
    than 2 CPUs reported -- a spread needs at least 2 points to mean
    anything, and 0 would be indistinguishable from "confirmed identical."
    """
    if not raw:
        return None
    values = [int(part) for part in raw.split(";") if part.isdigit() and int(part) != 0]
    if len(values) < 2:
        return None
    return max(values) - min(values)


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
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    cpu_rows = [r for r in rows if r.get("tag") == "CPU" and _to_int(r.get("repetition")) == repetition]
    energy_rows = [r for r in rows if r.get("tag") == "ENERGY" and _to_int(r.get("repetition")) == repetition]
    uncore_rows = [r for r in rows if r.get("tag") == "UNCORE" and _to_int(r.get("repetition")) == repetition]
    cpu_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    energy_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    uncore_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    return cpu_rows, energy_rows, uncore_rows


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


def _apply_uncore_intervals(
    windows: list[dict[str, Any]],
    cpu_rows: Sequence[dict[str, str]],
    uncore_rows: Sequence[dict[str, str]],
    run_start_ns: int,
    i_ridge_flops_per_byte: float,
) -> None:
    """ARC-119: broadcast each `perf stat -I` interval's already-computed
    delta (see UncoreSnapshot in metrics.hpp -- these are NOT cumulative,
    never differenced against each other) onto every CPU window it
    overlaps, mutating `windows` in place.

    Each UNCORE row stands alone as the traffic for
    (previous UNCORE row's timestamp, this row's timestamp], or
    (run_start_ns, this row's timestamp] for the first one -- unlike
    CPU/ENERGY rows there is no "first sample has no predecessor" case.
    Because perf's own interval (~10ms floor) is coarser than a CPU window
    (~1ms), a single interval's bytes cannot be assigned to just one CPU
    window: doing so would divide a wide-interval byte count by one
    window's narrow flops_measured_window, systematically and silently
    biasing operational_intensity_uncore_real toward memory_bound. Instead,
    flops_measured_window is summed across every CPU window whose t_end
    falls inside the interval, and the resulting intensity/label is
    broadcast unchanged to all of them -- correct at the interval's own
    granularity, never claimed at the finer per-window one.
    """
    if not uncore_rows:
        return

    interval_start_ns = run_start_ns
    # windows[0] is the first_row placeholder (no t_end); windows[i]
    # corresponds to cpu_rows[i] for i in 1..len(cpu_rows)-1.
    for interval_id, uncore_row in enumerate(uncore_rows, start=1):
        interval_end_ns = int(uncore_row["timestamp_ns"])
        cas_read = _to_int(uncore_row.get("uncore_cas_count_read_interval"))
        cas_write = _to_int(uncore_row.get("uncore_cas_count_write_interval"))
        bytes_this_interval = (
            (cas_read + cas_write) * _UNCORE_CAS_LINE_BYTES
            if cas_read is not None and cas_read >= 0 and cas_write is not None and cas_write >= 0
            else None
        )

        covered_indices = [
            i for i in range(1, len(cpu_rows))
            if interval_start_ns < int(cpu_rows[i]["timestamp_ns"]) <= interval_end_ns
        ]

        operational_intensity_uncore: float | None = None
        phase_label_uncore: str | None = None
        if covered_indices and bytes_this_interval is not None and bytes_this_interval > 0:
            flops_values = [windows[i].get("flops_measured_window") for i in covered_indices]
            if all(value is not None for value in flops_values):
                flops_in_interval = sum(flops_values)
                operational_intensity_uncore = flops_in_interval / bytes_this_interval
                phase_label_uncore = (
                    "memory_bound" if operational_intensity_uncore < i_ridge_flops_per_byte else "compute_bound"
                )

        for i in covered_indices:
            windows[i]["uncore_interval_id"] = interval_id
            windows[i]["uncore_t_start_ns"] = interval_start_ns
            windows[i]["uncore_t_end_ns"] = interval_end_ns
            windows[i]["uncore_delta_t_ns"] = interval_end_ns - interval_start_ns
            windows[i]["uncore_cas_count_read_interval"] = cas_read
            windows[i]["uncore_cas_count_write_interval"] = cas_write
            windows[i]["bytes_moved_uncore_real"] = bytes_this_interval
            windows[i]["operational_intensity_uncore_real"] = operational_intensity_uncore
            windows[i]["phase_label_uncore_real"] = phase_label_uncore

        interval_start_ns = interval_end_ns


# Statuses that must never be overridden by the intensity finalization
# below -- each already outranks "intensity_undefined" in _QUALITY_PRIORITY,
# so if one of these is already set, whatever caused it is a more
# fundamental problem than the classification being undefined.
_STATUSES_ABOVE_INTENSITY = frozenset({"first_sample_no_delta", "pmu_degraded", "warmup_excluded"})


def _finalize_operational_intensity(
    windows: list[dict[str, Any]],
    cpu_rows: Sequence[dict[str, str]],
) -> None:
    """ARC-123: the single place that decides operational_intensity/
    phase_label_train, exclusively from real uncore_imc data -- never from
    bytes_moved_window (the cache-misses proxy), which structurally cannot
    see hardware prefetch traffic (sec:intensidad) and was found to
    silently mix inconsistent measurement quality into the same label
    column when used as a fallback (ARC-122 superseded).

    Every window entering this function already has operational_intensity
    set to a NaN placeholder and phase_label_train to None (build_windows()
    no longer computes them from the proxy at all). Where
    _apply_uncore_intervals() found real coverage, that value becomes
    final. Where it did not (run start before the first interval closes,
    gaps if uncore degrades mid-run, or this run never opened uncore),
    the window stays undefined -- quality_status becomes
    "intensity_undefined" unless a higher-priority problem
    (_STATUSES_ABOVE_INTENSITY) already explains the row.

    Mutates `windows` in place.
    """
    for i in range(1, len(cpu_rows)):
        row = windows[i]
        if row.get("operational_intensity_uncore_real") is not None:
            row["operational_intensity"] = row["operational_intensity_uncore_real"]
            row["phase_label_train"] = row["phase_label_uncore_real"]
        elif row["quality_status"] not in _STATUSES_ABOVE_INTENSITY:
            row["quality_status"] = "intensity_undefined"


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
    cpu_rows, energy_rows, uncore_rows = _split_by_repetition_and_tag(rows)
    if not cpu_rows:
        return []

    # STALLS_MEM_ANY is a per-node capability, not a per-window one. The
    # launcher writes an empty column for every row if this model-gated raw
    # event cannot be opened; that is "not measured here", not a measured
    # zero nor a per-window PMU failure. A missing value on an otherwise
    # supported node remains a real anomaly and triggers pmu_degraded below.
    stall_mem_supported = any(
        r.get("stalled_cycles_mem_any") not in (None, "") for r in cpu_rows
    )
    # Same per-node-capability rule as STALLS_MEM_ANY above --
    # L2_LINES_IN_ALL is a raw event only opened on Ice Lake-SP, empty (not
    # "0") for every row when unsupported.
    l2_lines_in_all_supported = any(
        r.get("l2_lines_in_all") not in (None, "") for r in cpu_rows
    )
    # ARC-97: same per-node-capability rule -- the 4 FP_ARITH_INST_RETIRED
    # sub-events open/close together as a unit (see PerfReader::has_fp_arith
    # in telemetry/include/telemetry/perf_reader.hpp), so checking one is
    # enough to know whether all 4 columns are populated for this run.
    fp_arith_supported = any(
        r.get("fp_scalar_double") not in (None, "") for r in cpu_rows
    )

    run_total_instructions = _to_int(cpu_rows[-1].get("instructions"))
    run_start_ns = int(cpu_rows[0]["timestamp_ns"])
    run_end_ns = int(cpu_rows[-1]["timestamp_ns"])
    warmup_end_ns = run_start_ns + int(context.warmup_seconds * 1_000_000_000)
    freq_grace_ns = context.freq_grace_seconds * 1_000_000_000
    freq_tail_grace_ns = context.freq_tail_grace_seconds * 1_000_000_000
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

        # ARC-135: real per-window reading (sampled by the C++ collector on
        # the SAME tick as the PMU counters, telemetry/src/cpu_freq_reader.cpp)
        # overrides the single run-wide broadcast from WindowContext when
        # present -- frequency is instantaneous, not cumulative, so this uses
        # cur directly, never a delta. Falls back to the old context-level
        # value only for samples.csv predating this column (backward compat,
        # never fabricated).
        scaling_cur_freq_khz = _to_int(cur.get("scaling_cur_freq_khz"))
        if scaling_cur_freq_khz is not None:
            row["freq_khz_observed"] = scaling_cur_freq_khz

        # ARC-142: scaling_cur_freq_khz_all carries the SAME reading for
        # every delegated CPU, not just CPU0/freq_khz_observed above --
        # pacca's cpufreq domain is per-core (not per-socket like felix's),
        # so the other delegated CPUs can genuinely run at a different clock
        # under Turbo/HWP without this. freq_khz_observed_spread is
        # max-min across whichever CPUs actually reported a nonzero reading
        # this tick -- None (never 0) when fewer than 2 CPUs reported, same
        # "not enough data" convention as everywhere else in this file. A
        # positive spread means the delegated CPUs did NOT all run at the
        # same clock during this window.
        row["freq_khz_observed_spread"] = _observed_freq_spread(cur.get("scaling_cur_freq_khz_all"))

        # ARC-174: clasificación de frecuencia por ventana -- usa el mismo
        # tick (cur) que ya alimenta freq_khz_observed_spread arriba, y el
        # mismo criterio grace/tail_grace de validate_cpu_frequency_trace()
        # (ARC-166/169), pero evaluado por ventana en vez de agregado sobre
        # toda la corrida. expected_khz usa freq_khz_applied (nunca
        # freq_khz_requested) -- cubre redondeos reales del actuador que un
        # objetivo puramente nominal no vería.
        within_freq_grace = (
            (freq_grace_ns > 0 and (t_end_ns - run_start_ns) < freq_grace_ns)
            or (freq_tail_grace_ns > 0 and (run_end_ns - t_end_ns) < freq_tail_grace_ns)
        )
        freq_classification = validation_module.classify_frequency_window(
            cur.get("scaling_cur_freq_khz_all"),
            is_native_governor=context.freq_is_native_governor,
            expected_khz=context.freq_khz_applied,
            tolerance_fraction=context.freq_tolerance_fraction,
            within_grace=within_freq_grace,
        )
        row["frequency_quality_status"] = freq_classification.status
        row["frequency_outlier_cpu_count"] = freq_classification.outlier_cpu_count
        row["frequency_min_khz"] = freq_classification.min_khz
        row["frequency_max_khz"] = freq_classification.max_khz
        row["frequency_max_relative_error"] = freq_classification.max_relative_error

        delta_instructions = _delta(_to_int(cur.get("instructions")), _to_int(prev.get("instructions")))
        delta_cycles = _delta(_to_int(cur.get("cycles")), _to_int(prev.get("cycles")))
        delta_cache_references = _delta(_to_int(cur.get("cache_references")), _to_int(prev.get("cache_references")))
        delta_cache_misses = _delta(_to_int(cur.get("cache_misses")), _to_int(prev.get("cache_misses")))
        delta_stalled_cycles_mem_any = (
            _delta(_to_int(cur.get("stalled_cycles_mem_any")), _to_int(prev.get("stalled_cycles_mem_any")))
            if stall_mem_supported else None
        )
        delta_l2_lines_in_all = (
            _delta(_to_int(cur.get("l2_lines_in_all")), _to_int(prev.get("l2_lines_in_all")))
            if l2_lines_in_all_supported else None
        )
        # ARC-97: all 4 deltas are None together (fp_arith_supported gates
        # all of them at once, same as the single l2 column above).
        delta_fp_scalar_double = (
            _delta(_to_int(cur.get("fp_scalar_double")), _to_int(prev.get("fp_scalar_double")))
            if fp_arith_supported else None
        )
        delta_fp_128b_packed_double = (
            _delta(_to_int(cur.get("fp_128b_packed_double")), _to_int(prev.get("fp_128b_packed_double")))
            if fp_arith_supported else None
        )
        delta_fp_256b_packed_double = (
            _delta(_to_int(cur.get("fp_256b_packed_double")), _to_int(prev.get("fp_256b_packed_double")))
            if fp_arith_supported else None
        )
        delta_fp_512b_packed_double = (
            _delta(_to_int(cur.get("fp_512b_packed_double")), _to_int(prev.get("fp_512b_packed_double")))
            if fp_arith_supported else None
        )
        delta_running_ns = _delta(_to_int(cur.get("time_running_ns")), _to_int(prev.get("time_running_ns")))
        delta_enabled_ns = _delta(_to_int(cur.get("time_enabled_ns")), _to_int(prev.get("time_enabled_ns")))

        # POST-02: a negative delta means the counter wrapped or was reset
        # mid-window (no wrap-correction is attempted for perf counters,
        # unlike RAPL which the launcher already corrects). A missing field
        # is treated the same way: the window is kept but flagged
        # pmu_degraded, never silently fixed up or imputed. STALLS_MEM_ANY
        # only participates in this gate on nodes that support it (ARC-50) --
        # otherwise every window on an unsupported node would be flagged
        # degraded for a counter that was never going to exist.
        core_deltas = [delta_instructions, delta_cycles, delta_cache_references, delta_cache_misses]
        if stall_mem_supported:
            core_deltas.append(delta_stalled_cycles_mem_any)
        if l2_lines_in_all_supported:
            core_deltas.append(delta_l2_lines_in_all)
        if fp_arith_supported:
            # ARC-97: same gate as l2_lines_in_all above -- a negative or
            # missing FP delta on a node that DOES support the counter is a
            # real anomaly (e.g. counter overflow/reset), not silently
            # dropped from operational_intensity via flops_measured_window's
            # own None-check alone.
            core_deltas.extend([
                delta_fp_scalar_double,
                delta_fp_128b_packed_double,
                delta_fp_256b_packed_double,
                delta_fp_512b_packed_double,
            ])
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
        row["delta_stalled_cycles_mem_any"] = delta_stalled_cycles_mem_any
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
            # Fracción de ciclos de ejecución bloqueados mientras existe al
            # menos una carga pendiente en el subsistema de memoria.
            row["stall_mem_ratio"] = (
                delta_stalled_cycles_mem_any / delta_cycles
                if delta_cycles and delta_stalled_cycles_mem_any is not None
                else None
            )
        else:
            row["ips"] = row["ipc"] = row["mpki"] = row["llc_miss_rate"] = None
            row["stall_mem_ratio"] = None

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

        # POST-08/POST-09/POST-10: FLOPs measured directly by hardware
        # (ARC-97/98/99, single-node scope -- see docs/libro/main.tex Marco
        # Conceptual). All-or-nothing across the 4 SIMD widths: a partial
        # delta set (one width missing) is treated the same as none, since
        # silently omitting a width would under-count without any signal
        # that it happened. No instruction-prorated fallback: this project
        # never claimed portability beyond the validated platform, and a
        # silent estimate in place of a missing measurement would hide
        # exactly the kind of gap this instrument is designed to surface
        # (same principle as every other counter here -- see quality_status
        # taxonomy below). bytes_moved_window uses the node_profile's real
        # LLC line size.
        flops_measured_window = None
        if (
            valid_counters
            and fp_arith_supported
            and delta_fp_scalar_double is not None
            and delta_fp_128b_packed_double is not None
            and delta_fp_256b_packed_double is not None
            and delta_fp_512b_packed_double is not None
        ):
            flops_measured_window = (
                _FP_ARITH_DOUBLES_PER_EVENT["fp_scalar_double"] * delta_fp_scalar_double
                + _FP_ARITH_DOUBLES_PER_EVENT["fp_128b_packed_double"] * delta_fp_128b_packed_double
                + _FP_ARITH_DOUBLES_PER_EVENT["fp_256b_packed_double"] * delta_fp_256b_packed_double
                + _FP_ARITH_DOUBLES_PER_EVENT["fp_512b_packed_double"] * delta_fp_512b_packed_double
            )

        bytes_moved_window = (
            delta_cache_misses * context.llc_line_size_bytes
            if valid_counters and delta_cache_misses is not None
            else None
        )
        row["flops_measured_window"] = flops_measured_window
        row["bytes_moved_window"] = bytes_moved_window

        # ARC-63: independent cross-check for bytes_moved_window's bias
        # (F3.4/ARC-33, ARC-60) -- same line-size multiplier convention, so
        # it is directly comparable to bytes_moved_window per window. Still
        # a core-level (L2) proxy, not real DRAM bytes; never used in
        # operational_intensity/phase_label_train (uncore_imc is, when
        # available -- see ARC-119/ARC-123), purely a reported cross-check
        # column.
        row["bytes_moved_l2_proxy"] = (
            delta_l2_lines_in_all * context.llc_line_size_bytes
            if l2_lines_in_all_supported and delta_l2_lines_in_all is not None
            else None
        )

        # ARC-123: operational_intensity/phase_label_train are decided
        # EXCLUSIVELY from real uncore_imc data, in _finalize_operational_intensity()
        # after this loop -- bytes_moved_window (the cache-misses proxy) is
        # reported above for comparison only and never classifies a window.
        # The proxy structurally misses hardware prefetch traffic
        # (sec:intensidad); mixing it into the same label column as the
        # real measurement, even only for the windows uncore didn't cover,
        # would make windows.csv's classification quality silently
        # inconsistent row to row. A window with no real uncore coverage is
        # therefore left undefined (quality_status="intensity_undefined"),
        # the same outcome already used for a missing FLOPs/bytes reading
        # of any other kind -- kept in windows.csv, never dropped, but
        # never guessed either.
        row["operational_intensity"] = float("nan")
        row["phase_label_train"] = None

        no_freq_reading = context.freq_khz_observed is None

        row["quality_status"] = _resolve_quality_status({
            "pmu_degraded": pmu_degraded,
            "warmup_excluded": warmup_excluded,
            "energy_invalid": energy_invalid,
            "no_freq_reading": no_freq_reading,
        })
        windows.append(row)

    # ARC-119: broadcasts uncore_imc's own real bytes onto the CPU windows
    # each perf interval overlaps -- see _apply_uncore_intervals()
    # docstring. Mutates `windows` in place; a no-op when this run never
    # opened uncore (uncore_rows empty).
    _apply_uncore_intervals(windows, cpu_rows, uncore_rows, run_start_ns, context.i_ridge_flops_per_byte)
    # ARC-123: operational_intensity/phase_label_train get their final
    # value here -- see _finalize_operational_intensity() docstring.
    _finalize_operational_intensity(windows, cpu_rows)

    # ARC-70: filas GPU (tag=GPU, muestras NVML crudas) -- deliberadamente
    # NO se ventanean contra los límites de las ventanas de CPU de arriba.
    # A diferencia de CPU, no hay una intensidad operacional que calcular
    # aquí EN VIVO (ver Diseno_Politica_DVFS_CPU_GPU.md sección 3): NVML
    # solo expone potencia/utilización, nunca FLOPs ni bytes -- cada muestra
    # es un passthrough con el contexto de la corrida ya adjunto. Son las
    # *features* de entrenamiento del futuro modelo de GPU (Fase 2).
    #
    # ARC-80: la ETIQUETA (phase_label_train) sí se calcula aquí, en Fase 1
    # -- es la corrección al error de diseño de ARC-72/ARC-79, que decía
    # "esto espera al modelo de GPU" confundiendo la etiqueta de verdad
    # (Fase 1, análoga a bytes_moved_window/flops_measured_window de CPU)
    # con el modelo que la va a consumir (Fase 2). La intensidad operacional
    # de este kernel (context.gpu_operational_intensity, medida offline con
    # `ncu` una sola vez, ARC-80) es constante en todas las filas de esta
    # corrida; el ridge (context.gpu_i_ridge_flops_per_byte) es el calibrado
    # para la precisión de este kernel Y el freq_level_id de esta corrida
    # (run_gpu_calibration, ARC-80) -- nunca uno fijo para toda la campaña
    # (mismo principio que ARC-78 ya aplica al ridge de CPU). Si cualquiera
    # de los dos falta (kernel de calibración, o campaña sin
    # manifest.gpu["calibration"] declarado), la fila queda sin etiqueta en
    # vez de adivinar una.
    gpu_rows = [
        r for r in rows
        if r.get("tag") == "GPU" and _to_int(r.get("repetition")) == _LAUNCHER_INTERNAL_REPETITION
    ]
    gpu_rows.sort(key=lambda r: int(r["timestamp_ns"]))
    # ARC-95: gpu_energy_mj es un contador acumulado (nvmlDeviceGetTotalEnergyConsumption,
    # mJ desde que cargó el driver) -- exactamente igual que pkg_uj de RAPL,
    # necesita un delta por ventana para servir de insumo a EDP de GPU; antes
    # de esto solo se copiaba el acumulado crudo, insuficiente para el
    # análisis energético posterior. Mismo criterio de invalidez que RAPL
    # (ARC-56): primera muestra sin predecesor, o el contador retrocede
    # (wraparound o reinicio del driver) -> delta inválido, nunca negativo.
    previous_energy_mj: int | None = None
    for gpu_index, gpu_row in enumerate(gpu_rows):
        row = _base_row(context, window_index=gpu_index)
        row["t_start_ns"] = None
        row["t_end_ns"] = int(gpu_row["timestamp_ns"])
        row["delta_t_ns"] = None
        # ARC-94 (segunda ronda): _base_row() deja roofline_calibration_ref
        # apuntando al archivo de calibración de CPU para TODA fila -- una
        # fila GPU usa gpu_i_ridge_flops_per_byte (calibrado por separado,
        # run_gpu_calibration) para su phase_label_train, así que la
        # trazabilidad debe apuntar a ESE archivo, no al de CPU.
        if context.gpu_roofline_calibration_ref is not None:
            row["roofline_calibration_ref"] = context.gpu_roofline_calibration_ref
        row["gpu_power_mw"] = _to_int(gpu_row.get("gpu_power_mw"))
        row["gpu_util_pct"] = _to_int(gpu_row.get("gpu_util_pct"))
        row["gpu_mem_util_pct"] = _to_int(gpu_row.get("gpu_mem_util_pct"))
        # F1-GPU-001: el harness ya exporta celda vacía cuando NVML no
        # soporta reloj SM / energía / temperatura (nvml_reader.cpp mira el
        # código de retorno). Este `or None` es la red de seguridad para
        # `samples.csv` grabados con el contrato viejo ("0 = no disponible"):
        # un reloj SM de 0 MHz o una temperatura de 0 °C bajo carga no son
        # lecturas físicas reales, así que se tratan como ausentes en vez de
        # como un valor plausible.
        row["gpu_sm_clock_mhz"] = _to_int(gpu_row.get("gpu_sm_clock_mhz")) or None
        current_energy_mj = _to_int(gpu_row.get("gpu_energy_mj"))
        # Un contador acumulado (mJ desde que cargó el driver) nunca es 0 si
        # NVML lo soporta -- un 0 aquí es "no disponible", no energía cero.
        # Sin este guardia, en un driver sin soporte previous==current==0 y
        # desde la 2ª ventana se fabricaba gpu_energy_valid=True con delta 0.
        if current_energy_mj is not None and current_energy_mj <= 0:
            current_energy_mj = None
        row["gpu_energy_mj"] = current_energy_mj
        gpu_energy_delta_mj: int | None = None
        gpu_energy_valid = False
        if current_energy_mj is not None:
            if previous_energy_mj is not None and current_energy_mj >= previous_energy_mj:
                gpu_energy_delta_mj = current_energy_mj - previous_energy_mj
                gpu_energy_valid = True
            previous_energy_mj = current_energy_mj
        row["gpu_energy_delta_mj"] = gpu_energy_delta_mj
        row["gpu_energy_valid"] = gpu_energy_valid
        row["gpu_temperature_c"] = _to_int(gpu_row.get("gpu_temperature_c")) or None
        row["operational_intensity"] = context.gpu_operational_intensity
        row["i_ridge_used"] = context.gpu_i_ridge_flops_per_byte
        if context.gpu_operational_intensity is not None and context.gpu_i_ridge_flops_per_byte is not None:
            row["phase_label_train"] = (
                "memory_bound"
                if context.gpu_operational_intensity < context.gpu_i_ridge_flops_per_byte
                else "compute_bound"
            )
        # ARC-94: warmup_seconds declarado en el catálogo nunca se
        # comparaba contra el timestamp de la muestra GPU -- cada fila
        # quedaba "gpu_telemetry" sin importar si caía antes o después del
        # calentamiento, a diferencia de las ventanas CPU (que sí excluyen
        # warmup arriba, mismo run_start_ns/warmup_end_ns). Reusa la misma
        # referencia temporal de origen del run (cpu_rows[0]) para que
        # ambos ejes midan el calentamiento desde el mismo instante cero.
        row["quality_status"] = (
            "warmup_excluded" if int(gpu_row["timestamp_ns"]) < warmup_end_ns else "gpu_telemetry"
        )
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
        "gpu_freq_level_id": context.gpu_freq_level_id,
        "freq_khz_requested": context.freq_khz_requested,
        "freq_khz_applied": context.freq_khz_applied,
        "freq_khz_observed": context.freq_khz_observed,
        "freq_khz_observed_spread": None,
        "frequency_quality_status": None,
        "frequency_outlier_cpu_count": None,
        "frequency_min_khz": None,
        "frequency_max_khz": None,
        "frequency_max_relative_error": None,
        "window_index": window_index,
        "delta_instructions": None,
        "delta_cycles": None,
        "delta_cache_references": None,
        "delta_cache_misses": None,
        "delta_stalled_cycles_mem_any": None,
        "stall_mem_ratio": None,
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
        "flops_measured_window": None,
        "bytes_moved_window": None,
        "operational_intensity": None,
        "uncore_cas_count_read_interval": None,
        "uncore_cas_count_write_interval": None,
        "bytes_moved_uncore_real": None,
        "operational_intensity_uncore_real": None,
        "phase_label_uncore_real": None,
        "uncore_interval_id": None,
        "uncore_t_start_ns": None,
        "uncore_t_end_ns": None,
        "uncore_delta_t_ns": None,
        "i_ridge_used": context.i_ridge_flops_per_byte,
        "roofline_calibration_ref": context.roofline_calibration_ref,
        "node_profile_ref": context.node_profile_ref,
        "calibration_ref": context.calibration_ref,
        "binary_checksum": context.binary_checksum,
        "quality_status": "ok",
        "gpu_power_mw": None,
        "gpu_util_pct": None,
        "gpu_mem_util_pct": None,
        "gpu_sm_clock_mhz": None,
        "gpu_energy_mj": None,
        "gpu_energy_delta_mj": None,
        "gpu_energy_valid": False,
        "gpu_temperature_c": None,
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


def _sum_interval_values(rows: Sequence[dict[str, Any]], column: str) -> int | None:
    """Suma deltas únicamente si cada ventana aportó una lectura válida."""
    values = [row.get(column) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def build_training_cpu_intervals(windows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """F1-CPU-002: convierte la traza CPU fina en ejemplos por intervalo IMC.

    Las tasas se vuelven a calcular desde deltas, nunca promediando ratios de
    ticks de distinta duración. La etiqueta y los bytes ya fueron calculados
    en ``_apply_uncore_intervals`` sobre este mismo conjunto de ventanas.
    Un intervalo defectuoso se emite para auditoría con ``training_quality_*``
    pero no puede entrar al entrenador.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in windows:
        interval_id = row.get("uncore_interval_id")
        if interval_id is not None:
            grouped.setdefault(int(interval_id), []).append(row)

    output: list[dict[str, Any]] = []
    required_deltas = (
        "delta_instructions", "delta_cycles", "delta_cache_references",
        "delta_cache_misses", "delta_stalled_cycles_mem_any",
        "delta_running_ns", "delta_enabled_ns",
    )
    for interval_id, rows in sorted(grouped.items()):
        first = rows[0]
        result = {column: None for column in TRAINING_CPU_INTERVAL_COLUMNS}
        for column in ("run_id", "repetition", "kernel_ref", "node_id", "freq_level_id",
                       "uncore_t_start_ns", "uncore_t_end_ns", "uncore_delta_t_ns"):
            result[column] = first.get(column)
        result["uncore_interval_id"] = interval_id
        result["cpu_window_count"] = len(rows)

        # A run should not mix these values; detecting it avoids silently
        # joining rows from different contexts if a malformed trace is read.
        consistent_context = all(
            all(row.get(column) == first.get(column) for row in rows)
            for column in ("run_id", "repetition", "kernel_ref", "freq_level_id",
                           "uncore_t_start_ns", "uncore_t_end_ns", "uncore_delta_t_ns")
        )
        reason: str | None = None
        if not consistent_context:
            reason = "inconsistent_interval_context"
        elif any(row.get("quality_status") != "ok" for row in rows):
            reason = "source_window_not_ok"
        else:
            frequency_statuses = {row.get("frequency_quality_status") for row in rows}
            if len(frequency_statuses) != 1 or not frequency_statuses <= {"valid", "not_applicable_native"}:
                reason = "frequency_not_usable"
            else:
                result["frequency_quality_status"] = frequency_statuses.pop()

        labels = {row.get("phase_label_train") for row in rows}
        if reason is None:
            if len(labels) != 1 or None in labels or "" in labels:
                reason = "label_missing_or_inconsistent"
            else:
                result["phase_label_train"] = labels.pop()

        sums = {column: _sum_interval_values(rows, column) for column in required_deltas}
        result.update(sums)
        if reason is None and any(value is None or value < 0 for value in sums.values()):
            reason = "counter_delta_missing_or_invalid"

        cas_read = first.get("uncore_cas_count_read_interval")
        cas_write = first.get("uncore_cas_count_write_interval")
        bytes_real = first.get("bytes_moved_uncore_real")
        oi = first.get("operational_intensity_uncore_real")
        ridge = first.get("i_ridge_used")
        if all(all(row.get(column) == first.get(column) for row in rows)
               for column in ("uncore_cas_count_read_interval", "uncore_cas_count_write_interval",
                              "bytes_moved_uncore_real", "operational_intensity_uncore_real", "i_ridge_used")):
            result.update({
                "uncore_cas_count_read_interval": cas_read,
                "uncore_cas_count_write_interval": cas_write,
                "bytes_moved_uncore_real": bytes_real,
                "operational_intensity_uncore_real": oi,
                "i_ridge_used": ridge,
            })
        elif reason is None:
            reason = "inconsistent_uncore_measurement"

        flops = _sum_interval_values(rows, "flops_measured_window")
        result["flops_measured_interval"] = flops
        duration_ns = result["uncore_delta_t_ns"]
        if reason is None and (
            not isinstance(duration_ns, int) or duration_ns <= 0
            or cas_read is None or cas_write is None or bytes_real is None or bytes_real <= 0
            or flops is None or oi is None or ridge is None
        ):
            reason = "uncore_measurement_missing_or_invalid"
        if reason is None:
            # Defensa contra una unión accidental de límites distintos: la
            # verdad Roofline difundida a las ventanas debe coincidir con el
            # FLOP agregado y los bytes del mismo intervalo que formarán la
            # fila de entrenamiento.
            recomputed_oi = flops / bytes_real
            expected_label = "memory_bound" if recomputed_oi < ridge else "compute_bound"
            if not math.isclose(float(oi), recomputed_oi, rel_tol=1e-12, abs_tol=0.0) or result[
                "phase_label_train"
            ] != expected_label:
                reason = "roofline_interval_mismatch"
            else:
                result["operational_intensity_uncore_real"] = recomputed_oi

        instructions, cycles = sums["delta_instructions"], sums["delta_cycles"]
        references, misses = sums["delta_cache_references"], sums["delta_cache_misses"]
        stalled = sums["delta_stalled_cycles_mem_any"]
        running, enabled = sums["delta_running_ns"], sums["delta_enabled_ns"]
        if reason is None and (not instructions or not cycles or not references or not enabled):
            reason = "zero_rate_denominator"

        frequencies = [row.get("freq_khz_observed") for row in rows]
        if reason is None and any(value is None or int(value) <= 0 for value in frequencies):
            reason = "frequency_missing"

        if reason is None:
            result["ipc"] = instructions / cycles
            result["mpki"] = misses / instructions * 1000.0
            result["llc_miss_rate"] = misses / references
            result["stall_mem_ratio"] = stalled / cycles
            result["ips"] = instructions / (duration_ns / 1_000_000_000)
            result["running_ratio"] = running / enabled
            # Mediana (F1-CPU-002): robusta frente a un tick DVFS aislado;
            # queda registrada como criterio fijo en la metadata del modelo.
            result["freq_khz_observed"] = int(median(int(value) for value in frequencies))
            spreads = [row.get("freq_khz_observed_spread") for row in rows]
            valid_spreads = [int(value) for value in spreads if value is not None]
            result["freq_khz_observed_spread"] = max(valid_spreads) if valid_spreads else None
            result["training_quality_status"] = "ok"
            result["training_quality_reason"] = ""
        else:
            result["training_quality_status"] = "rejected"
            result["training_quality_reason"] = reason
        output.append(result)
    return output


def write_training_cpu_intervals_csv(rows: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    """Escribe el dataset F1-CPU-002 sin alterar el CSV crudo/auditable."""
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(TRAINING_CPU_INTERVAL_COLUMNS)
        for row in rows:
            writer.writerow([_format_cell(row.get(column)) for column in TRAINING_CPU_INTERVAL_COLUMNS])
    return path


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
    gpu_freq_level_id: str | None = None,
    freq_tolerance_fraction: float | None = None,
    freq_expected_cpu_count: int | None = None,
    freq_grace_seconds: float = 0.0,
    freq_tail_grace_seconds: float = 0.0,
    freq_is_native_governor: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    """Orchestrates one run's samples.csv -> windows.csv + training_cpu_intervals.csv.

    ARC-174: ``freq_tolerance_fraction``/``freq_expected_cpu_count``/
    ``freq_grace_seconds``/``freq_tail_grace_seconds``/
    ``freq_is_native_governor`` alimentan la clasificación de frecuencia
    POR VENTANA (``validation.classify_frequency_window()``, ver
    ``build_windows()``) -- todos con default que preserva "sin clasificar"
    para callers que no los pasan (p.ej. kernels de GPU, donde esto nunca
    aplica). Para un kernel de CPU, escribe además
    ``frequency_quality_summary.json`` junto a ``windows.csv`` (cobertura y
    secuencias anómalas, ver ``validation.summarize_frequency_quality()``).

    ``output_dir`` (ARC-174): cuando se pasa, ``windows.csv``/
    ``frequency_quality_summary.json`` se escriben AHÍ en vez de en
    ``run_dir`` -- ``samples.csv`` se sigue leyendo de ``run_dir`` (los
    crudos originales), pero la salida derivada nunca sobrescribe el
    ``windows.csv``/``verdict.json`` de la corrida original. Pensado para
    el reprocesamiento offline de corridas ya ejecutadas (ver
    ``orchestrator/schemas/tools/reprocess_frequency_quality_v2.py``); None (el
    default) preserva el comportamiento anterior byte a byte.

    POST-15: calibration.load_calibration() refuses (raises) a calibration
    whose D03 plausibility check failed, so an unverified I_ridge can never
    reach this far. POST-10: the LLC line size always comes from
    node_profile.json, never a hardcoded constant.

    ARC-78: loads the i_ridge calibrated at THIS run's own freq_level_id,
    never a single campaign-wide value -- P_pico scales with core clock but
    BW_pico does not, so a window from a reduced-frequency level must never
    be classified against the reference (native_governor) level's ridge.

    ARC-80: for a GPU dataset kernel (device="gpu" with
    gpu_precision/operational_intensity_flops_per_byte declared), also loads
    the GPU ridge for this run's precision/freq_level_id and derives
    phase_label_train for its passthrough rows -- never reuses the CPU
    roofline loaded above for that. Missing GPU calibration for this
    calibration_dir (pre-ARC-80 campaigns, or manifest.gpu without
    "calibration" declared) degrades to gpu_i_ridge=None, never a hard
    failure -- the CPU dataset's classification must not depend on whether
    a GPU calibration happens to exist.

    ARC-129: `gpu_freq_level_id`, when given, is the id the GPU ridge is
    looked up under -- the GPU's own clock, independent of `freq_level_id`
    (CPU axis) for a cartesian CPU x GPU combination. None (the default)
    falls back to `freq_level_id`, exactly as before this parameter existed
    -- the GPU ridge was always calibrated per the (shared) freq_level_id.
    """
    roofline = calibration_module.load_calibration(calibration_dir, freq_level_id)
    profile = node_profile_module.load_node_profile(calibration_dir)

    effective_gpu_freq_level_id = gpu_freq_level_id if gpu_freq_level_id is not None else freq_level_id
    gpu_operational_intensity = getattr(kernel_entry, "operational_intensity_flops_per_byte", None)
    gpu_precision = getattr(kernel_entry, "gpu_precision", None)
    gpu_i_ridge = None
    gpu_roofline_calibration_ref = None
    if getattr(kernel_entry, "device", "cpu") == "gpu" and gpu_precision:
        gpu_roofline_calibration_ref = str(
            Path(calibration_dir)
            / calibration_module.calibration_filename(effective_gpu_freq_level_id, gpu_precision=gpu_precision)
        )
        try:
            gpu_i_ridge = calibration_module.load_calibration(
                calibration_dir, effective_gpu_freq_level_id, gpu_precision=gpu_precision
            ).i_ridge_flops_per_byte
        except (FileNotFoundError, calibration_module.CalibrationError):
            gpu_i_ridge = None

    run_dir = Path(run_dir)

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
        roofline_calibration_ref=str(
            Path(calibration_dir) / calibration_module.calibration_filename(freq_level_id)
        ),
        node_profile_ref=str(Path(calibration_dir) / "node_profile.json"),
        calibration_ref=str(Path(calibration_dir) / "calibration_references.json"),
        i_ridge_flops_per_byte=roofline.i_ridge_flops_per_byte,
        llc_line_size_bytes=profile.cache_line_size_bytes,
        warmup_seconds=warmup_seconds,
        running_ratio_min=running_ratio_min,
        rapl_enabled=rapl_enabled,
        calibration_references=calibration_references,
        gpu_operational_intensity=gpu_operational_intensity,
        gpu_i_ridge_flops_per_byte=gpu_i_ridge,
        gpu_roofline_calibration_ref=gpu_roofline_calibration_ref,
        gpu_freq_level_id=gpu_freq_level_id,
        freq_tolerance_fraction=freq_tolerance_fraction,
        freq_expected_cpu_count=freq_expected_cpu_count,
        freq_grace_seconds=freq_grace_seconds,
        freq_tail_grace_seconds=freq_tail_grace_seconds,
        freq_is_native_governor=freq_is_native_governor,
    )
    windows = build_windows(run_dir / "samples.csv", context)
    destination_dir = Path(output_dir) if output_dir is not None else run_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    windows_path = write_windows_csv(windows, destination_dir / "windows.csv")
    write_training_cpu_intervals_csv(
        build_training_cpu_intervals(windows),
        destination_dir / TRAINING_CPU_INTERVALS_FILENAME,
    )

    # ARC-174: el resumen de cobertura/calidad de frecuencia solo tiene
    # sentido para CPU -- GPU tiene otra cadencia/señal (gpu_sm_clock_mhz)
    # y sus propios factores G, deliberadamente fuera de este cambio.
    if getattr(kernel_entry, "device", "cpu") != "gpu":
        summary = validation_module.summarize_frequency_quality(windows_path)
        summary_path = destination_dir / "frequency_quality_summary.json"
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)
            summary_file.write("\n")

    return windows_path
