"""F1-GPU-003 -- contrato de granularidad GPU y dataset intermedio por ventana.

Problema
--------
`postprocess.py` produce, para GPU, una fila por muestra NVML periódica, todas
con la MISMA intensidad operacional medida offline con `ncu` (constante por
kernel). Eso permite clasificar el régimen predominante del kernel, pero:

- una muestra NVML aislada NO es un ejemplo ML independiente (la evidencia de
  F1-GPU-002 mostró escalones de ~105-120 ms en potencia/utilización: muchas
  lecturas consecutivas comparten valor);
- no hay marcas de fase para kernels de terceros (la intercepción de
  `cudaLaunchKernel` vía LD_PRELOAD no funciona, ver `fase3_daemon/README.md`),
  así que no se pueden probar transiciones internas.

Contrato de granularidad GPU (formal)
-------------------------------------
- Unidad de fila del dataset de entrenamiento GPU = **una ventana temporal de
  120 ms** alineada a la resolución física observada de NVML. NUNCA una
  muestra NVML periódica de 5 ms.
- Las features NVML de esa fila son agregados robustos de las muestras NVML
  que caen en la misma ventana. La etiqueta se deriva de la actividad CUDA
  CUPTI y del trabajo analítico de los lanzamientos que se solapan con ella.
- La etiqueta Roofline, la intensidad y el ridge son verdad/trazabilidad; el
  entrenador GPU no puede leerlos como features (fuga).
- `gpu_phasic_*` (microbenchmarks sintéticos propios con fases programadas)
  NO es elegible para entrenamiento con la etiqueta constante del catálogo:
  solo lo sería si existieran marcas de fase y verdad offline alineada. Por
  defecto queda como control diagnóstico (`training_eligible = False`).

Este módulo no entrena nada. Se ejecuta como parte del postproceso GPU y
produce `training_gpu_phases.csv` + `training_gpu_phases_contract.json`.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

GPU_PHASE_DATASET_FILENAME = "training_gpu_phases.csv"
GPU_PHASE_CONTRACT_FILENAME = "training_gpu_phases_contract.json"

# Prefijos de kernels sintéticos propios con fases programadas. Sin marcas de
# fase + verdad offline alineada NO entran a entrenamiento.
PHASIC_CONTROL_PREFIXES: tuple[str, ...] = ("gpu_phasic", "phasic")

# Señales NVML que se agregan (nombre de columna en windows.csv de GPU).
_SIGNALS: tuple[str, ...] = (
    "gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw",
    "gpu_sm_clock_mhz", "gpu_temperature_c",
)

_AGG_SUFFIXES = ("median", "trimmed_mean", "std", "iqr", "min", "max",
                 "n_distinct", "valid_frac")

# El reloj SM de una A100 cae al estado idle aunque el lock de aplicación
# siga vigente. Solo las muestras que realmente observaron trabajo GPU son
# evidencia para validar el reloj solicitado.
_GPU_UTIL_NOISE_FLOOR_PCT = 5.0

# Verdad Roofline / trazabilidad: se copia tal cual, el entrenador no la lee.
_TRACE_COLUMNS: tuple[str, ...] = (
    "run_id", "repetition", "kernel_ref", "node_id",
    "freq_level_id", "gpu_freq_level_id",
    "binary_checksum", "roofline_calibration_ref",
    "operational_intensity", "i_ridge_used", "phase_label_train",
)

GPU_PHASE_COLUMNS: tuple[str, ...] = (
    *_TRACE_COLUMNS,
    "kernel_family",
    "granularity",              # "run" histórico | "time_window"
    "window_start_ns", "window_end_ns", "window_duration_ns",
    "cuda_active_ns", "cuda_active_fraction", "cuda_launch_count",
    "analytic_flops", "analytic_bytes_moved", "analytic_model_ids",
    "phase_quality_status",     # ok | insufficient_samples | label_missing | no_cuda_activity | ...
    "phase_quality_reason",
    "training_eligible",        # bool
    "gpu_freq_mhz_requested",
    "gpu_freq_mhz_applied",
    "gpu_frequency_quality_status",  # valid | invalid | not_applicable_native
    "gpu_frequency_valid_fraction",
    "n_nvml_samples",
    "n_nvml_samples_warmup_excluded",
    "n_nvml_samples_transition_excluded",
    "usable_sample_fraction",
    "covered_duration_ns",
    "gpu_energy_delta_mj_sum",
    "gpu_energy_covered",
    *(f"{sig}_{suf}" for sig in _SIGNALS for suf in _AGG_SUFFIXES),
)


# --------------------------------------------------------------- aggregation

def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _median(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _trimmed_mean(xs: Sequence[float], trim: float = 0.1) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = int(len(s) * trim)
    core = s[k:len(s) - k] if len(s) - 2 * k >= 1 else s
    return sum(core) / len(core)


def _std(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _iqr(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    def q(p: float) -> float:
        idx = p * (len(s) - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return q(0.75) - q(0.25)


def _aggregate_signal(values_with_validity: list[float | None]) -> dict[str, float | None]:
    valid = [v for v in values_with_validity if v is not None]
    total = len(values_with_validity)
    return {
        "median": _median(valid),
        "trimmed_mean": _trimmed_mean(valid),
        "std": _std(valid),
        "iqr": _iqr(valid),
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
        # frescura: nº de valores distintos (cota inferior de actualizaciones
        # físicas del sensor -- ver F1-GPU-002). 1 = el sensor nunca cambió.
        "n_distinct": float(len({round(v, 6) for v in valid})) if valid else 0.0,
        "valid_frac": (len(valid) / total) if total else 0.0,
    }


def _kernel_family(kernel_ref: str) -> str:
    try:
        from fase2_clasificador.eval.protocol import derive_kernel_family
        return derive_kernel_family(kernel_ref)
    except Exception:
        return kernel_ref


def _is_phasic_control(kernel_ref: str) -> bool:
    return any(kernel_ref.startswith(p) for p in PHASIC_CONTROL_PREFIXES)


# --------------------------------------------------------------- builder

def granularity_contract() -> dict[str, Any]:
    """Contrato de granularidad GPU, escrito como sidecar junto al CSV."""
    return {
        "schema": "f1-gpu-004/gpu_time_window_contract/1",
        "row_unit": "time_window",
        "window_ns": 120_000_000,
        "row_unit_legacy": "run (only for historical datasets without CUPTI Activity)",
        "nvml_sample_is_independent_example": False,
        "nvml_features_are": "robust aggregates over valid NVML samples inside the same time window",
        "label_source": "CUPTI Activity timestamps + analytical FLOPs/bytes per CUDA launch vs precision/frequency-specific ridge",
        "launch_boundary_rule": "FLOPs/bytes are prorated by temporal overlap; uniform work-rate assumption",
        "label_and_truth_columns_forbidden_as_features": [
            "operational_intensity", "i_ridge_used", "phase_label_train",
        ],
        "phasic_kernels_training_eligible": False,
        "phasic_kernels_note": "eligible only with aligned phase marks + offline truth",
    }


def build_gpu_phase_rows(
    windows: Iterable[dict[str, Any]],
    *,
    min_nvml_samples: int = 8,
    min_usable_sample_fraction: float = 0.5,
    phase_marks: dict[str, list] | None = None,
    gpu_freq_mhz_requested: int | None = None,
    gpu_freq_mhz_applied: int | None = None,
    gpu_freq_tolerance_fraction: float = 0.05,
) -> list[dict[str, Any]]:
    """Agrupa las ventanas GPU de `windows.csv` por corrida y produce una fila
    de dataset por corrida (o fase, si `phase_marks` -- hook no implementado).

    Una fila NUNCA representa una muestra NVML periódica aislada. Filas no
    usables se emiten con `phase_quality_status`/`training_eligible=False` para
    auditoría; el entrenador solo acepta `training_eligible == True`.
    """
    by_run: dict[str, list[dict[str, Any]]] = {}
    warmup_excluded: dict[str, int] = {}
    # F1-GPU-002/ARC-155/170: T_transicion_gpu_ns_conservative, distinto del
    # warmup del propio kernel -- ver postprocess.py::run_postprocess().
    transition_excluded: dict[str, int] = {}
    for w in windows:
        qs = w.get("quality_status")
        if qs not in ("gpu_telemetry", "warmup_excluded", "excluded_transition_not_settled"):
            continue  # no es una fila GPU
        rid = w.get("run_id")
        if rid is None:
            continue
        if qs == "warmup_excluded":
            warmup_excluded[rid] = warmup_excluded.get(rid, 0) + 1
            continue
        if qs == "excluded_transition_not_settled":
            transition_excluded[rid] = transition_excluded.get(rid, 0) + 1
            continue
        by_run.setdefault(rid, []).append(w)

    out: list[dict[str, Any]] = []
    for rid, rows in sorted(by_run.items()):
        rows = sorted(rows, key=lambda r: int(r.get("t_end_ns") or 0))
        first = rows[0]
        result: dict[str, Any] = {c: None for c in GPU_PHASE_COLUMNS}
        for c in _TRACE_COLUMNS:
            result[c] = first.get(c)
        kernel_ref = first.get("kernel_ref") or ""
        result["kernel_family"] = _kernel_family(kernel_ref)
        result["granularity"] = "run"
        result["gpu_freq_mhz_requested"] = gpu_freq_mhz_requested
        result["gpu_freq_mhz_applied"] = gpu_freq_mhz_applied
        result["n_nvml_samples"] = len(rows)
        result["n_nvml_samples_warmup_excluded"] = warmup_excluded.get(rid, 0)
        result["n_nvml_samples_transition_excluded"] = transition_excluded.get(rid, 0)

        n_total_gpu = len(rows) + warmup_excluded.get(rid, 0) + transition_excluded.get(rid, 0)
        result["usable_sample_fraction"] = (len(rows) / n_total_gpu) if n_total_gpu else 0.0

        ts = [int(r["t_end_ns"]) for r in rows if r.get("t_end_ns") not in (None, "")]
        result["covered_duration_ns"] = (max(ts) - min(ts)) if len(ts) >= 2 else 0

        energy_deltas = [
            _to_float(r.get("gpu_energy_delta_mj"))
            for r in rows
            if str(r.get("gpu_energy_valid")).lower() in ("true", "1")
        ]
        energy_deltas = [e for e in energy_deltas if e is not None]
        result["gpu_energy_delta_mj_sum"] = sum(energy_deltas) if energy_deltas else None
        result["gpu_energy_covered"] = bool(energy_deltas)

        for sig in _SIGNALS:
            agg = _aggregate_signal([_to_float(r.get(sig)) for r in rows])
            for suf in _AGG_SUFFIXES:
                result[f"{sig}_{suf}"] = agg[suf]

        clocks = [_to_float(row.get("gpu_sm_clock_mhz")) for row in rows]
        clocks = [clock for clock in clocks if clock is not None]
        if gpu_freq_mhz_applied is None:
            result["gpu_frequency_quality_status"] = "not_applicable_native"
            result["gpu_frequency_valid_fraction"] = None
        else:
            tolerance = max(abs(gpu_freq_mhz_applied) * gpu_freq_tolerance_fraction, 1.0)
            valid_count = sum(abs(clock - gpu_freq_mhz_applied) <= tolerance for clock in clocks)
            valid_fraction = valid_count / len(clocks) if clocks else 0.0
            result["gpu_frequency_valid_fraction"] = valid_fraction
            result["gpu_frequency_quality_status"] = "valid" if valid_fraction >= 0.9 else "invalid"

        # --- calidad / elegibilidad ---
        label = first.get("phase_label_train")
        status, reason, eligible = "ok", "", True
        if _is_phasic_control(kernel_ref):
            status, eligible = "phasic_control_needs_marks", False
            reason = ("kernel sintético con fases programadas: sin marcas de fase "
                      "alineadas con verdad offline, solo control diagnóstico")
        elif label in (None, "", "nan"):
            status, eligible = "label_missing", False
            reason = "sin phase_label_train (falta calibración ncu/ridge de GPU para esta precisión/frecuencia)"
        elif len(rows) == 0:
            status, eligible = "no_usable_samples", False
        elif len(rows) < min_nvml_samples:
            status, eligible = "insufficient_samples", False
            reason = f"{len(rows)} < min_nvml_samples={min_nvml_samples}"
        elif result["usable_sample_fraction"] < min_usable_sample_fraction:
            status, eligible = "insufficient_samples", False
            reason = (f"usable_sample_fraction={result['usable_sample_fraction']:.2f} "
                      f"< {min_usable_sample_fraction}")
        elif result["gpu_frequency_quality_status"] == "invalid":
            status, eligible = "gpu_frequency_invalid", False
            reason = (f"solo {result['gpu_frequency_valid_fraction']:.1%} de clocks NVML "
                      f"dentro de ±{gpu_freq_tolerance_fraction:.1%} de "
                      f"{gpu_freq_mhz_applied} MHz")
        result["phase_quality_status"] = status
        result["phase_quality_reason"] = reason
        result["training_eligible"] = eligible
        out.append(result)
    return out


def build_gpu_time_window_rows(
    windows: Iterable[dict[str, Any]],
    *,
    launch_work: Iterable[Any],
    i_ridge_flops_per_byte: float | None,
    window_ns: int = 120_000_000,
    min_nvml_samples: int = 8,
    min_usable_sample_fraction: float = 0.5,
    gpu_freq_mhz_requested: int | None = None,
    gpu_freq_mhz_applied: int | None = None,
    gpu_freq_tolerance_fraction: float = 0.05,
    measured_start_ns: int | None = None,
    measured_end_ns: int | None = None,
) -> list[dict[str, Any]]:
    """Construye ejemplos GPU por ventana temporal, nunca por muestra NVML.

    ``launch_work`` viene de CUPTI Activity + un modelo analítico por
    lanzamiento. FLOPs y bytes de un lanzamiento que cruza un borde se
    prorratean por solape temporal (supuesto explícito de tasa uniforme).
    Una ventana sin actividad CUDA no recibe etiqueta y no entra a entrenar.
    """
    from fase1_telemetria.gpu_window_oi import aggregate_launches_in_window

    if window_ns <= 0:
        raise ValueError("window_ns debe ser positivo")
    launches = list(launch_work)
    if (measured_start_ns is None) != (measured_end_ns is None):
        raise ValueError("la región medida GPU requiere inicio y fin juntos")
    if measured_start_ns is not None and measured_end_ns <= measured_start_ns:
        raise ValueError("la región medida GPU requiere start < end")
    usable_statuses = {"gpu_telemetry"}
    # Si el binario declara con timestamps MONOTONIC su región realmente
    # medida, esos límites sustituyen el recorte heurístico de warmup: las
    # muestras ya están fuera de setup/cold por construcción.
    if measured_start_ns is not None:
        usable_statuses.add("warmup_excluded")
    usable = [
        row for row in windows
        if row.get("quality_status") in usable_statuses
        and row.get("t_end_ns") not in (None, "")
        and (measured_start_ns is None or measured_start_ns <= int(row["t_end_ns"]) <= measured_end_ns)
    ]
    if not usable:
        return []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in usable:
        by_run.setdefault(str(row["run_id"]), []).append(row)

    out: list[dict[str, Any]] = []
    for run_id, samples in sorted(by_run.items()):
        samples.sort(key=lambda row: int(row["t_end_ns"]))
        first_ts = int(samples[0]["t_end_ns"])
        last_ts = int(samples[-1]["t_end_ns"])
        # El origen se fija a la primera muestra post-warmup del run. No se
        # pretende inventar una fase anterior que NVML no observó.
        cursor = first_ts
        ordinal = 0
        while cursor <= last_ts:
            end_ns = cursor + window_ns
            sample_rows = [
                row for row in samples
                if cursor <= int(row["t_end_ns"]) < end_ns
            ]
            if not sample_rows:
                cursor = end_ns
                ordinal += 1
                continue
            truth = aggregate_launches_in_window(
                launches, start_ns=cursor, end_ns=end_ns
            )
            model_ids = sorted({
                launch.model_id for launch in launches
                if getattr(launch, "model_id", None)
                and launch.start_ns < end_ns and launch.end_ns > cursor
            })
            first = sample_rows[0]
            result: dict[str, Any] = {c: None for c in GPU_PHASE_COLUMNS}
            for c in _TRACE_COLUMNS:
                result[c] = first.get(c)
            kernel_ref = first.get("kernel_ref") or ""
            result.update({
                "kernel_family": _kernel_family(kernel_ref),
                "granularity": "time_window",
                "window_start_ns": cursor,
                "window_end_ns": end_ns,
                "window_duration_ns": window_ns,
                "cuda_active_ns": truth.active_ns,
                "cuda_active_fraction": truth.active_fraction,
                "cuda_launch_count": truth.overlapping_launches,
                "analytic_flops": truth.flops,
                "analytic_bytes_moved": truth.bytes_moved,
                "analytic_model_ids": ";".join(model_ids),
                "gpu_freq_mhz_requested": gpu_freq_mhz_requested,
                "gpu_freq_mhz_applied": gpu_freq_mhz_applied,
                "n_nvml_samples": len(sample_rows),
                "n_nvml_samples_warmup_excluded": 0,
                "n_nvml_samples_transition_excluded": 0,
                "usable_sample_fraction": 1.0,
                "covered_duration_ns": (
                    int(sample_rows[-1]["t_end_ns"]) - int(sample_rows[0]["t_end_ns"])
                    if len(sample_rows) >= 2 else 0
                ),
            })
            for sig in _SIGNALS:
                agg = _aggregate_signal([_to_float(row.get(sig)) for row in sample_rows])
                for suffix in _AGG_SUFFIXES:
                    result[f"{sig}_{suffix}"] = agg[suffix]
            energy_deltas = [
                _to_float(row.get("gpu_energy_delta_mj")) for row in sample_rows
                if str(row.get("gpu_energy_valid")).lower() in ("true", "1")
            ]
            energy_deltas = [v for v in energy_deltas if v is not None]
            result["gpu_energy_delta_mj_sum"] = sum(energy_deltas) if energy_deltas else None
            result["gpu_energy_covered"] = bool(energy_deltas)

            active_sample_rows = [
                row for row in sample_rows
                if (_to_float(row.get("gpu_util_pct")) or 0.0) >= _GPU_UTIL_NOISE_FLOOR_PCT
            ]
            clocks = [
                v for v in (_to_float(r.get("gpu_sm_clock_mhz")) for r in active_sample_rows)
                if v is not None
            ]
            if gpu_freq_mhz_applied is None:
                result["gpu_frequency_quality_status"] = "not_applicable_native"
            else:
                tolerance = max(abs(gpu_freq_mhz_applied) * gpu_freq_tolerance_fraction, 1.0)
                fraction = sum(abs(clock - gpu_freq_mhz_applied) <= tolerance for clock in clocks) / len(clocks) if clocks else 0.0
                result["gpu_frequency_valid_fraction"] = fraction
                result["gpu_frequency_quality_status"] = "valid" if fraction >= 0.9 else "invalid"

            status, reason, eligible = "ok", "", True
            if _is_phasic_control(kernel_ref):
                status, reason, eligible = "phasic_control_needs_marks", "control sintético sin verdad por fase", False
            elif truth.active_ns <= 0:
                status, reason, eligible = "no_cuda_activity", "ningún lanzamiento CUDA se solapa con la ventana", False
            elif i_ridge_flops_per_byte is None:
                status, reason, eligible = "label_missing", "sin ridge GPU calibrado para precisión/frecuencia", False
            elif len(sample_rows) < min_nvml_samples:
                status, reason, eligible = "insufficient_samples", f"{len(sample_rows)} < min_nvml_samples={min_nvml_samples}", False
            elif result["gpu_frequency_quality_status"] == "invalid":
                status, reason, eligible = "gpu_frequency_invalid", "reloj SM fuera de tolerancia", False
            else:
                result["operational_intensity"] = truth.operational_intensity
                result["i_ridge_used"] = i_ridge_flops_per_byte
                result["phase_label_train"] = (
                    "memory_bound" if truth.operational_intensity < i_ridge_flops_per_byte
                    else "compute_bound"
                )
            result["phase_quality_status"] = status
            result["phase_quality_reason"] = reason
            result["training_eligible"] = eligible
            out.append(result)
            cursor = end_ns
            ordinal += 1
    return out


def write_gpu_phases_csv(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    path = Path(out_path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(GPU_PHASE_COLUMNS)
        for r in rows:
            writer.writerow(["" if r.get(c) is None else r.get(c) for c in GPU_PHASE_COLUMNS])
    return path


def write_contract(out_path: str | Path) -> Path:
    path = Path(out_path)
    path.write_text(json.dumps(granularity_contract(), indent=2, ensure_ascii=False))
    return path
