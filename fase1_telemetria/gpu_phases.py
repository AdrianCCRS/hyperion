"""F1-GPU-003 -- contrato de granularidad GPU y dataset intermedio por fase.

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
- Unidad de fila del dataset de entrenamiento GPU = **una corrida** (un
  `run_id` = kernel_ref x nivel_frecuencia_gpu x repeticion), o **una fase
  estable** si en el futuro existen marcas de fase alineadas con verdad
  offline. NUNCA una muestra NVML periódica.
- Las features NVML de esa fila son AGREGADOS robustos sobre las muestras NVML
  post-warmup y válidas de la corrida (mediana, media recortada, dispersión,
  min/max, duración cubierta, nº de muestras, nº de valores distintos como
  indicador de frescura, fracción de muestras usables).
- La etiqueta Roofline (`phase_label_train`), la intensidad operacional `ncu`
  y el ridge se conservan SOLO para trazabilidad y como verdad; el entrenador
  GPU no puede leerlas como features (fuga -- ver
  `fase2_clasificador/analysis/feature_contract.py`).
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
    "granularity",              # "run" | "phase"
    "phase_quality_status",     # ok | insufficient_samples | no_usable_samples | label_missing | phasic_control_needs_marks
    "phase_quality_reason",
    "training_eligible",        # bool
    "n_nvml_samples",
    "n_nvml_samples_warmup_excluded",
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
        "schema": "f1-gpu-003/gpu_phase_granularity_contract/1",
        "row_unit": "run",
        "row_unit_alternatives": ["phase (requires aligned phase marks + offline truth)"],
        "nvml_sample_is_independent_example": False,
        "nvml_features_are": "robust aggregates over post-warmup valid NVML samples of the run",
        "label_source": "offline ncu operational intensity vs precision/frequency-specific ridge",
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
) -> list[dict[str, Any]]:
    """Agrupa las ventanas GPU de `windows.csv` por corrida y produce una fila
    de dataset por corrida (o fase, si `phase_marks` -- hook no implementado).

    Una fila NUNCA representa una muestra NVML periódica aislada. Filas no
    usables se emiten con `phase_quality_status`/`training_eligible=False` para
    auditoría; el entrenador solo acepta `training_eligible == True`.
    """
    by_run: dict[str, list[dict[str, Any]]] = {}
    warmup_excluded: dict[str, int] = {}
    for w in windows:
        qs = w.get("quality_status")
        if qs not in ("gpu_telemetry", "warmup_excluded"):
            continue  # no es una fila GPU
        rid = w.get("run_id")
        if rid is None:
            continue
        if qs == "warmup_excluded":
            warmup_excluded[rid] = warmup_excluded.get(rid, 0) + 1
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
        result["n_nvml_samples"] = len(rows)
        result["n_nvml_samples_warmup_excluded"] = warmup_excluded.get(rid, 0)

        n_total_gpu = len(rows) + warmup_excluded.get(rid, 0)
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
        result["phase_quality_status"] = status
        result["phase_quality_reason"] = reason
        result["training_eligible"] = eligible
        out.append(result)
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
