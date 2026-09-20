"""Construye observaciones GPU históricas independientes por corrida.

Esta ruta existe únicamente para campañas previas a CUPTI Activity. No
convierte ventanas NVML en fases: conserva la etiqueta Roofline histórica de
la corrida y resume solo las muestras NVML estables. La ruta nueva por ventana
(``training_gpu_phases.csv``) continúa siendo el contrato para CUPTI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fase2_clasificador.eval.protocol import derive_kernel_family

SIGNALS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw", "gpu_sm_clock_mhz")
LABELS = {"compute_bound", "memory_bound"}
DEFAULT_QUALITY = frozenset({"gpu_telemetry"})


def _as_bool(series: pd.Series) -> pd.Series:
    """Convierte CSVs con booleanos reales o texto sin aceptar ``'False'``."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _duration_weights(frame: pd.DataFrame) -> np.ndarray:
    """Usa delta explícito o, en campañas antiguas GPU, timestamps NVML."""
    explicit = pd.to_numeric(frame.get("delta_t_ns"), errors="coerce")
    if explicit.notna().any() and explicit.gt(0).any():
        return explicit.to_numpy(dtype=float)
    # La campaña de 2026-08 guardó el timestamp final de NVML, pero no un
    # inicio ni delta. Cada lectura representa el intervalo hasta la próxima
    # lectura distinta; al último punto se le asigna la cadencia mediana.
    if "t_end_ns" not in frame:
        return np.full(len(frame), np.nan)
    ends = pd.to_numeric(frame["t_end_ns"], errors="coerce").to_numpy(dtype=float)
    if len(ends) < 2 or not np.isfinite(ends).any():
        return np.full(len(frame), np.nan)
    deltas = np.diff(ends)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    if not len(positive):
        return np.full(len(frame), np.nan)
    last = float(np.median(positive))
    return np.append(deltas, last)


def _aggregate_group(run_id: str, group: pd.DataFrame) -> dict[str, object]:
    labels = set(group["phase_label_train"].dropna().astype(str))
    if labels - LABELS or len(labels) != 1:
        raise ValueError(
            f"{run_id!r}: la etiqueta histórica debe ser una sola de {sorted(LABELS)}, "
            f"se encontró {sorted(labels)}"
        )
    row: dict[str, object] = {
        "run_id": run_id,
        "source_campaign_id": group["source_campaign_id"].iloc[0],
        "kernel_ref": group["kernel_ref"].iloc[0],
        "kernel_family": derive_kernel_family(str(group["kernel_ref"].iloc[0])),
        "gpu_freq_level_id": group["gpu_freq_level_id"].iloc[0],
        "phase_label_train": next(iter(labels)),
        "n_nvml_samples": int(len(group)),
        "aggregation_granularity": "historical_run",
    }
    weights = _duration_weights(group)
    for signal in SIGNALS:
        values = pd.to_numeric(group[signal], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"{run_id!r}: no hay muestras válidas para {signal}")
        row[f"{signal}_median"] = float(np.median(values))
        row[f"{signal}_time_weighted_mean"] = _weighted_mean(
            pd.to_numeric(group[signal], errors="coerce").to_numpy(dtype=float), weights
        )
        row[f"{signal}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        row[f"{signal}_p90"] = float(np.percentile(values, 90))
    # Una alternativa de sensibilidad que no usa identificadores ni verdad
    # Roofline: potencia normalizada por el reloj efectivo de la corrida.
    clock = float(row["gpu_sm_clock_mhz_median"])
    row["gpu_power_mw_per_mhz_median"] = float(row["gpu_power_mw_median"]) / clock if clock > 0 else float("nan")
    return row


def build_historical_run_dataset(
    frames: Iterable[pd.DataFrame],
    allowed_quality: frozenset[str] = DEFAULT_QUALITY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Filtra muestras y devuelve una fila independiente por ``run_id``.

    Las filas de warmup, de una corrida rechazada o con cualquiera de las
    cuatro señales NVML ausentes se descartan antes del agregado. Una corrida
    que quede sin muestras estables no se conserva silenciosamente.
    """
    source = pd.concat(list(frames), ignore_index=True)
    required = {"run_id", "kernel_ref", "gpu_freq_level_id", "phase_label_train",
                "run_accepted", "quality_status", "delta_t_ns", *SIGNALS}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"faltan columnas históricas requeridas: {sorted(missing)}")
    if "source_campaign_id" not in source:
        source["source_campaign_id"] = "unknown_legacy_campaign"

    initial_rows = len(source)
    accepted = _as_bool(source["run_accepted"])
    quality = source["quality_status"].fillna("").astype(str).isin(allowed_quality)
    numeric = source.loc[:, SIGNALS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    stable = source[accepted & quality & numeric & source["phase_label_train"].isin(LABELS)].copy()
    if stable.empty:
        raise ValueError("ninguna muestra histórica sobrevivió los filtros de calidad")

    identity_cols = ("source_campaign_id", "kernel_ref", "gpu_freq_level_id")
    for run_id, group in stable.groupby("run_id", sort=True):
        if any(group[col].nunique(dropna=False) != 1 for col in identity_cols):
            raise ValueError(f"{run_id!r}: identidad de corrida inconsistente entre muestras")
    runs = pd.DataFrame(_aggregate_group(run_id, group) for run_id, group in stable.groupby("run_id", sort=True))
    runs = runs.sort_values("run_id").reset_index(drop=True)
    report = {
        "schema": "historical_gpu_run_dataset/1",
        "row_unit": "accepted_run_after_stable_nvml_filter",
        "label_source": "historical static Roofline label; no CUPTI-derived phases",
        "allowed_quality_status": sorted(allowed_quality),
        "input_rows": initial_rows,
        "stable_rows": int(len(stable)),
        "excluded_rows": int(initial_rows - len(stable)),
        "accepted_runs": int(len(runs)),
        "kernels": int(runs["kernel_ref"].nunique()),
        "families": int(runs["kernel_family"].nunique()),
        "source_campaigns": sorted(runs["source_campaign_id"].astype(str).unique()),
        "class_balance": runs["phase_label_train"].value_counts().to_dict(),
    }
    return runs, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV(s) windows.csv.gz de campañas históricas")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-quality", default="gpu_telemetry",
                        help="Estados quality_status admitidos, separados por coma")
    args = parser.parse_args()
    allowed = frozenset(x.strip() for x in args.allow_quality.split(",") if x.strip())
    # Las campañas archivadas tienen millones de ventanas y muchas columnas
    # CPU/Roofline que aquí están prohibidas como features. Leer solo el
    # contrato histórico reduce memoria y hace que el reanálisis sea viable.
    needed = ["run_id", "kernel_ref", "gpu_freq_level_id", "phase_label_train",
              "run_accepted", "quality_status", "delta_t_ns", "t_end_ns", "source_campaign_id", *SIGNALS]
    def read_contract(path: Path) -> pd.DataFrame:
        header = set(pd.read_csv(path, nrows=0).columns)
        return pd.read_csv(path, usecols=[name for name in needed if name in header], low_memory=False)
    runs, report = build_historical_run_dataset(
        (read_contract(path) for path in args.inputs), allowed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output, index=False)
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(runs)} corridas históricas escritas en {args.output}")
    print(f"Reporte de calidad: {report_path}")


if __name__ == "__main__":
    main()
