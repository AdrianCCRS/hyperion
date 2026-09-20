"""Fusiona filas GPU históricas ya agregadas por corrida.

Acepta únicamente ``training_gpu_phases.csv`` con ``granularity=run``. Es un
contrato deliberadamente distinto del dataset CUPTI por ventana: no permite
mezclarlos, ni reetiqueta la verdad Roofline ya registrada por cada corrida.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fase1_telemetria.gpu_phases import GPU_PHASE_DATASET_FILENAME

FEATURES = ("gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median")
LABELS = {"compute_bound", "memory_bound"}
REQUIRED = {"run_id", "kernel_ref", "kernel_family", "gpu_freq_level_id", "phase_label_train",
            "granularity", "phase_quality_status", "training_eligible", "gpu_frequency_quality_status", *FEATURES}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_and_merge(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    if not paths:
        raise ValueError("no se recibió ningún training_gpu_phases.csv")
    frames = []
    rejected: dict[str, int] = {"not_run_granularity": 0, "not_eligible": 0, "quality_or_frequency": 0, "missing_label_or_feature": 0}
    input_rows = 0
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: faltan columnas requeridas: {sorted(missing)}")
        input_rows += len(frame)
        frame["source_file"] = str(path)
        frame["source_campaign_id"] = frame["run_id"].astype(str).str.split("__").str[0]
        run = frame["granularity"].eq("run")
        eligible = _as_bool(frame["training_eligible"])
        quality = frame["phase_quality_status"].eq("ok") & frame["gpu_frequency_quality_status"].isin({"valid", "not_applicable_native"})
        numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        labels = frame["phase_label_train"].isin(LABELS)
        rejected["not_run_granularity"] += int((~run).sum())
        rejected["not_eligible"] += int((run & ~eligible).sum())
        rejected["quality_or_frequency"] += int((run & eligible & ~quality).sum())
        rejected["missing_label_or_feature"] += int((run & eligible & quality & ~(numeric & labels)).sum())
        frames.append(frame[run & eligible & quality & numeric & labels].copy())
    valid_frames = [frame for frame in frames if not frame.empty]
    if not valid_frames:
        raise ValueError("ninguna corrida pasó los gates históricos")
    merged = pd.concat(valid_frames, ignore_index=True)
    duplicated = merged["run_id"].duplicated(keep=False)
    if duplicated.any():
        values = sorted(merged.loc[duplicated, "run_id"].unique())
        raise ValueError(f"run_id duplicado entre campañas: {values[:5]}")
    report = {
        "schema": "historical_gpu_run_merge/1",
        "row_unit": "accepted historical run",
        "forbidden_mix": "CUPTI time_window rows are rejected",
        "input_files": len(paths), "input_rows": input_rows, "accepted_runs": len(merged),
        "rejected": rejected,
        "campaigns": sorted(merged["source_campaign_id"].unique()),
        "families": int(merged["kernel_family"].nunique()),
        "class_balance": merged["phase_label_train"].value_counts().to_dict(),
    }
    return merged.sort_values("run_id").reset_index(drop=True), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="archivos o directorios que contienen training_gpu_phases.csv")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted({file for item in args.inputs for file in
                    ([item] if item.name == GPU_PHASE_DATASET_FILENAME else item.rglob(GPU_PHASE_DATASET_FILENAME))})
    merged, report = load_and_merge(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(merged)} corridas históricas fusionadas en {args.output}")


if __name__ == "__main__":
    main()
