"""Añade dinámica temporal a la campaña histórica basada en gpu_windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fase2_clasificador.analysis.extract_historical_gpu_temporal import SIGNALS, temporal_features
from fase2_clasificador.analysis.historical_gpu_runs import build_historical_run_dataset


def build_dataset(path: Path) -> pd.DataFrame:
    columns = ["run_id", "t_end_ns", "window_index", "quality_status", "run_accepted", "phase_label_train", *SIGNALS]
    raw = pd.read_csv(path, usecols=columns, low_memory=False)
    runs, _ = build_historical_run_dataset([pd.read_csv(path, low_memory=False)])
    temporal_rows = []
    valid = raw[raw["quality_status"].eq("gpu_telemetry")].copy()
    for run_id, group in valid.groupby("run_id", sort=False):
        if run_id not in set(runs["run_id"]):
            continue
        samples = group.rename(columns={"t_end_ns": "timestamp_ns"})
        temporal_rows.append({"run_id": run_id, **temporal_features(samples)})
    temporal = pd.DataFrame(temporal_rows)
    result = runs.merge(temporal, on="run_id", how="inner", validate="one_to_one")
    if len(result) != len(runs):
        raise ValueError(f"cobertura temporal incompleta: {len(result)}/{len(runs)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"{len(result)} corridas temporales históricas escritas en {args.output}")


if __name__ == "__main__":
    main()
