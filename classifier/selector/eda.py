"""EDA reproducible a granularidad de corrida/configuracion."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from . import label_health


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(7.0, min(18.0, 0.42 * max(1, len(matrix.columns))))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _numeric_correlations(frame: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).copy()
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    result: dict[str, Any] = {"rows": len(frame), "numeric_columns": list(numeric.columns)}
    if numeric.empty:
        return result
    for method in ("pearson", "spearman"):
        matrix = numeric.corr(method=method)
        matrix.to_csv(output_dir / f"{stem}_{method}.csv")
        _save_heatmap(matrix, output_dir / f"{stem}_{method}.png", f"{stem} — {method}")
    missing = frame.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    missing.to_csv(output_dir / f"{stem}_missingness.csv")
    result["columns_with_missing"] = int((missing > 0).sum())
    return result


def generate_eda(dataset_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir else dataset_dir / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "runs": dataset_dir / "run_regions.csv",
        "candidates": dataset_dir / "candidate_summary.csv",
        "strategy_a": dataset_dir / "strategy_a_candidates.csv",
        "strategy_c": dataset_dir / "strategy_c_candidates.csv",
    }
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in inputs.items() if path.exists()}
    if "runs" not in frames or "candidates" not in frames:
        raise FileNotFoundError("faltan run_regions.csv o candidate_summary.csv")

    summary: dict[str, Any] = {}
    for name, frame in frames.items():
        if frame.empty:
            summary[name] = {"rows": 0}
            continue
        if "region" in frame:
            summary[name] = {}
            for region, subset in frame.groupby("region", dropna=False, observed=True):
                summary[name][str(region)] = _numeric_correlations(
                    subset, output_dir, f"{name}_{region}",
                )
        else:
            summary[name] = _numeric_correlations(frame, output_dir, name)

    candidates = frames["candidates"]
    candidates[[
        "config_id", "operation", "size", "device", "action_id", "cpu_level",
        "gpu_level", "region", "time_mean", "energy_mean", "edp_mean",
        "time_cv_pct", "energy_cv_pct", "edp_cv_pct",
    ]].to_csv(output_dir / "candidate_outcomes_and_cv.csv", index=False)
    candidates.groupby(["operation", "region", "action_id"], observed=True).agg(
        n=("config_id", "nunique"),
        edp_mean=("edp_mean", "mean"),
        edp_cv_pct=("edp_cv_pct", "mean"),
        time_mean=("time_mean", "mean"),
        energy_mean=("energy_mean", "mean"),
    ).reset_index().to_csv(output_dir / "curves_by_operation_action.csv", index=False)
    candidates.groupby(
        ["operation", "size", "region", "device", "cpu_level", "gpu_level"],
        dropna=False, observed=True,
    ).agg(
        candidates=("config_id", "nunique"),
        time_mean=("time_mean", "mean"),
        energy_mean=("energy_mean", "mean"),
        edp_mean=("edp_mean", "mean"),
        time_cv_pct=("time_cv_pct", "mean"),
        energy_cv_pct=("energy_cv_pct", "mean"),
        edp_cv_pct=("edp_cv_pct", "mean"),
    ).reset_index().to_csv(output_dir / "cv_and_edp_by_size_frequency.csv", index=False)

    outcome_rows = []
    for (device, region), group in candidates.groupby(["device", "region"], observed=True):
        for outcome in ("time_mean", "energy_mean", "edp_mean"):
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            outcome_rows.append({
                "device": device, "region": region, "outcome": outcome,
                "count": len(values), "mean": values.mean(), "std": values.std(),
                "min": values.min(), "p25": values.quantile(0.25),
                "median": values.median(), "p75": values.quantile(0.75),
                "max": values.max(),
            })
    pd.DataFrame(outcome_rows).to_csv(output_dir / "outcome_distributions.csv", index=False)

    runs = frames["runs"]
    runs.groupby(["device", "region", "energy_resolution_status"], observed=True).size().rename(
        "rows"
    ).reset_index().to_csv(output_dir / "energy_resolution_distribution.csv", index=False)
    runs.loc[runs["region"] == "cold", [
        "run_id", "config_id", "operation", "device", "action_id",
        "region_to_sampling_ratio", "sampling_resolution_ns", "energy_resolution_status",
    ]].to_csv(output_dir / "cold_sampling_resolution.csv", index=False)

    for name in ("strategy_a", "strategy_c"):
        frame = frames.get(name)
        if frame is None or frame.empty:
            continue
        optima = frame[frame["is_optimal"] == 1]
        optima.groupby(["operation", "candidate_device", "action_id"], observed=True).size().rename(
            "count"
        ).reset_index().to_csv(output_dir / f"{name}_optimal_distribution.csv", index=False)
        pd.crosstab(optima["operation"], optima["candidate_device"]).to_csv(
            output_dir / f"{name}_operation_optimal_device.csv"
        )
        optima[[
            "decision_group_id", "config_id", "operation", "candidate_device",
            "action_id", "margin_edp_pct", "optimum_stability",
        ]].to_csv(output_dir / f"{name}_optimal_margins.csv", index=False)
        frame.groupby("is_optimal", observed=True).size().rename("count").to_csv(
            output_dir / f"{name}_class_distribution.csv"
        )
        frame.groupby("optimum_stability", observed=True).size().rename("count").to_csv(
            output_dir / f"{name}_stability_distribution.csv"
        )

    health: dict[str, Any] = {}
    for name in ("strategy_a", "strategy_c"):
        frame = frames.get(name)
        if frame is None or frame.empty or "is_optimal" not in frame:
            continue
        health[name] = label_health.assess_label_health(frame)
    if health:
        (output_dir / "label_health.json").write_text(
            json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["label_health"] = health

    report = output_dir / "eda_summary.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output_dir": output_dir, "summary": report}
