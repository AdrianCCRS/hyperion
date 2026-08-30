"""Orquestador reproducible de la Fase R1 del selector CPU/GPU.

R1 no entrena un modelo. Construye el problema compacto, audita su sanidad,
calcula amortizacion y headroom DVFS, y evalua baselines con particiones por
tamano. La unidad independiente es siempre ``config_id``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

from .compact import (
    DEFAULT_Z_SCORE,
    amortization_map,
    attach_probe_features,
    build_compact_dataset,
    dvfs_headroom,
)
from .datacard import build_datacard, write_datacard
from .sizes import (
    baseline_headroom_report,
    extrapolation_folds,
    interpolation_folds,
    run_baselines,
)


class R1ContractError(ValueError):
    """Los artefactos de nivel 2 no satisfacen el contrato de R1."""


def _read_required(dataset_dir: Path, name: str) -> pd.DataFrame:
    path = dataset_dir / name
    if not path.is_file():
        raise R1ContractError(f"falta artefacto de nivel 2: {path}")
    return pd.read_csv(path, low_memory=False)


def _fold_manifest(
    folds: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name, train, test in folds:
        for split, frame in (("train", train), ("test", test)):
            for row in frame[["config_id", "operation", "size"]].drop_duplicates().to_dict("records"):
                records.append({"fold": name, "split": split, **row})
    return pd.DataFrame(records)


def _finite_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "count": int(len(values)),
        "min": float(values.min()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def _summary(
    compact: pd.DataFrame,
    amortization: pd.DataFrame,
    headroom: pd.DataFrame,
    signal: pd.DataFrame,
) -> dict[str, Any]:
    finite = np.isfinite(amortization["k_break_even"])
    return {
        "effective_config_count": int(compact["config_id"].nunique()),
        "compact_row_count": int(len(compact)),
        "device_labels_by_resource_state": {
            str(state): group["device_label"].value_counts().to_dict()
            for state, group in compact.groupby("resource_state", observed=True)
        },
        "device_decision_stability_by_resource_state": {
            str(state): {
                "separated": int(group["device_decision_separated"].sum()),
                "uncertain": int((1 - group["device_decision_separated"]).sum()),
                "uncertainty_methods": sorted(
                    group["device_decision_uncertainty_method"].astype(str).unique()
                ),
            }
            for state, group in compact.groupby("resource_state", observed=True)
        },
        "device_decision_exploratory_ci_by_resource_state": {
            str(state): {
                "available": int(group["device_decision_ci_separated_normal_approx"].notna().sum()),
                "separated": int(group["device_decision_ci_separated_normal_approx"].fillna(0).sum()),
                "uncertain": int(
                    group["device_decision_ci_separated_normal_approx"].notna().sum()
                    - group["device_decision_ci_separated_normal_approx"].fillna(0).sum()
                ),
                "status": "exploratory_not_frozen_rule",
            }
            for state, group in compact.groupby("resource_state", observed=True)
        },
        "amortization": {
            "finite_count": int(finite.sum()),
            "infinite_count": int((~finite).sum()),
            "finite_k": _finite_summary(amortization.loc[finite, "k_break_even"]),
            "uncertainty_contract": (
                "empirical marginal-extrema envelope; not a confidence interval"
            ),
        },
        "dvfs_headroom_by_resource_state": {
            str(state): {
                "count": int(len(group)),
                "above_noise_floor": int(group["above_noise_floor"].sum()),
                "median_pct": float(group["dvfs_headroom_pct"].median()),
                "p95_pct": float(group["dvfs_headroom_pct"].quantile(0.95)),
                "max_pct": float(group["dvfs_headroom_pct"].max()),
            }
            for state, group in headroom.groupby("resource_state", observed=True)
        },
        "baseline_oracle_headroom": signal.to_dict("records"),
        "interpretation_contract": {
            "k_is_supplied_not_predicted": True,
            "device_generalization_domain": "known_operations_unseen_sizes",
            "config_id_is_independent_unit": True,
            "probe_is_one_real_dispatch_not_three_run_mean": True,
            "dvfs_headroom_is_upper_bound_before_actuation_cost": True,
            "sum_dispatch_edp_is_not_application_edp": True,
        },
    }


def run_r1_analysis(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    z_score: float = DEFAULT_Z_SCORE,
) -> dict[str, Path]:
    """Ejecuta R1 desde artefactos de nivel 2 y escribe resultados auditables."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / "r1"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _read_required(dataset_dir, "candidate_summary.csv")
    run_regions = _read_required(dataset_dir, "run_regions.csv")

    compact_static = build_compact_dataset(candidates, z_score=z_score)
    compact_probe = attach_probe_features(compact_static, run_regions)
    amortization = amortization_map(candidates, run_regions)
    headroom = dvfs_headroom(candidates)

    interpolation = interpolation_folds(compact_static)
    extrapolation = extrapolation_folds(compact_static)
    interpolation_results = run_baselines(interpolation)
    extrapolation_results = run_baselines(extrapolation)
    baseline_results = pd.concat(
        [interpolation_results, extrapolation_results], ignore_index=True,
    )
    signal = baseline_headroom_report(baseline_results)
    folds = [*interpolation, *extrapolation]

    card = build_datacard(
        compact_probe,
        candidates,
        amortization=amortization,
        headroom=headroom,
        folds=folds,
        run_regions=run_regions,
    )
    card_paths = write_datacard(card, output_dir)

    paths = {
        "compact_static": output_dir / "compact_static.csv",
        "compact_with_probe": output_dir / "compact_with_probe.csv",
        "amortization_map": output_dir / "amortization_map.csv",
        "dvfs_headroom": output_dir / "dvfs_headroom.csv",
        "size_folds": output_dir / "size_folds.csv",
        "interpolation_baselines": output_dir / "interpolation_baselines.csv",
        "extrapolation_baselines": output_dir / "extrapolation_baselines.csv",
        "baseline_metrics": output_dir / "baseline_metrics.csv",
        "baseline_oracle_headroom": output_dir / "baseline_oracle_headroom.csv",
        "datacard_json": Path(card_paths["json"]),
        "datacard_markdown": Path(card_paths["markdown"]),
        "summary": output_dir / "r1_summary.json",
    }
    frames = {
        "compact_static": compact_static,
        "compact_with_probe": compact_probe,
        "amortization_map": amortization,
        "dvfs_headroom": headroom,
        "size_folds": _fold_manifest(folds),
        "interpolation_baselines": interpolation_results,
        "extrapolation_baselines": extrapolation_results,
        "baseline_metrics": baseline_results,
        "baseline_oracle_headroom": signal,
    }
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)
    paths["summary"].write_text(
        json.dumps(
            _summary(compact_static, amortization, headroom, signal),
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    return paths
