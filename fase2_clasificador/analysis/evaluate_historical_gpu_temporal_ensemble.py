"""Combina de forma anidada un clasificador NVML estático y uno temporal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import (
    _family_weights, _fit, _folds, add_physical_features, feature_variants,
)
from fase2_clasificador.eval import protocol
from fase2_clasificador.training import model_specs

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPORAL_ARMS = (
    ("temporal_dynamics", "xgboost"),
    ("temporal_dynamics_iqr", "arbol_prof6"),
    ("temporal_dynamics_iqr", "random_forest"),
)


def _probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def _fit_pair(frame: pd.DataFrame, train_idx, test_idx, temporal_variant: str, temporal_model: str, seed: int):
    variants = feature_variants()
    y = (frame["phase_label_train"] == "memory_bound").to_numpy()
    pos, neg = int(y[train_idx].sum()), int(len(train_idx) - y[train_idx].sum())
    models = model_specs.build_models(seed, neg / pos if pos else 1.0)
    weights = _family_weights(frame.iloc[train_idx])
    static = clone(models["regresion_log"])
    temporal = clone(models[temporal_model])
    if "n_jobs" in temporal.get_params(deep=False):
        temporal.set_params(n_jobs=1)
    X_static = frame[variants["median_iqr"]].to_numpy(dtype=np.float32)
    X_temporal = frame[variants[temporal_variant]].to_numpy(dtype=np.float32)
    _fit(static, X_static[train_idx], y[train_idx], weights)
    _fit(temporal, X_temporal[train_idx], y[train_idx], weights)
    return y[test_idx], _probability(static, X_static[test_idx]), _probability(temporal, X_temporal[test_idx])


def run_nested(df: pd.DataFrame, seed: int = 20260917) -> dict:
    variants = feature_variants()
    needed = sorted(set(variants["median_iqr"]) | {
        feature for variant, _ in TEMPORAL_ARMS for feature in variants[variant]
    })
    frame = df.dropna(subset=needed + ["kernel_family", "phase_label_train"]).reset_index(drop=True)
    scores, selections = {}, {}
    for outer_number, (train_idx, test_idx, fold) in enumerate(_folds(frame, "mixed-families", seed), start=1):
        train = frame.iloc[train_idx].reset_index(drop=True)
        candidates = []
        for temporal_variant, temporal_model in TEMPORAL_ARMS:
            inner_predictions = [
                _fit_pair(train, a, b, temporal_variant, temporal_model, seed + outer_number)
                for a, b, _ in _folds(train, "mixed-families", seed + outer_number)
            ]
            for static_weight in WEIGHTS:
                fold_scores = []
                for y, static_p, temporal_p in inner_predictions:
                    probability = static_weight * static_p + (1.0 - static_weight) * temporal_p
                    fold_scores.append(f1_score(y, probability >= 0.5, labels=[False, True], average="macro", zero_division=0))
                candidates.append((float(np.mean(fold_scores)), temporal_variant, temporal_model, static_weight))
        inner_f1, temporal_variant, temporal_model, static_weight = max(
            candidates, key=lambda item: (item[0], item[1], item[2], item[3])
        )
        y, static_p, temporal_p = _fit_pair(frame, train_idx, test_idx, temporal_variant, temporal_model, seed)
        prediction = static_weight * static_p + (1.0 - static_weight) * temporal_p >= 0.5
        scores[fold] = float(f1_score(y, prediction, labels=[False, True], average="macro", zero_division=0))
        selections[fold] = {
            "static_model": "regresion_log+median_iqr", "temporal_variant": temporal_variant,
            "temporal_model": temporal_model, "static_weight": static_weight,
            "inner_f1_macro_mean": inner_f1,
        }
    summary = protocol.fold_summary(scores)
    return {
        "schema": "historical_gpu_temporal_ensemble_nested/1", "n_runs": len(frame),
        "n_families": int(frame.kernel_family.nunique()), "outer_f1_macro_mean": summary["mean"],
        "outer_f1_macro_std": summary["std"], "outer_f1_macro_worst_fold": summary["min"],
        "worst_fold": summary["worst_kernel"], "per_outer_fold": scores,
        "selection_by_outer_fold": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_nested(add_physical_features(pd.read_csv(args.dataset, low_memory=False)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"F1 macro externo={result['outer_f1_macro_mean']:.3f}; resultado en {args.output}")


if __name__ == "__main__":
    main()
