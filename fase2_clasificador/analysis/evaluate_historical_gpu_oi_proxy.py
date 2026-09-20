"""Clasifica GPU aprendiendo una OI proxy y aplicando el ridge conocido.

La OI y el ridge nunca entran como features. Durante entrenamiento la OI es el
objetivo intermedio; durante inferencia el regresor estima OI desde NVML y la
compara con el ridge de la configuración de hardware. La salida final continúa
siendo exclusivamente ``compute_bound`` o ``memory_bound``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import (
    _family_weights, _folds, add_physical_features, feature_variants,
)
from fase2_clasificador.eval import protocol


def _models(seed: int):
    return {
        **{f"ridge_a{alpha:g}": make_pipeline(StandardScaler(), Ridge(alpha=alpha))
           for alpha in (0.1, 1.0, 10.0, 100.0)},
    }


def _fit_predict(frame: pd.DataFrame, features: list[str], train_idx, test_idx, model_name: str, seed: int):
    X = frame[features].to_numpy(dtype=np.float32)
    log_oi = np.log10(frame["operational_intensity"].to_numpy(dtype=float))
    log_ridge = np.log10(frame["i_ridge_used"].to_numpy(dtype=float))
    truth = (frame["phase_label_train"] == "compute_bound").to_numpy()
    model = clone(_models(seed)[model_name])
    weights = _family_weights(frame.iloc[train_idx])
    if hasattr(model, "steps"):
        model.fit(X[train_idx], log_oi[train_idx], **{f"{model.steps[-1][0]}__sample_weight": weights})
    else:
        model.fit(X[train_idx], log_oi[train_idx], sample_weight=weights)
    predicted_log_oi = model.predict(X[test_idx])
    predicted_class = predicted_log_oi >= log_ridge[test_idx]
    return (
        float(f1_score(truth[test_idx], predicted_class, labels=[False, True], average="macro", zero_division=0)),
        float(mean_absolute_error(log_oi[test_idx], predicted_log_oi)),
    )


def run_nested(df: pd.DataFrame, seed: int = 20260917) -> dict:
    variants_all = feature_variants()
    keep = {"median", "median_iqr", "full_distribution", "physical_ratios", "physics_extended"}
    variants = {name: features for name, features in variants_all.items()
                if name in keep and set(features).issubset(df.columns)}
    required = sorted({x for values in variants.values() for x in values})
    frame = df.dropna(subset=required + ["operational_intensity", "i_ridge_used", "phase_label_train",
                                         "kernel_family"]).reset_index(drop=True)
    if (frame[["operational_intensity", "i_ridge_used"]] <= 0).any().any():
        raise ValueError("OI y ridge deben ser positivos")
    outer_scores, outer_mae, selections = {}, {}, {}
    names = sorted(_models(seed))
    for number, (train_idx, test_idx, fold) in enumerate(_folds(frame, "mixed-families", seed), start=1):
        train = frame.iloc[train_idx].reset_index(drop=True)
        candidates = []
        for variant, features in variants.items():
            for model_name in names:
                inner = [_fit_predict(train, features, a, b, model_name, seed + number)[0]
                         for a, b, _ in _folds(train, "mixed-families", seed + number)]
                candidates.append((float(np.mean(inner)), variant, model_name))
        inner_f1, variant, model_name = max(candidates, key=lambda x: (x[0], x[1], x[2]))
        f1, mae = _fit_predict(frame, variants[variant], train_idx, test_idx, model_name, seed)
        outer_scores[fold], outer_mae[fold] = f1, mae
        selections[fold] = {"feature_variant": variant, "regressor": model_name,
                            "inner_f1_macro_mean": inner_f1}
    summary = protocol.fold_summary(outer_scores)
    return {
        "schema": "historical_gpu_oi_proxy_nested/1", "n_runs": len(frame),
        "n_families": int(frame.kernel_family.nunique()), "outer_f1_macro_mean": summary["mean"],
        "outer_f1_macro_std": summary["std"], "outer_f1_macro_worst_fold": summary["min"],
        "worst_fold": summary["worst_kernel"], "outer_log10_oi_mae_mean": float(np.mean(list(outer_mae.values()))),
        "per_outer_fold": outer_scores, "log10_oi_mae_by_fold": outer_mae,
        "selection_by_outer_fold": selections,
        "inference_rule": "predicted_log10_OI >= log10(known_ridge) => compute_bound",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    args = parser.parse_args()
    df = add_physical_features(pd.read_csv(args.dataset, low_memory=False))
    df = df[~df["kernel_family"].isin(args.exclude_family)].copy()
    result = run_nested(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"F1 macro externo={result['outer_f1_macro_mean']:.3f}; resultado en {args.output}")


if __name__ == "__main__":
    main()
