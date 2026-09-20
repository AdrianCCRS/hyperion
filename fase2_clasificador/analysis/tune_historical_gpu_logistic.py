"""Búsqueda anidada y acotada de regresión logística para GPU histórico.

No selecciona C, penalización ni representación mirando las familias del
fold externo. Está destinado a la ruta histórica por corrida, no a CUPTI.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import (
    add_physical_features, feature_variants, _family_weights, _folds,
)
from fase2_clasificador.eval import protocol


# scikit-learn 1.8 anuncia un cambio futuro de ``penalty`` aunque liblinear
# todavía requiere el parámetro en las versiones soportadas por el proyecto.
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", category=UserWarning)


def _model(c: float, penalty: str):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, penalty=penalty, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=20260917,
    ))


def _score(frame: pd.DataFrame, features: list[str], train_idx, test_idx, c: float, penalty: str) -> float:
    X = frame[features].to_numpy(dtype=np.float32)
    y = (frame["phase_label_train"] == "memory_bound").to_numpy()
    model = _model(c, penalty)
    model.fit(X[train_idx], y[train_idx], logisticregression__sample_weight=_family_weights(frame.iloc[train_idx]))
    return float(f1_score(y[test_idx], model.predict(X[test_idx]), labels=[False, True], average="macro", zero_division=0))


def run_nested(df: pd.DataFrame, seed: int = 20260917) -> dict:
    variants = {name: values for name, values in feature_variants().items() if set(values).issubset(df.columns)}
    needed = sorted({col for values in variants.values() for col in values})
    frame = df.dropna(subset=needed + ["kernel_family", "phase_label_train"]).reset_index(drop=True)
    grid = [(c, penalty) for c in (0.01, 0.1, 1.0, 10.0, 100.0) for penalty in ("l1", "l2")]
    outer_scores, choices = {}, {}
    for outer_i, (train_idx, test_idx, fold) in enumerate(_folds(frame, "mixed-families", seed), start=1):
        train = frame.iloc[train_idx].reset_index(drop=True)
        candidates = []
        for name, features in variants.items():
            for c, penalty in grid:
                inner = [_score(train, features, a, b, c, penalty)
                         for a, b, _ in _folds(train, "mixed-families", seed + outer_i)]
                candidates.append((float(np.mean(inner)), name, c, penalty))
        score, variant, c, penalty = max(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
        outer_scores[fold] = _score(frame, variants[variant], train_idx, test_idx, c, penalty)
        choices[fold] = {"aggregation_variant": variant, "C": c, "penalty": penalty, "inner_f1_macro_mean": score}
    summary = protocol.fold_summary(outer_scores)
    return {"schema": "historical_gpu_logistic_nested_tuning/1", "n_runs": len(frame),
            "n_families": int(frame.kernel_family.nunique()), "outer_f1_macro_mean": summary["mean"],
            "outer_f1_macro_std": summary["std"], "outer_f1_macro_worst_fold": summary["min"],
            "worst_fold": summary["worst_kernel"], "per_outer_fold": outer_scores, "selection_by_outer_fold": choices,
            "grid": [{"C": c, "penalty": p} for c, p in grid]}


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
