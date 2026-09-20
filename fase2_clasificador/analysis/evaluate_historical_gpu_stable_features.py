"""Compara normalización por condición y selección estable entre familias.

Todas las decisiones se toman dentro del entrenamiento de cada fold LOFO. La
salida principal concatena predicciones OOF, porque varias familias contienen
una sola clase y el promedio de F1 por fold no es representativo.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import add_physical_features, _family_weights
from fase2_clasificador.eval import protocol

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)

BASE = [
    "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median",
    "gpu_util_pct_iqr", "gpu_mem_util_pct_iqr", "gpu_power_mw_iqr", "gpu_sm_clock_mhz_iqr",
]
CONDITION = BASE + [
    "gpu_temperature_c_median", "gpu_temperature_c_iqr", "gpu_power_mw_per_mhz_median",
    "gpu_mem_util_to_gpu_util", "gpu_power_mw_per_activity_mhz", "gpu_temperature_c_per_power_w",
]


def _model(*, l1: bool = False):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1 if l1 else 1.0, penalty="l1" if l1 else "l2", solver="liblinear",
        class_weight="balanced", max_iter=2000, random_state=20260917,
    ))


def _fit_predict(frame: pd.DataFrame, features: list[str], train: np.ndarray, test: np.ndarray) -> np.ndarray:
    y = frame["phase_label_train"].eq("memory_bound").to_numpy()
    model = _model()
    model.fit(frame.iloc[train][features], y[train],
              logisticregression__sample_weight=_family_weights(frame.iloc[train]))
    return model.predict(frame.iloc[test][features])


def _stable_features(frame: pd.DataFrame, train: np.ndarray, seed: int, min_fraction: float) -> list[str]:
    subset = frame.iloc[train].reset_index(drop=True)
    y = subset["phase_label_train"].eq("memory_bound").to_numpy()
    counts = np.zeros(len(CONDITION), dtype=int)
    folds = list(protocol.leave_one_kernel_out(subset, kernel_col="kernel_family"))
    for inner_train, _, _ in folds:
        model = _model(l1=True)
        model.fit(subset.iloc[inner_train][CONDITION], y[inner_train],
                  logisticregression__sample_weight=_family_weights(subset.iloc[inner_train]))
        counts += np.abs(model.named_steps["logisticregression"].coef_[0]) > 1e-8
    chosen = [feature for feature, count in zip(CONDITION, counts) if count / len(folds) >= min_fraction]
    # Evita una selección vacía o una representación de una sola señal.
    return chosen if len(chosen) >= 2 else BASE


def evaluate(frame: pd.DataFrame, stable_fraction: float) -> dict[str, object]:
    y = frame["phase_label_train"].eq("memory_bound").to_numpy()
    methods = {"baseline_median_iqr": BASE, "condition_normalized": CONDITION, "stable_l1": None}
    results: dict[str, dict[str, object]] = {}
    for method, features in methods.items():
        pred = np.empty(len(frame), dtype=bool)
        selected: dict[str, list[str]] = {}
        for fold_number, (train, test, fold) in enumerate(protocol.leave_one_kernel_out(frame, kernel_col="kernel_family"), start=1):
            use = _stable_features(frame, train, 20260917 + fold_number, stable_fraction) if features is None else features
            pred[test] = _fit_predict(frame, use, train, test)
            selected[fold] = use
        results[method] = {
            "f1_macro_oof_pooled": float(f1_score(y, pred, labels=[False, True], average="macro", zero_division=0)),
            "accuracy_oof": float(accuracy_score(y, pred)),
            "stable_features_by_fold": selected if method == "stable_l1" else None,
        }
    return {"schema": "historical_gpu_stable_features/1", "n_runs": int(len(frame)),
            "n_families": int(frame.kernel_family.nunique()), "stable_fraction": stable_fraction,
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--stable-fraction", type=float, default=0.70)
    args = parser.parse_args()
    if not 0 < args.stable_fraction <= 1:
        raise ValueError("--stable-fraction debe estar entre 0 y 1")
    frame = add_physical_features(pd.read_csv(args.dataset, low_memory=False))
    frame = frame[~frame["kernel_family"].isin(args.exclude_family)].dropna(
        subset=CONDITION + ["kernel_family", "phase_label_train"]
    ).reset_index(drop=True)
    result = evaluate(frame, args.stable_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for method, metrics in result["results"].items():
        print(f"{method}: F1={metrics['f1_macro_oof_pooled']:.3f}; accuracy={metrics['accuracy_oof']:.3f}")


if __name__ == "__main__":
    main()
