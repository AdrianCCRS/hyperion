"""Evalúa una regresión logística con reponderación al peor caso por familia."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fase2_clasificador.eval import protocol

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
FEATURES = [
    "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median",
    "gpu_util_pct_iqr", "gpu_mem_util_pct_iqr", "gpu_power_mw_iqr", "gpu_sm_clock_mhz_iqr",
]


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=20260917,
    ))


def _fit_group_robust(X: np.ndarray, y: np.ndarray, families: np.ndarray, rounds: int, eta: float):
    names, inverse = np.unique(families, return_inverse=True)
    q = np.full(len(names), 1.0 / len(names))
    counts = np.bincount(inverse)
    for _ in range(rounds):
        weights = q[inverse] / counts[inverse]
        weights /= weights.mean()
        model = _model()
        model.fit(X, y, logisticregression__sample_weight=weights)
        probability = model.predict_proba(X)[:, 1]
        losses = np.array([log_loss(y[inverse == group], probability[inverse == group], labels=[False, True])
                           for group in range(len(names))])
        q *= np.exp(eta * (losses - losses.mean()))
        q /= q.sum()
    return model, dict(zip(names.tolist(), q.tolist()))


def run(frame: pd.DataFrame, rounds: int, eta: float) -> dict[str, object]:
    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame["phase_label_train"].eq("memory_bound").to_numpy()
    families = frame["kernel_family"].to_numpy()
    prediction = np.empty(len(frame), dtype=bool)
    fold_weights: dict[str, dict[str, float]] = {}
    for train, test, fold in protocol.leave_one_kernel_out(frame, kernel_col="kernel_family"):
        model, weights = _fit_group_robust(X[train], y[train], families[train], rounds, eta)
        prediction[test] = model.predict(X[test])
        fold_weights[fold] = weights
    family_accuracy = pd.DataFrame({"family": families, "correct": prediction == y}).groupby("family")["correct"].mean()
    return {
        "schema": "historical_gpu_group_robust/1", "n_runs": int(len(frame)),
        "rounds": rounds, "eta": eta,
        "f1_macro_oof_pooled": float(f1_score(y, prediction, labels=[False, True], average="macro", zero_division=0)),
        "accuracy_oof": float(accuracy_score(y, prediction)),
        "family_accuracy_mean": float(family_accuracy.mean()), "family_accuracy_worst": float(family_accuracy.min()),
        "worst_family": str(family_accuracy.idxmin()), "final_group_weights_by_outer_fold": fold_weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--eta", type=float, default=0.5)
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset, low_memory=False)
    frame = frame[~frame["kernel_family"].isin(args.exclude_family)].dropna(
        subset=FEATURES + ["kernel_family", "phase_label_train"]
    ).reset_index(drop=True)
    result = run(frame, args.rounds, args.eta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"F1={result['f1_macro_oof_pooled']:.3f}; peor familia={result['family_accuracy_worst']:.3f}")


if __name__ == "__main__":
    main()
