"""Evalúa clasificación selectiva GPU: etiqueta solo predicciones confiables.

El modelo sigue prediciendo únicamente ``compute_bound`` o ``memory_bound``.
La tercera salida operativa, ``revisar``, se produce cuando la probabilidad no
supera un umbral elegido exclusivamente dentro del entrenamiento de cada fold.
No se debe comparar su F1 selectivo con un F1 de cobertura total sin reportar
también la cobertura alcanzada.
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

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import _family_weights, _folds
from fase2_clasificador.eval import protocol

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)

FEATURES = [
    "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median",
    "gpu_util_pct_iqr", "gpu_mem_util_pct_iqr", "gpu_power_mw_iqr", "gpu_sm_clock_mhz_iqr",
]
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=20260917,
    ))


def _prediction(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = frame[FEATURES].to_numpy(dtype=np.float32)
    y = (frame["phase_label_train"] == "memory_bound").to_numpy()
    model = _model()
    model.fit(X[train_idx], y[train_idx], logisticregression__sample_weight=_family_weights(frame.iloc[train_idx]))
    probabilities = model.predict_proba(X[test_idx])
    return y[test_idx], probabilities[:, 1]


def _threshold_score(y: np.ndarray, probability: np.ndarray, threshold: float) -> tuple[float, float]:
    confidence = np.maximum(probability, 1.0 - probability)
    selected = confidence >= threshold
    coverage = float(selected.mean())
    if not selected.any():
        return 0.0, coverage
    prediction = probability[selected] >= 0.5
    return float(f1_score(y[selected], prediction, labels=[False, True], average="macro", zero_division=0)), coverage


def run_nested(df: pd.DataFrame, seed: int = 20260917, min_coverage: float = 0.60,
               scheme: str = "lofo") -> dict:
    frame = df.dropna(subset=FEATURES + ["kernel_family", "phase_label_train"]).reset_index(drop=True)
    fold_results: dict[str, dict[str, object]] = {}
    total_selected = total_correct = total = 0
    oof_truth: list[np.ndarray] = []
    oof_prediction: list[np.ndarray] = []
    per_fold_f1: dict[str, float] = {}
    for outer_number, (train_idx, test_idx, fold) in enumerate(_folds(frame, scheme, seed), start=1):
        train = frame.iloc[train_idx].reset_index(drop=True)
        candidates = []
        for threshold in THRESHOLDS:
            inner = []
            for a, b, _ in _folds(train, scheme, seed + outer_number):
                y_inner, p_inner = _prediction(train, a, b)
                inner.append(_threshold_score(y_inner, p_inner, threshold))
            mean_f1 = float(np.mean([x[0] for x in inner]))
            mean_coverage = float(np.mean([x[1] for x in inner]))
            if mean_coverage >= min_coverage:
                candidates.append((mean_f1, mean_coverage, threshold))
        # Si el objetivo mínimo no es viable, conserva máxima cobertura y deja
        # explícito el incumplimiento en el reporte.
        if not candidates:
            candidates = [(float(np.mean([_threshold_score(*_prediction(train, a, b), t)[0]
                                         for a, b, _ in _folds(train, "mixed-families", seed + outer_number)])),
                           0.0, 0.50) for t in (0.50,)]
        _, inner_coverage, threshold = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
        y_test, p_test = _prediction(frame, train_idx, test_idx)
        confidence = np.maximum(p_test, 1.0 - p_test)
        selected = confidence >= threshold
        prediction = p_test >= 0.5
        coverage = float(selected.mean())
        f1 = float(f1_score(y_test[selected], prediction[selected], labels=[False, True], average="macro", zero_division=0)) if selected.any() else 0.0
        accuracy = float(accuracy_score(y_test[selected], prediction[selected])) if selected.any() else 0.0
        total += len(y_test)
        total_selected += int(selected.sum())
        total_correct += int((prediction[selected] == y_test[selected]).sum())
        oof_truth.append(y_test[selected])
        oof_prediction.append(prediction[selected])
        per_fold_f1[fold] = f1
        fold_results[fold] = {
            "threshold": threshold, "inner_f1_macro_mean": max(candidates)[0],
            "inner_coverage_mean": inner_coverage, "test_coverage": coverage,
            "test_selected": int(selected.sum()), "test_total": int(len(selected)),
            "test_f1_macro_selected": f1, "test_accuracy_selected": accuracy,
        }
    summary = protocol.fold_summary(per_fold_f1)
    return {
        "schema": "historical_gpu_selective_evaluation/1", "model": "logistic_l2_C1_median_iqr_family_balanced",
        "decision": "compute_bound|memory_bound when confidence >= threshold; otherwise revisar",
        "minimum_inner_coverage_target": min_coverage, "n_runs": len(frame),
        "n_families": int(frame.kernel_family.nunique()), "cv_scheme": scheme,
        "selected_coverage_micro": total_selected / total,
        "selected_accuracy_micro": total_correct / total_selected if total_selected else 0.0,
        "selected_f1_macro_oof_pooled": float(f1_score(
            np.concatenate(oof_truth), np.concatenate(oof_prediction), labels=[False, True],
            average="macro", zero_division=0,
        )) if total_selected else 0.0,
        "selected_f1_macro_fold_mean": summary["mean"], "selected_f1_macro_fold_std": summary["std"],
        "selected_f1_macro_worst_fold": summary["min"], "worst_fold": summary["worst_kernel"],
        "per_outer_fold": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--scheme", choices=("lofo", "mixed-families"), default="lofo")
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        raise ValueError("--min-coverage debe estar entre 0 y 1")
    df = pd.read_csv(args.dataset, low_memory=False)
    df = df[~df["kernel_family"].isin(args.exclude_family)].copy()
    result = run_nested(df, min_coverage=args.min_coverage, scheme=args.scheme)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Cobertura={result['selected_coverage_micro']:.3f}; F1 selectivo agrupado={result['selected_f1_macro_oof_pooled']:.3f}")


if __name__ == "__main__":
    main()
