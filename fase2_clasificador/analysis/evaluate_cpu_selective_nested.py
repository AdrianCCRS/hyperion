"""Validación CPU LOFO anidada: representación y abstención sin fuga.

La selección de variante PMU y del umbral de abstención se hace solamente
dentro del entrenamiento de cada familia externa. ``revisar`` no es una clase
del modelo: es una decisión operativa cuando ``max(p, 1-p)`` no alcanza el
umbral elegido. Por eso siempre se informa junto con la cobertura.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fase2_clasificador.analysis.cpu_feature_strategies import production_variants
from fase2_clasificador.analysis.evaluate_cpu_feature_strategies import (
    _fit, _prepare, _prototype, _weights,
)
from fase2_clasificador.eval import protocol


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98)


def _cell_coverage_and_balance(y: np.ndarray, prediction: np.ndarray, selected: np.ndarray,
                               family: np.ndarray) -> tuple[float, float]:
    """Exactitud balanceada por celda familia x clase sobre lo decidido y cobertura media por celda.

    Cada celda observada pesa lo mismo, sin importar cuantos intervalos aporte; una celda sin ningun
    intervalo decidido no tiene recall y se omite de la exactitud, pero cuenta con cobertura cero.
    """
    recalls: list[float] = []
    coverages: list[float] = []
    for name in np.unique(family):
        in_family = family == name
        for cls in (False, True):
            cell = in_family & (y == cls)
            if not cell.any():
                continue
            coverages.append(float(selected[cell].mean()))
            decided = cell & selected
            if decided.any():
                recalls.append(float((prediction[decided] == cls).mean()))
    return (float(np.mean(recalls)) if recalls else 0.0), float(np.mean(coverages))


def _score(y: np.ndarray, probability: np.ndarray, threshold: float,
           family: np.ndarray | None = None) -> tuple[float, float]:
    """(exactitud balanceada por celda sobre lo decidido, cobertura media por celda)."""
    selected = np.maximum(probability, 1.0 - probability) >= threshold
    labels = np.zeros(len(y), dtype=int) if family is None else np.asarray(family)
    return _cell_coverage_and_balance(y.astype(bool), probability >= 0.5, selected, labels)


def _probabilities(frame: pd.DataFrame, features: list[str], train_idx: np.ndarray,
                   test_idx: np.ndarray, seed: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame["_class"].to_numpy(dtype=bool)
    # _weights ya distribuye masa igual por familia×clase. Mantener el
    # scale_pos_weight propio de XGBoost además de esos pesos duplicaría el
    # balance de clase.
    model = _prototype("xgboost", seed, y[train_idx], n_jobs, scale_pos_weight=1.0)
    _fit(model, X[train_idx], y[train_idx], _weights(frame.iloc[train_idx]))
    return y[test_idx], model.predict_proba(X[test_idx])[:, 1]


def _inner_choice(train: pd.DataFrame, variants: dict[str, list[str]], seed: int,
                  n_jobs: int, min_coverage: float) -> tuple[str, float, float, float]:
    """Elige (variante, umbral) por LOFO interno; devuelve (variante, umbral, exactitud por celda, cobertura).

    El criterio es la exactitud balanceada por celda familia x clase sobre lo decidido, calculada con las
    predicciones fuera de muestra de todos los pliegues internos, sujeta a una cobertura media por celda
    minima. Es la misma metrica principal con la que se reporta el resultado.
    """
    candidates: list[tuple[float, float, float, str]] = []
    for variant, features in variants.items():
        ys, ps, fs = [], [], []
        for number, (a, b, name) in enumerate(protocol.leave_one_kernel_out(train, kernel_col="kernel_family"), start=1):
            y, p = _probabilities(train, features, a, b, seed + number, n_jobs)
            ys.append(y.astype(bool)); ps.append(p); fs.append(np.full(len(y), name))
        y_all, p_all, f_all = np.concatenate(ys), np.concatenate(ps), np.concatenate(fs)
        for threshold in THRESHOLDS:
            score, coverage = _score(y_all, p_all, threshold, f_all)
            if coverage >= min_coverage:
                candidates.append((score, coverage, threshold, variant))
    if not candidates:
        # El umbral 0.50 siempre cubre todo; este guardarraíl hace explícito
        # que no se alcanzó el objetivo en vez de inventar un umbral externo.
        raise RuntimeError("ninguna variante alcanzó la cobertura interna mínima")
    score, coverage, threshold, variant = max(candidates, key=lambda row: (row[0], row[1], -row[2], row[3]))
    return variant, threshold, score, coverage


def run_nested(frame: pd.DataFrame, variants: dict[str, list[str]], seed: int,
               n_jobs: int, min_coverage: float) -> dict[str, object]:
    features_needed = sorted({feature for value in variants.values() for feature in value})
    clean = frame.dropna(subset=features_needed).reset_index(drop=True)
    fold_results: dict[str, dict[str, object]] = {}
    selected_truth: list[np.ndarray] = []
    selected_prediction: list[np.ndarray] = []
    fold_f1: dict[str, float] = {}
    total = total_selected = total_correct = 0
    for number, (train_idx, test_idx, family) in enumerate(
        protocol.leave_one_kernel_out(clean, kernel_col="kernel_family"), start=1
    ):
        protocol.assert_no_familia_leak(clean, train_idx, test_idx, kernel_col="kernel_family", family_fn=lambda value: value)
        train = clean.iloc[train_idx].reset_index(drop=True)
        variant, threshold, inner_score, inner_coverage = _inner_choice(
            train, variants, seed + number * 1000, n_jobs, min_coverage
        )
        y, probability = _probabilities(clean, variants[variant], train_idx, test_idx, seed, n_jobs)
        prediction = probability >= 0.5
        selected = np.maximum(probability, 1.0 - probability) >= threshold
        coverage = float(selected.mean())
        f1 = float(f1_score(y[selected], prediction[selected], labels=[False, True], average="macro", zero_division=0)) if selected.any() else 0.0
        accuracy = float(accuracy_score(y[selected], prediction[selected])) if selected.any() else 0.0
        selected_truth.append(y[selected])
        selected_prediction.append(prediction[selected])
        total += len(y)
        total_selected += int(selected.sum())
        total_correct += int((prediction[selected] == y[selected]).sum())
        fold_f1[family] = f1
        fold_results[family] = {
            "selected_variant": variant, "threshold": threshold,
            "inner_cell_balanced_accuracy": inner_score, "inner_coverage_mean": inner_coverage,
            "test_coverage": coverage, "test_selected": int(selected.sum()), "test_total": int(len(selected)),
            "test_f1_macro_selected": f1, "test_accuracy_selected": accuracy,
        }
    summary = protocol.fold_summary(fold_f1)
    return {
        "schema": "cpu_selective_nested_lofo/1",
        "model_candidates": {name: {"model": "xgboost", "features": features} for name, features in variants.items()},
        "decision": "compute_bound|memory_bound when confidence >= threshold; otherwise revisar",
        "selection_protocol": "LOFO externo; variante y umbral seleccionados por LOFO interno del entrenamiento externo",
        "minimum_inner_coverage_target": min_coverage, "threshold_candidates": list(THRESHOLDS),
        "n_rows": int(len(clean)), "n_families": int(clean.kernel_family.nunique()),
        "selected_coverage_micro": total_selected / total,
        "selected_accuracy_micro": total_correct / total_selected if total_selected else 0.0,
        "selected_f1_macro_oof_pooled": float(f1_score(np.concatenate(selected_truth), np.concatenate(selected_prediction), labels=[False, True], average="macro", zero_division=0)) if total_selected else 0.0,
        "selected_f1_macro_fold_mean": summary["mean"], "selected_f1_macro_fold_std": summary["std"],
        "selected_f1_macro_worst_fold": summary["min"], "worst_fold": summary["worst_kernel"],
        "per_outer_fold": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-per-family-class", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--variants", default="baseline_pmu,pmu_interactions",
                        help="variantes que compiten en el LOFO interno; la final es baseline_pmu (6 tasas base)")
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        raise ValueError("--min-coverage debe estar entre 0 y 1")
    all_variants = production_variants()
    requested = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = set(requested) - set(all_variants)
    if unknown:
        raise ValueError(f"variantes desconocidas: {sorted(unknown)}")
    source = pd.read_csv(args.dataset, low_memory=False)
    result = run_nested(_prepare(source, args.max_per_family_class, args.seed),
                        {name: all_variants[name] for name in requested}, args.seed, args.n_jobs, args.min_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"cobertura={result['selected_coverage_micro']:.3f}; F1 selectivo={result['selected_f1_macro_oof_pooled']:.3f}")


if __name__ == "__main__":
    main()
