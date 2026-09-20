"""Ablaciones CPU reproducibles sin fuga Roofline.

Compara variantes de señales PMU disponibles en producción con validación
leave-one-family-out. Las filas se submuestrean de forma estratificada solo
dentro del entrenamiento/diagnóstico para limitar coste, nunca se mezclan
familias entre lados de un pliegue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from fase2_clasificador.analysis.cpu_feature_strategies import (
    FORBIDDEN_FEATURES, add_online_physical_features, production_variants,
)
from fase2_clasificador.eval import protocol
from fase2_clasificador.training import model_specs


def _prepare(frame: pd.DataFrame, max_per_family_class: int, seed: int) -> pd.DataFrame:
    required = {"kernel_ref", "phase_label_train"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"faltan columnas de evaluación: {sorted(missing)}")
    out = add_online_physical_features(frame)
    out["kernel_family"] = out["kernel_ref"].map(protocol.derive_kernel_family)
    out = out[out["phase_label_train"].isin(["compute_bound", "memory_bound"])].copy()
    out["_class"] = out["phase_label_train"].eq("memory_bound")
    # El muestreo conserva soporte de ambas clases/familias y evita que las
    # cargas largas decidan por volumen la selección de representación.
    if max_per_family_class > 0:
        # No usar groupby.apply(): pandas puede omitir las columnas de grupo
        # en versiones recientes, precisamente las que el protocolo necesita
        # después para comprobar que no hay fuga entre familias.
        rng = np.random.default_rng(seed)
        out["_sample_order"] = rng.random(len(out))
        out["_sample_rank"] = out.groupby(["kernel_family", "_class"], sort=True)["_sample_order"].rank(method="first")
        out = out[out["_sample_rank"] <= max_per_family_class].drop(columns=["_sample_order", "_sample_rank"])
    return out.reset_index(drop=True)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    """Peso igual para cada celda familia×clase dentro del entrenamiento."""
    cells = frame.groupby(["kernel_family", "_class"], sort=False).size()
    weight = frame.apply(lambda row: 1.0 / cells[(row["kernel_family"], row["_class"])], axis=1)
    values = weight.to_numpy(dtype=float)
    return values / values.mean()


def _fit(model, X: np.ndarray, y: np.ndarray, weight: np.ndarray | None) -> None:
    if weight is None:
        model.fit(X, y)
    elif hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": weight})
    else:
        model.fit(X, y, sample_weight=weight)


def _prototype(name: str, seed: int, train_y: np.ndarray, n_jobs: int,
               scale_pos_weight: float | None = None):
    positive = int(train_y.sum())
    negative = int(len(train_y) - positive)
    # XGBoost no acepta ``class_weight``. Si se le pasan sample weights que
    # ya equilibran cada celda familia×clase, su scale_pos_weight debe ser 1;
    # aplicar también negative/positive corregiría la clase dos veces.
    if scale_pos_weight is None:
        scale_pos_weight = negative / positive if positive else 1.0
    model = clone(model_specs.build_models(seed, scale_pos_weight)[name])
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=n_jobs)
    return model


def evaluate_lofo(frame: pd.DataFrame, features: list[str], models: list[str], seed: int,
                  family_balanced: bool, n_jobs: int) -> list[dict[str, object]]:
    leak = set(features) & FORBIDDEN_FEATURES
    if leak:
        raise ValueError(f"variante con fuga/no desplegable: {sorted(leak)}")
    clean = frame.dropna(subset=features).reset_index(drop=True)
    X = clean[features].to_numpy(dtype=np.float32)
    y = clean["_class"].to_numpy(dtype=bool)
    scores = {name: {} for name in models}
    pooled = {name: np.zeros((2, 2), dtype=int) for name in models}
    for train, test, fold in protocol.leave_one_kernel_out(clean, kernel_col="kernel_family"):
        protocol.assert_no_familia_leak(clean, train, test, kernel_col="kernel_family", family_fn=lambda x: x)
        weights = _weights(clean.iloc[train]) if family_balanced else None
        for name in models:
            model = _prototype(name, seed, y[train], n_jobs, 1.0 if family_balanced else None)
            _fit(model, X[train], y[train], weights)
            prediction = model.predict(X[test])
            scores[name][fold] = float(f1_score(y[test], prediction, labels=[False, True], average="macro", zero_division=0))
            pooled[name] += confusion_matrix(y[test], prediction, labels=[False, True])
    rows = []
    for name, per_fold in scores.items():
        summary = protocol.fold_summary(per_fold)
        matrix = pooled[name]
        truth = np.repeat([False, True], matrix.sum(axis=1))
        prediction = np.concatenate([
            np.repeat(False, matrix[0, 0]), np.repeat(True, matrix[0, 1]),
            np.repeat(False, matrix[1, 0]), np.repeat(True, matrix[1, 1]),
        ])
        rows.append({
            "model": name, "n_rows": int(len(clean)), "n_families": int(clean.kernel_family.nunique()),
            "features": features, "family_balanced_training": family_balanced,
            "f1_macro_fold_mean": summary["mean"], "f1_macro_fold_std": summary["std"],
            "f1_macro_worst_fold": summary["min"], "worst_fold": summary["worst_kernel"],
            "f1_macro_oof_pooled": float(f1_score(truth, prediction, labels=[False, True], average="macro", zero_division=0)),
            "oof_accuracy": float(accuracy_score(truth, prediction)), "confusion_matrix": matrix.tolist(),
            "per_fold": per_fold,
        })
    return rows


def random_split_diagnostic(frame: pd.DataFrame, features: list[str], seed: int, n_jobs: int) -> float:
    """Diagnóstico de interpolación, explícitamente no elegible para selección."""
    clean = frame.dropna(subset=features).reset_index(drop=True)
    X = clean[features].to_numpy(dtype=np.float32)
    y = clean["_class"].to_numpy(dtype=bool)
    train, test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y))
    model = _prototype("arbol_prof6", seed, y[train], n_jobs)
    _fit(model, X[train], y[train], None)
    return float(f1_score(y[test], model.predict(X[test]), labels=[False, True], average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-per-family-class", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--models", default="mayoritaria,arbol_prof1,regresion_log,arbol_prof6,random_forest,extra_trees,xgboost")
    args = parser.parse_args()
    source = pd.read_csv(args.dataset, low_memory=False)
    frame = _prepare(source, args.max_per_family_class, args.seed)
    variants = production_variants()
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(names) - set(model_specs.build_models(args.seed))
    if unknown:
        raise ValueError(f"modelos desconocidos: {sorted(unknown)}")
    results = []
    for variant, features in variants.items():
        for balanced in (False, True):
            for row in evaluate_lofo(frame, features, names, args.seed, balanced, args.n_jobs):
                row["variant"] = variant
                results.append(row)
    diagnostics = {name: random_split_diagnostic(frame, features, args.seed, args.n_jobs) for name, features in variants.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: v for k, v in row.items() if k not in {"features", "per_fold", "confusion_matrix"}} for row in results])
    summary.sort_values(["f1_macro_fold_mean", "f1_macro_oof_pooled"], ascending=False).to_csv(args.output_dir / "cpu_feature_strategy_summary.csv", index=False)
    (args.output_dir / "cpu_feature_strategy_results.json").write_text(json.dumps({
        "schema": "cpu_feature_strategy_evaluation/1", "source": str(args.dataset),
        "selection_protocol": "LOFO por familia; split aleatorio solo diagnóstico", "results": results,
        "random_split_diagnostic_f1_macro": diagnostics,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"rows={len(frame)} families={frame.kernel_family.nunique()} comparisons={len(results)}")


if __name__ == "__main__":
    main()
