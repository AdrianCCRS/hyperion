"""Valida un clasificador GPU entrenado en una campaña contra otra campaña."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import _family_weights, _fit, add_physical_features, feature_variants
from fase2_clasificador.training import model_specs


def evaluate(train: pd.DataFrame, test: pd.DataFrame, variant: str, model_name: str) -> dict:
    features = feature_variants()[variant]
    train = train.dropna(subset=features + ["kernel_family", "phase_label_train"]).reset_index(drop=True)
    test = test.dropna(subset=features + ["kernel_family", "phase_label_train"]).reset_index(drop=True)
    y_train = train["phase_label_train"].eq("memory_bound").to_numpy()
    y_test = test["phase_label_train"].eq("memory_bound").to_numpy()
    model = clone(model_specs.build_models(20260917)[model_name])
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=1)
    _fit(model, train[features].to_numpy(np.float32), y_train, _family_weights(train))
    prediction = model.predict(test[features].to_numpy(np.float32))
    per_family = {}
    for family, group in test.assign(_prediction=prediction).groupby("kernel_family"):
        truth = group["phase_label_train"].eq("memory_bound")
        per_family[family] = float(accuracy_score(truth, group["_prediction"]))
    return {
        "schema": "historical_gpu_external_evaluation/1", "variant": variant, "model": model_name,
        "n_train_runs": len(train), "n_train_families": int(train.kernel_family.nunique()),
        "n_test_runs": len(test), "n_test_families": int(test.kernel_family.nunique()),
        "f1_macro_pooled": float(f1_score(y_test, prediction, labels=[False, True], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "confusion_matrix_compute_memory": confusion_matrix(y_test, prediction, labels=[False, True]).tolist(),
        "family_accuracy_mean": float(np.mean(list(per_family.values()))),
        "family_accuracy_worst": float(np.min(list(per_family.values()))),
        "worst_accuracy_family": min(per_family, key=per_family.get), "per_family_accuracy": per_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=feature_variants().keys())
    parser.add_argument("--model", required=True, choices=model_specs.build_models(20260917).keys())
    parser.add_argument("--exclude-test-family", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train = add_physical_features(pd.read_csv(args.train, low_memory=False))
    test = add_physical_features(pd.read_csv(args.test, low_memory=False))
    test = test[~test["kernel_family"].isin(args.exclude_test_family)].copy()
    result = evaluate(train, test, args.variant, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"F1 macro externo={result['f1_macro_pooled']:.3f}; accuracy por familia={result['family_accuracy_mean']:.3f}")


if __name__ == "__main__":
    main()
