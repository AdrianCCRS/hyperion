"""Interfaz uniforme de modelos candidatos del selector."""
from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np
import pandas as pd


BASE_CATEGORICAL_FEATURES = (
    "operation", "family", "candidate_device", "resource_state",
)
BASE_NUMERIC_FEATURES = (
    "log10_n", "flops_per_dispatch_analytic", "log10_flops_per_dispatch",
    "logical_bytes_per_dispatch", "log10_logical_bytes",
    "arithmetic_intensity_analytic", "candidate_cpu_fraction",
    "candidate_gpu_fraction", "candidate_cpu_is_ref", "candidate_gpu_is_ref",
    "requires_cold_start",
)
FORBIDDEN_MODEL_FEATURES = {
    "is_optimal", "edp_mean", "edp_std", "edp_min", "edp_max",
    "energy_mean", "energy_std", "time_mean", "time_std", "iterations",
    "margin_edp_pct", "n_repetitions", "action_id", "config_id",
    "decision_group_id", "target_region",
}


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in BASE_CATEGORICAL_FEATURES if column in frame]
    if "probe_device" in frame:
        categorical.append("probe_device")
    numeric = [column for column in BASE_NUMERIC_FEATURES if column in frame]
    numeric.extend(
        column for column in frame.columns
        if column.startswith("probe_")
        and column != "probe_device"
        and pd.api.types.is_numeric_dtype(frame[column])
    )
    leaked = (set(categorical) | set(numeric)) & FORBIDDEN_MODEL_FEATURES
    if leaked:
        raise AssertionError(f"fuga de objetivo en features: {sorted(leaked)}")
    if not categorical and not numeric:
        raise ValueError("dataset sin features permitidas")
    return sorted(set(categorical)), sorted(set(numeric))


def _preprocessor(categorical: list[str], numeric: list[str], *, scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
    )


def suggest_parameters(trial: Any, family: str) -> dict[str, Any]:
    if family == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        }
    if family == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 12),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5]),
        }
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
    raise ValueError(f"familia desconocida: {family}")


def default_parameters(family: str) -> dict[str, Any]:
    defaults = {
        "logistic": {"C": 1.0, "penalty": "l2", "tol": 1e-4},
        "decision_tree": {"max_depth": 5, "criterion": "gini", "min_samples_split": 2,
                          "min_samples_leaf": 1, "max_features": None},
        "random_forest": {"n_estimators": 100, "max_depth": 12, "min_samples_split": 2,
                          "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True},
        "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                    "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
                    "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0},
    }
    if family not in defaults:
        raise ValueError(f"familia desconocida: {family}")
    return dict(defaults[family])


def build_pipeline(
    family: str,
    params: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    seed: int,
    scale_pos_weight: float | None = None,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    categorical, numeric = feature_columns(frame)
    if family == "logistic":
        logistic_params = dict(params)
        penalty = logistic_params.pop("penalty", "l2")
        estimator = LogisticRegression(
            **logistic_params, solver="liblinear", class_weight="balanced",
            l1_ratio=1.0 if penalty == "l1" else 0.0,
            max_iter=3000, random_state=seed,
        )
        scale = True
    elif family == "decision_tree":
        estimator = DecisionTreeClassifier(
            **params, class_weight="balanced", random_state=seed,
        )
        scale = False
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            **params, class_weight="balanced_subsample", random_state=seed,
            n_jobs=1,
        )
        scale = False
    elif family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("XGBoost no esta instalado; instale classifier/requirements.txt") from error
        estimator = XGBClassifier(
            **params, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cpu", n_jobs=1, random_state=seed,
            scale_pos_weight=1.0 if scale_pos_weight is None else scale_pos_weight,
        )
        scale = False
    else:
        raise ValueError(f"familia desconocida: {family}")
    return Pipeline([
        ("preprocessor", _preprocessor(categorical, numeric, scale=scale)),
        ("model", estimator),
    ])


def positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1:
        raise ValueError(f"modelo sin clase positiva unica: {classes}")
    return probabilities[:, positions[0]]


def class_weight_ratio(labels: pd.Series | np.ndarray) -> float:
    values = np.asarray(labels, dtype=int)
    positives = int((values == 1).sum())
    negatives = int((values == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("entrenamiento requiere ambas clases")
    return negatives / positives


def model_tree_count(model: Any) -> int:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "n_estimators"):
        return int(estimator.n_estimators)
    if hasattr(estimator, "tree_"):
        return 1
    return 0
