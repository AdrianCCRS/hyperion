"""Fase R2: regresores simples sobre el target de horizonte `resource_state x K`.

Implementa las secciones 4 y 5 del protocolo congelado
(`docs/general/protocolo_congelado_confirmatorio_20260830.md`) sobre el target
de la enmienda **2026-08-30-A** (`horizon.build_horizon_dataset`), en lugar
del target de K=1 de `compact.build_compact_dataset`.

Punto central del modulo: el modelo y las baselines de `sizes.BASELINES` se
evaluan sobre **los mismos pliegues** (`sizes.interpolation_folds` /
`extrapolation_folds`) y con la misma funcion de metricas
(`sizes.evaluate_devices`), para que la comparacion sea pareada y la regla
bloqueante de la seccion 6 se pueda aplicar sin ambiguedad.

Diferencia deliberada de diseno entre modelo y baselines: las baselines se
ajustan **por estado** (no reciben `resource_state` como entrada, son reglas
simples por estado); el modelo se entrena **una vez sobre los tres estados
juntos**, porque `resource_state` es la caracteristica 9 de la seccion 3.1 --
la formulacion asume que el modelo aprende la interaccion estado x K, no que
se entrena un modelo distinto por estado.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import math
import pickle
import time

import numpy as np
import pandas as pd

from .compact import NOISE_FLOOR_PCT, CompactDatasetError, assert_no_leakage
from .horizon import HORIZON_FEATURES, K_GRID, horizon_feature_columns
from .sizes import BASELINES, evaluate_devices

#: Objetivo de regresion (seccion 1 del protocolo, extendido por la
#: enmienda 2026-08-30-A a horizonte K). Se conserva la magnitud del error.
TARGET_COLUMN = "y_log_edp_ratio_k"

#: Semilla fija congelada (seccion 5.5 del protocolo).
FROZEN_SEED = 20260830

#: Familias de regresores (seccion 4 del protocolo). Deliberadamente simples.
REGRESSOR_FAMILIES: tuple[str, ...] = ("ridge", "elasticnet", "huber", "tree", "random_forest")

#: Umbral de abstencion congelado (seccion 7): ``|y| < log(1 + piso_de_ruido)``.
ABSTENTION_LOG_THRESHOLD = math.log(1.0 + NOISE_FLOOR_PCT / 100.0)

_CATEGORICAL_BASE = ("operation", "resource_state")
_NUMERIC_BASE = tuple(
    column for column in HORIZON_FEATURES if column not in _CATEGORICAL_BASE
)


class R2ModelError(RuntimeError):
    """La configuracion de entrenamiento/evaluacion de R2 no es valida."""


# --------------------------------------------------------------------------
# Preprocesamiento y familias de modelos (seccion 4)
# --------------------------------------------------------------------------


def _split_feature_types(features: Sequence[str], frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in features if column in _CATEGORICAL_BASE or column == "probe_device"]
    numeric = [column for column in features if column not in categorical]
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise R2ModelError(f"faltan columnas de caracteristicas en el frame: {missing}")
    return categorical, numeric


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


def _base_estimator(family: str, seed: int, params: Mapping[str, Any] | None = None):
    params = dict(params or {})
    if family == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**{"alpha": 1.0, **params})
    if family == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**{"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10_000, "random_state": seed, **params})
    if family == "huber":
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(**{"epsilon": 1.35, "alpha": 1e-3, "max_iter": 2_000, **params})
    if family == "tree":
        from sklearn.tree import DecisionTreeRegressor
        # max_depth <= 3 congelado en la seccion 4 del protocolo.
        return DecisionTreeRegressor(**{"max_depth": 3, "random_state": seed, **params})
    if family == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        # n_estimators=200, max_depth<=5 congelados en la seccion 4.
        return RandomForestRegressor(**{
            "n_estimators": 200, "max_depth": 5, "random_state": seed, "n_jobs": 1, **params,
        })
    raise R2ModelError(f"familia de regresor desconocida: {family}")


#: Rejillas pequenas para la validacion cruzada interna (seccion 5.3). El
#: `max_depth` de `tree`/`random_forest` nunca excede el limite congelado.
_PARAM_GRID: dict[str, dict[str, list[Any]]] = {
    "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
    "elasticnet": {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    # `epsilon` fijo en el valor por defecto de sklearn: solo se ajusta
    # `alpha`. El solver de HuberRegressor (IRLS) es, con margen, el mas caro
    # de las cinco familias sobre este objetivo (rango ~40 en `y`); una
    # rejilla mas ancha no cambia la conclusion con n=68 y multiplica el
    # costo de la CV interna sin aportar precision util.
    "huber": {"model__alpha": [1e-4, 1e-3, 1e-2]},
    "tree": {"model__max_depth": [1, 2, 3]},
    "random_forest": {"model__max_depth": [2, 3, 4, 5]},
}


def build_regressor_pipeline(family: str, features: Sequence[str], frame: pd.DataFrame, *, seed: int):
    from sklearn.pipeline import Pipeline

    categorical, numeric = _split_feature_types(features, frame)
    scale = family in ("ridge", "elasticnet", "huber")
    return Pipeline([
        ("preprocessor", _preprocessor(categorical, numeric, scale=scale)),
        ("model", _base_estimator(family, seed)),
    ])


def fit_tuned_regressor(
    family: str, train: pd.DataFrame, features: Sequence[str], *, seed: int, n_splits: int = 3,
):
    """Ajusta `family` con hiperparametros elegidos por CV agrupada en `config_id`.

    Seccion 5.3: la seleccion de hiperparametros ocurre DENTRO de `train`,
    nunca toca `test`. Si `train` no tiene suficientes `config_id` distintos
    para una particion agrupada valida, se usan los parametros por defecto
    (esto solo ocurre en pruebas unitarias con datos sinteticos minimos; el
    dataset real tiene 68 `config_id`).
    """
    from sklearn.model_selection import GroupKFold, GridSearchCV

    pipeline = build_regressor_pipeline(family, features, train, seed=seed)
    groups = train["config_id"].astype(str).to_numpy()
    n_groups = len(set(groups))
    grid = _PARAM_GRID[family]
    splits = min(n_splits, n_groups)
    if splits < 2:
        pipeline.fit(train[list(features)], train[TARGET_COLUMN].to_numpy(dtype=float))
        return pipeline, {"tuned": False, "reason": "insufficient_groups_for_cv"}
    cv = list(GroupKFold(n_splits=splits).split(train, groups=groups))
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(train[list(features)], train[TARGET_COLUMN].to_numpy(dtype=float))
    return search.best_estimator_, {"tuned": True, "best_params": search.best_params_}


def build_classification_pipeline(features: Sequence[str], frame: pd.DataFrame, *, seed: int):
    """Contraste secundario de clasificacion binaria (seccion 1 del protocolo).

    Deliberadamente un solo modelo simple (regresion logistica): la formulacion
    primaria congelada es la regresion; esto es un contraste, no una segunda
    suite completa de cinco familias.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    categorical, numeric = _split_feature_types(features, frame)
    return Pipeline([
        ("preprocessor", _preprocessor(categorical, numeric, scale=True)),
        ("model", LogisticRegression(
            C=1.0, max_iter=3_000, class_weight="balanced", random_state=seed,
        )),
    ])


# --------------------------------------------------------------------------
# Prediccion de dispositivo con abstencion (seccion 7)
# --------------------------------------------------------------------------


def devices_from_predictions(y_pred: np.ndarray, resource_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(devices, abstained)`` aplicando la regla de abstencion congelada.

    Al abstenerse se aplica la politica segura del estado: permanecer en el
    dispositivo preparado (`gpu_ready` -> gpu; `cpu_ready`/`none_ready` -> cpu).
    La abstencion se calibra con una constante preregistrada (seccion 7), por
    lo que "calibrar solo dentro de entrenamiento" se cumple trivialmente: no
    hay nada que ajustar con datos de prueba.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    state = np.asarray(resource_state, dtype=object)
    abstained = np.abs(y_pred) < ABSTENTION_LOG_THRESHOLD
    raw = np.where(y_pred < 0, "gpu", "cpu")
    safe = np.where(state == "gpu_ready", "gpu", "cpu")
    devices = np.where(abstained, safe, raw)
    return devices, abstained


# --------------------------------------------------------------------------
# Evaluacion por pliegue (seccion 5 / 8)
# --------------------------------------------------------------------------


def _model_rows(
    fold_name: str, regime: str, train: pd.DataFrame, test: pd.DataFrame,
    *, with_probe: bool, seed: int, k_grid: Sequence[int],
) -> list[dict[str, Any]]:
    features = horizon_feature_columns(train, with_probe=with_probe)
    assert_no_leakage(features)
    records: list[dict[str, Any]] = []
    for family in REGRESSOR_FAMILIES:
        model, tuning = fit_tuned_regressor(family, train, features, seed=seed)
        y_pred_all = model.predict(test[features])
        devices_all, abstained_all = devices_from_predictions(
            y_pred_all, test["resource_state"].to_numpy()
        )
        for state in sorted(test["resource_state"].astype(str).unique()):
            state_mask = (test["resource_state"] == state).to_numpy()
            for k in k_grid:
                mask = state_mask & (test["k"] == k).to_numpy()
                if not mask.any():
                    continue
                test_slice = test.loc[mask]
                metrics = evaluate_devices(test_slice, devices_all[mask])
                records.append({
                    "fold": fold_name, "regime": regime, "resource_state": state, "k": int(k),
                    "method": "model_regression", "name": family, "with_probe": bool(with_probe),
                    "abstention_rate": float(abstained_all[mask].mean()),
                    "tuned": tuning.get("tuned", False),
                    **metrics,
                })
    return records


def _classification_rows(
    fold_name: str, regime: str, train: pd.DataFrame, test: pd.DataFrame,
    *, with_probe: bool, seed: int, k_grid: Sequence[int],
) -> list[dict[str, Any]]:
    features = horizon_feature_columns(train, with_probe=with_probe)
    assert_no_leakage(features)
    y_train = (train["device_label"] == "gpu").astype(int).to_numpy()
    if len(set(y_train)) < 2:
        # Sin las dos clases en entrenamiento la clasificacion binaria no esta
        # definida; se omite en lugar de fabricar un modelo degenerado.
        return []
    model = build_classification_pipeline(features, train, seed=seed)
    model.fit(train[features], y_train)
    predicted = model.predict(test[features])
    devices_all = np.where(predicted == 1, "gpu", "cpu")
    records: list[dict[str, Any]] = []
    for state in sorted(test["resource_state"].astype(str).unique()):
        state_mask = (test["resource_state"] == state).to_numpy()
        for k in k_grid:
            mask = state_mask & (test["k"] == k).to_numpy()
            if not mask.any():
                continue
            test_slice = test.loc[mask]
            metrics = evaluate_devices(test_slice, devices_all[mask])
            records.append({
                "fold": fold_name, "regime": regime, "resource_state": state, "k": int(k),
                "method": "model_classification", "name": "logistic", "with_probe": bool(with_probe),
                "abstention_rate": 0.0, "tuned": False,
                **metrics,
            })
    return records


def _baseline_rows(
    fold_name: str, regime: str, train: pd.DataFrame, test: pd.DataFrame, *, k_grid: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for state in sorted(test["resource_state"].astype(str).unique()):
        train_state = train[train["resource_state"] == state]
        test_state = test[test["resource_state"] == state]
        if train_state.empty or test_state.empty:
            continue
        for name, factory in BASELINES.items():
            predict = factory(train_state)
            for k in k_grid:
                test_slice = test_state[test_state["k"] == k]
                if test_slice.empty:
                    continue
                devices = predict(test_slice)
                metrics = evaluate_devices(test_slice, devices)
                records.append({
                    "fold": fold_name, "regime": regime, "resource_state": state, "k": int(k),
                    "method": "baseline", "name": name, "with_probe": None, "abstention_rate": None,
                    "tuned": None, **metrics,
                })
    return records


def evaluate_r2(
    folds: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
    *,
    seed: int = FROZEN_SEED,
    k_grid: Sequence[int] = K_GRID,
    with_probe_variants: Sequence[bool] = (False, True),
) -> pd.DataFrame:
    """Evalua baselines, regresores y el contraste de clasificacion en `folds`.

    Todos los metodos se evaluan sobre las mismas rebanadas
    ``(fold, resource_state, k)`` de `test`, que es la comparacion pareada
    que exige el protocolo (seccion 5 / 6).
    """
    records: list[dict[str, Any]] = []
    for fold_name, train, test in folds:
        regime = fold_name.split("_", 1)[0]
        records.extend(_baseline_rows(fold_name, regime, train, test, k_grid=k_grid))
        for with_probe in with_probe_variants:
            records.extend(_model_rows(
                fold_name, regime, train, test, with_probe=with_probe, seed=seed, k_grid=k_grid,
            ))
            records.extend(_classification_rows(
                fold_name, regime, train, test, with_probe=with_probe, seed=seed, k_grid=k_grid,
            ))
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Seleccion de modelo final y regla bloqueante (secciones 5.4 y 6)
# --------------------------------------------------------------------------


def select_final_model(results: pd.DataFrame) -> dict[str, Any]:
    """Familia (y variante de sondeo) con mejor `edp_sum_ratio_vs_oracle` en extrapolacion.

    Seccion 5.4: esta eleccion se hace y se registra ANTES de tocar los datos
    confirmatorios; no se vuelve a elegir despues.
    """
    models = results[
        (results["method"] == "model_regression") & (results["regime"] == "extrapolation")
    ]
    if models.empty:
        return {"family": None, "with_probe": None, "reason": "sin_filas_de_extrapolacion"}
    aggregated = models.groupby(["name", "with_probe"], observed=True)["edp_sum_ratio_vs_oracle"].mean()
    best_key = aggregated.idxmin()
    return {
        "family": str(best_key[0]),
        "with_probe": bool(best_key[1]),
        "mean_edp_sum_ratio_vs_oracle_extrapolation": float(aggregated.loc[best_key]),
        "criterion": "mean_edp_sum_ratio_vs_oracle_over_extrapolation_folds",
    }


def blocking_rule_report(results: pd.DataFrame) -> pd.DataFrame:
    """Aplica la regla bloqueante de la seccion 6, por `(regime, resource_state, k)`.

    Compara la MEJOR baseline pertinente (excluyendo `oracle`/`oracle_k`, que
    son cotas superiores con conocimiento posterior, no politicas desplegables)
    contra el MEJOR modelo (cualquier familia/variante de sondeo) en la misma
    rebanada de `test`. "Superar" significa mejorar `edp_sum_ratio_vs_oracle`
    por encima del piso de ruido global (seccion 7); no admite excepcion por
    cercania.
    """
    records: list[dict[str, Any]] = []
    keys = results[["regime", "resource_state", "k"]].drop_duplicates()
    for _, key in keys.iterrows():
        regime, state, k = key["regime"], key["resource_state"], key["k"]
        subset = results[
            (results["regime"] == regime) & (results["resource_state"] == state) & (results["k"] == k)
        ]
        baselines = subset[
            (subset["method"] == "baseline") & (~subset["name"].isin({"oracle", "oracle_k"}))
        ]
        models = subset[subset["method"].isin({"model_regression", "model_classification"})]
        if baselines.empty or models.empty:
            continue
        best_baseline_row = baselines.loc[baselines["edp_sum_ratio_vs_oracle"].idxmin()]
        best_model_row = models.loc[models["edp_sum_ratio_vs_oracle"].idxmin()]
        improvement_pct = 100.0 * (
            1.0 - best_model_row["edp_sum_ratio_vs_oracle"] / best_baseline_row["edp_sum_ratio_vs_oracle"]
        )
        model_wins = improvement_pct > NOISE_FLOOR_PCT
        records.append({
            "regime": regime, "resource_state": state, "k": int(k),
            "best_baseline": str(best_baseline_row["name"]),
            "best_baseline_edp_sum_ratio_vs_oracle": float(best_baseline_row["edp_sum_ratio_vs_oracle"]),
            "best_model": f"{best_model_row['method']}:{best_model_row['name']}"
                          f"{'+probe' if best_model_row.get('with_probe') else ''}",
            "best_model_edp_sum_ratio_vs_oracle": float(best_model_row["edp_sum_ratio_vs_oracle"]),
            "model_improvement_pct": float(improvement_pct),
            "model_beats_baseline_above_noise_floor": bool(model_wins),
            "adopted_policy": "model" if model_wins else "baseline",
        })
    return pd.DataFrame(records).sort_values(
        ["regime", "resource_state", "k"], kind="mergesort",
    ).reset_index(drop=True)


def static_vs_probe_report(results: pd.DataFrame) -> pd.DataFrame:
    """Contraste H2: modelo estatico contra modelo con sondeo, mismos pliegues."""
    models = results[results["method"] == "model_regression"]
    if models.empty:
        return pd.DataFrame()
    aggregated = models.groupby(
        ["regime", "name", "with_probe"], observed=True,
    ).agg(
        edp_sum_ratio_vs_oracle=("edp_sum_ratio_vs_oracle", "mean"),
        regret_ratio_mean=("regret_ratio_mean", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
    ).reset_index()
    static = aggregated[~aggregated["with_probe"]].set_index(["regime", "name"])
    probe = aggregated[aggregated["with_probe"]].set_index(["regime", "name"])
    common = static.index.intersection(probe.index)
    records = []
    for key in common:
        records.append({
            "regime": key[0], "family": key[1],
            "edp_sum_ratio_static": float(static.loc[key, "edp_sum_ratio_vs_oracle"]),
            "edp_sum_ratio_probe": float(probe.loc[key, "edp_sum_ratio_vs_oracle"]),
            "probe_improves_pct": float(100.0 * (
                1.0 - probe.loc[key, "edp_sum_ratio_vs_oracle"] / static.loc[key, "edp_sum_ratio_vs_oracle"]
            )),
            "probe_beats_static_above_noise_floor": bool(
                100.0 * (1.0 - probe.loc[key, "edp_sum_ratio_vs_oracle"] / static.loc[key, "edp_sum_ratio_vs_oracle"])
                > NOISE_FLOOR_PCT
            ),
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Latencia de inferencia y tamano del modelo (seccion 10.3)
# --------------------------------------------------------------------------


def measure_inference_latency(
    model: Any, sample: pd.DataFrame, *, warmups: int = 50, repeats: int = 200,
) -> dict[str, float]:
    """p50/p95/p99 de latencia de inferencia sobre una sola fila (una decision)."""
    row = sample.iloc[[0]]
    for _ in range(warmups):
        model.predict(row)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model.predict(row)
        timings.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "latency_p50_us": float(np.percentile(timings, 50)),
        "latency_p95_us": float(np.percentile(timings, 95)),
        "latency_p99_us": float(np.percentile(timings, 99)),
    }


def model_size_bytes(model: Any) -> int:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


# --------------------------------------------------------------------------
# Orquestador reproducible (analogo a `r1.run_r1_analysis`)
# --------------------------------------------------------------------------


def run_r2_analysis(
    dataset_dir: "str | Path",
    output_dir: "str | Path | None" = None,
    *,
    seed: int = FROZEN_SEED,
    k_grid: Sequence[int] = K_GRID,
) -> dict[str, "Path"]:
    """Ejecuta R2 sobre `selector_final_20260830`: horizonte, modelos, regla bloqueante."""
    from pathlib import Path as _Path
    import json

    from .compact import attach_probe_features
    from .sizes import extrapolation_folds, interpolation_folds

    dataset_dir = _Path(dataset_dir)
    output_dir = _Path(output_dir) if output_dir is not None else dataset_dir / "r2"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_summary.csv", low_memory=False)
    run_regions_path = dataset_dir / "run_regions.csv"
    run_regions = pd.read_csv(run_regions_path, low_memory=False) if run_regions_path.is_file() else None

    horizon_static = _build_horizon(candidates, k_grid=k_grid)
    horizon_probe = (
        attach_probe_features(horizon_static, run_regions)
        if run_regions is not None else horizon_static.copy()
    )

    interpolation = interpolation_folds(horizon_probe)
    extrapolation = extrapolation_folds(horizon_probe)
    folds = [*interpolation, *extrapolation]

    results = evaluate_r2(folds, seed=seed, k_grid=k_grid)
    blocking = blocking_rule_report(results)
    static_vs_probe = static_vs_probe_report(results)
    final_selection = select_final_model(results)

    latency: dict[str, Any] = {}
    if final_selection.get("family") is not None and extrapolation:
        _, train, test = extrapolation[-1]
        features = horizon_feature_columns(train, with_probe=bool(final_selection["with_probe"]))
        model, _ = fit_tuned_regressor(final_selection["family"], train, features, seed=seed)
        latency = {
            **measure_inference_latency(model, test[features]),
            "model_size_bytes": model_size_bytes(model),
            "fold_used": extrapolation[-1][0],
        }

    paths = {
        "horizon_dataset": output_dir / "horizon_dataset.csv",
        "r2_results": output_dir / "r2_results.csv",
        "blocking_rule": output_dir / "blocking_rule.csv",
        "static_vs_probe": output_dir / "static_vs_probe.csv",
        "summary": output_dir / "r2_summary.json",
    }
    horizon_probe.to_csv(paths["horizon_dataset"], index=False)
    results.to_csv(paths["r2_results"], index=False)
    blocking.to_csv(paths["blocking_rule"], index=False)
    static_vs_probe.to_csv(paths["static_vs_probe"], index=False)
    paths["summary"].write_text(
        json.dumps({
            "final_model_selection": final_selection,
            "latency_and_size": latency,
            "n_folds": len(folds),
            "k_grid": list(k_grid),
            "blocking_rule_pass_count": int(blocking["model_beats_baseline_above_noise_floor"].sum())
            if not blocking.empty else 0,
            "blocking_rule_total_count": int(len(blocking)),
            "interpretation_contract": {
                "k_is_supplied_not_predicted": True,
                "model_trained_across_all_resource_states": True,
                "baselines_fit_per_resource_state": True,
                "abstention_threshold_frozen_not_fit_on_test": True,
            },
        }, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return paths


def _build_horizon(candidates: pd.DataFrame, *, k_grid: Sequence[int]) -> pd.DataFrame:
    from .horizon import build_horizon_dataset
    return build_horizon_dataset(candidates, k_grid=k_grid)
