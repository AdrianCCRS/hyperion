"""Fase R2-estructurada: target de horizonte compuesto en tres capas.

Implementa la enmienda **2026-08-30-B** del protocolo congelado
(`docs/general/protocolo_congelado_confirmatorio_20260830.md`, seccion 13) y
la nota `6.4-bis` del plan de reformulacion. Contrasta el regresor directo
sobre ``y_log_edp_ratio_k`` con tres capas separadas. Los conteos exploratorios
iniciales quedaron supersedidos por la enmienda 2026-08-30-C: la evaluacion
vigente congela una politica por formulacion y conserva `fold` en toda
comparacion.

1. **Prediccion** (`fit_layer1`/`predict_layer1`): cuatro primitivas de costo
   **calientes** (`E_warm`, `T_warm` x CPU, GPU), en funcion de
   `(operacion, tamano)` -- ley de potencias en log-log, con R^2 agrupado
   fuera de muestra entre 0.941 y 0.983 en los datos exploratorios.
2. **Calibracion** (`calibrate_startup`): arranque por dispositivo,
   ``costo_frio(d) - costo_caliente(d)``, calculado SOLO con datos de
   entrenamiento. Decide constante global vs. tabla por operacion mediante
   validacion cruzada agrupada por `config_id` (nunca por conveniencia).
3. **Composicion** (`compose_device_costs`): costo frio predicho = caliente
   predicho + arranque calibrado; `EDP_total(d, K | estado)` se deriva con
   `horizon.edp_total_state`, la MISMA formula ya congelada -- no se reaprende
   ninguna cuarta capa.

El sondeo (`with_probe=True`) sustituye la primitiva fria predicha por la
capa 1+2 con la medida real del dispositivo que sondeo, cuando existe para
ese `config_id` (seccion 13.2). No cambia las capas 1/2.

Evaluacion: mismos pliegues, misma rejilla de `K`, mismas baselines y la
misma `evaluate_devices` que `r2.py` (seccion 13.3 del protocolo -- esto NO
reemplaza a `r2.py`, compite con el con una comparacion pareada explicita,
ver `three_way_blocking_rule`).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

import numpy as np
import pandas as pd

from .compact import (
    NOISE_FLOOR_PCT,
    STATIC_FEATURES,
    CompactDatasetError,
    _ref_lookup,
    assert_no_leakage,
    ref_configurations,
)
from .dataset import _static_descriptors
from .horizon import K_GRID, device_costs, edp_total_state
from .r2 import (
    FROZEN_SEED,
    REGRESSOR_FAMILIES,
    build_regressor_pipeline,
    devices_from_predictions,
)
from .sizes import BASELINES, evaluate_devices

#: Caracteristicas de la capa 1: las mismas del modelo estatico (seccion 3.1)
#: MENOS `resource_state`, que no aplica a una primitiva por-dispositivo (la
#: primitiva caliente de un `config_id` no depende del estado de recurso).
PRIMITIVE_FEATURES: tuple[str, ...] = tuple(
    column for column in STATIC_FEATURES if column != "resource_state"
)

#: Las cuatro primitivas calientes de la capa 1 (seccion 13.2, punto 1).
LAYER1_TARGETS: tuple[str, ...] = ("cpu_e_warm", "cpu_t_warm", "gpu_e_warm", "gpu_t_warm")

_DEVICES = ("cpu", "gpu")
_QUANTITIES = ("e", "t")


class StructuredModelError(RuntimeError):
    """La configuracion del modelo estructurado no es valida."""


# --------------------------------------------------------------------------
# Dataset de primitivas: una fila por `config_id` (seccion 13.2, capa 1)
# --------------------------------------------------------------------------


def build_primitives_dataset(candidates: pd.DataFrame) -> pd.DataFrame:
    """Una fila por `config_id` con las 8 primitivas de costo (frio/caliente x CPU/GPU).

    Reusa `horizon.device_costs`, que ya valida que las dos acciones REF
    existen en `cold` y `warm`. Los targets de la capa 1 se guardan en
    log-natural (``log_<device>_<primitivo>``) porque la ley de potencias es
    lineal en escala log-log; los costos crudos se conservan para calibrar
    la capa 2 y para pruebas de consistencia, nunca como entrada de un
    regresor (no estan en `PRIMITIVE_FEATURES`).
    """
    costs = device_costs(candidates)
    lookup = _ref_lookup(candidates)
    records: list[dict[str, Any]] = []
    for config_id in ref_configurations(candidates):
        sample = lookup[(config_id, "cpu", "cold")]
        operation, size = str(sample["operation"]), int(sample["size"])
        record: dict[str, Any] = {"config_id": config_id, "operation": operation, "size": size}
        record.update(_static_descriptors(operation, size))
        for device in _DEVICES:
            for field in ("e_cold", "t_cold", "e_warm", "t_warm"):
                value = float(costs[config_id][device][field])
                if not (value > 0):
                    raise StructuredModelError(
                        f"costo no positivo en {config_id}/{device}/{field}: {value}"
                    )
                record[f"{device}_{field}"] = value
                record[f"log_{device}_{field}"] = math.log(value)
            record[f"{device}_startup_e"] = record[f"{device}_e_cold"] - record[f"{device}_e_warm"]
            record[f"{device}_startup_t"] = record[f"{device}_t_cold"] - record[f"{device}_t_warm"]
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise StructuredModelError("ninguna configuracion tiene las dos acciones REF completas")
    return frame.sort_values("config_id", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Capa 1: prediccion de costos calientes
# --------------------------------------------------------------------------


def fit_layer1(
    primitives_train: pd.DataFrame, family: str, *, seed: int = FROZEN_SEED,
) -> dict[str, Any]:
    """Ajusta un pipeline por primitiva caliente (4 en total), sin GridSearchCV.

    Deliberado: la capa 1 es una ley de potencias casi perfecta (R^2 entre
    0.974 y 0.998 verificado en la sesion exploratoria, ver
    `layer1_grouped_cv_r2`), ajustar 4 targets x 5 familias x N pliegues con
    la misma rejilla de hiperparametros que `r2.py` multiplicaria el costo de
    computo del modelo estructurado sin cambiar la conclusion -- el protocolo
    (seccion 13.4) explicitamente permite que la capa 2 no necesite ajuste de
    hiperparametros; se extiende el mismo criterio a la capa 1 porque la
    evidencia de ajuste ya esta verificada, no supuesta. Se usan los
    hiperparametros por defecto de cada familia (`r2._base_estimator`).
    """
    missing = sorted(set(PRIMITIVE_FEATURES) - set(primitives_train.columns))
    if missing:
        raise StructuredModelError(f"faltan columnas de caracteristicas de capa 1: {missing}")
    models: dict[str, Any] = {}
    for target in LAYER1_TARGETS:
        log_col = f"log_{target}"
        pipeline = build_regressor_pipeline(
            family, PRIMITIVE_FEATURES, primitives_train, seed=seed,
        )
        pipeline.fit(
            primitives_train[list(PRIMITIVE_FEATURES)],
            primitives_train[log_col].to_numpy(dtype=float),
        )
        models[target] = pipeline
    return models


def predict_layer1(models: Mapping[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Costos calientes predichos (escala natural, no log) para cada fila de `frame`."""
    out: dict[str, np.ndarray] = {}
    for target, pipeline in models.items():
        log_pred = pipeline.predict(frame[list(PRIMITIVE_FEATURES)])
        out[target] = np.exp(np.asarray(log_pred, dtype=float))
    return out


def layer1_grouped_cv_r2(
    primitives: pd.DataFrame, family: str = "ridge", *, seed: int = FROZEN_SEED, n_splits: int = 3,
) -> dict[str, float]:
    """R^2 fuera de muestra de la capa 1, por primitiva, con CV agrupada por `config_id`.

    Diagnostico exploratorio (no se usa para entrenar el modelo final): sirve
    para verificar sobre datos reales la afirmacion del protocolo de que la
    ley de potencias generaliza (R^2 entre 0.974 y 0.998), en vez de asumirla
    de memoria.
    """
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    groups = primitives["config_id"].astype(str).to_numpy()
    n_groups = len(set(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return {target: float("nan") for target in LAYER1_TARGETS}
    out: dict[str, float] = {}
    for target in LAYER1_TARGETS:
        log_col = f"log_{target}"
        y_true_all: list[float] = []
        y_pred_all: list[float] = []
        for train_idx, test_idx in GroupKFold(n_splits=splits).split(primitives, groups=groups):
            train = primitives.iloc[train_idx]
            test = primitives.iloc[test_idx]
            pipeline = build_regressor_pipeline(family, PRIMITIVE_FEATURES, train, seed=seed)
            pipeline.fit(train[list(PRIMITIVE_FEATURES)], train[log_col].to_numpy(dtype=float))
            y_pred_all.extend(pipeline.predict(test[list(PRIMITIVE_FEATURES)]).tolist())
            y_true_all.extend(test[log_col].tolist())
        out[target] = float(r2_score(y_true_all, y_pred_all))
    return out


# --------------------------------------------------------------------------
# Capa 2: calibracion de arranque
# --------------------------------------------------------------------------


def _grouped_cv_mae(
    primitives_train: pd.DataFrame, device: str, quantity: str, mode: str, *, n_splits: int,
) -> float:
    """MAE fuera de muestra (CV agrupada por `config_id`) de un modo de arranque.

    `mode` in {"constant", "per_operation"}. Ajustado y evaluado enteramente
    dentro de `primitives_train`: ningun dato de prueba externo interviene en
    esta decision (seccion 5.3 del protocolo, aplicada a la capa 2).
    """
    from sklearn.model_selection import GroupKFold

    column = f"{device}_startup_{quantity}"
    groups = primitives_train["config_id"].astype(str).to_numpy()
    n_groups = len(set(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return float("nan")
    errors: list[float] = []
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(primitives_train, groups=groups):
        train = primitives_train.iloc[train_idx]
        test = primitives_train.iloc[test_idx]
        global_constant = float(train[column].median())
        if mode == "constant":
            predicted = np.full(len(test), global_constant)
        else:
            table = train.groupby("operation", observed=True)[column].median()
            predicted = np.array([
                float(table.get(operation, global_constant))
                for operation in test["operation"].astype(str)
            ])
        errors.extend(np.abs(test[column].to_numpy(dtype=float) - predicted).tolist())
    return float(np.mean(errors))


def calibrate_startup(
    primitives_train: pd.DataFrame, *, n_splits: int = 3,
) -> dict[str, Any]:
    """Arranque por dispositivo x magnitud, con la decision constante/por-operacion
    hecha por CV agrupada, no supuesta (seccion 13.2, capa 2).

    Devuelve, por `(device, quantity)`:
    - ``mode``: ``"constant"`` o ``"per_operation"``, el que tuvo menor MAE de
      CV agrupada en `primitives_train`;
    - ``constant``: mediana global (fallback tambien cuando `mode` es
      `"per_operation"` y aparece una operacion no vista en entrenamiento);
    - ``table``: mediana por operacion (solo se usa si `mode` es `per_operation`);
    - ``cv_mae_constant`` / ``cv_mae_per_operation``: diagnostico de la decision.
    """
    calibration: dict[str, Any] = {}
    for device in _DEVICES:
        calibration[device] = {}
        for quantity in _QUANTITIES:
            column = f"{device}_startup_{quantity}"
            global_constant = float(primitives_train[column].median())
            table = primitives_train.groupby("operation", observed=True)[column].median().to_dict()
            mae_constant = _grouped_cv_mae(primitives_train, device, quantity, "constant", n_splits=n_splits)
            mae_per_operation = _grouped_cv_mae(
                primitives_train, device, quantity, "per_operation", n_splits=n_splits,
            )
            # Empate (incluido un empate hasta ruido de punto flotante) o CV
            # no disponible (muy pocos config_id, solo en pruebas sinteticas
            # minimas) -> constante, por ser el modelo mas simple. El margen
            # relativo evita que una diferencia de 1e-14 entre dos MAE
            # numericamente identicos decida a favor de la tabla mas compleja.
            improves = mae_constant - mae_per_operation
            meaningful = improves > max(1e-9, 1e-6 * mae_constant)
            if np.isfinite(mae_per_operation) and meaningful:
                mode = "per_operation"
            else:
                mode = "constant"
            calibration[device][quantity] = {
                "mode": mode,
                "constant": global_constant,
                "table": {str(k): float(v) for k, v in table.items()},
                "cv_mae_constant": mae_constant,
                "cv_mae_per_operation": mae_per_operation,
            }
    return calibration


def predict_startup(calibration: Mapping[str, Any], device: str, quantity: str, operation: str) -> float:
    entry = calibration[device][quantity]
    if entry["mode"] == "per_operation":
        return float(entry["table"].get(operation, entry["constant"]))
    return float(entry["constant"])


# --------------------------------------------------------------------------
# Sondeo: cache de costo frio medido por (config_id, device)
# --------------------------------------------------------------------------


def build_probe_cold_cache(horizon_probe: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """`(config_id, device) -> (e_cold_medido, t_cold_medido)` desde el sondeo.

    `attach_probe_features` (compact.py) adjunta el sondeo unicamente en la
    fila `resource_state == "{device}_ready"`, porque ese es el estado en el
    que ese dispositivo ya se ejecuto y produjo telemetria de su propia
    region fria (la ejecucion que lo llevo a "ready"). El costo frio medido
    de un dispositivo es una propiedad de `(device, config_id)`, no de
    `resource_state`: se agrega aqui una sola vez por config_id x dispositivo
    y se reutiliza para CUALQUIER fila de ese config_id en la capa 3 (seccion
    13.2: "sustituye... la medida real del dispositivo que sondeo").
    """
    if "probe_device" not in horizon_probe or "probe_energy_per_dispatch_j" not in horizon_probe:
        return {}
    cache: dict[tuple[str, str], tuple[float, float]] = {}
    work = horizon_probe[
        horizon_probe["probe_energy_per_dispatch_j"].notna()
        & horizon_probe["probe_time_per_dispatch_s"].notna()
    ]
    for row in work.drop_duplicates(["config_id", "probe_device"]).to_dict("records"):
        device = row.get("probe_device")
        if not device or pd.isna(device):
            continue
        cache[(str(row["config_id"]), str(device))] = (
            float(row["probe_energy_per_dispatch_j"]), float(row["probe_time_per_dispatch_s"]),
        )
    return cache


# --------------------------------------------------------------------------
# Capa 3: composicion analitica (misma formula que horizon.py, no se reaprende)
# --------------------------------------------------------------------------


def compose_device_costs(
    e_warm: float, t_warm: float, e_startup: float, t_startup: float,
    *, probe: tuple[float, float] | None = None,
) -> dict[str, float]:
    """``e_cold/t_cold = caliente + arranque``, o el valor medido si hay sondeo.

    El arranque puede empujar el costo frio por debajo del caliente si el
    ruido de calibracion lo hace negativo; se recorta a un minimo del costo
    caliente porque un dispositivo no puede arrancar mas rapido de lo que
    corre ya caliente -- es una salvaguarda de composicion, no una regla
    aprendida.
    """
    if probe is not None:
        e_cold, t_cold = probe
    else:
        e_cold = max(e_warm + e_startup, e_warm)
        t_cold = max(t_warm + t_startup, t_warm)
    return {"e_warm": e_warm, "t_warm": t_warm, "e_cold": e_cold, "t_cold": t_cold}


def structured_costs_by_config(
    layer1_models: Mapping[str, Any],
    calibration: Mapping[str, Any],
    primitives_frame: pd.DataFrame,
    *,
    probe_cache: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """`config_id -> device -> {e_warm,t_warm,e_cold,t_cold}` compuestos (capa 3)."""
    frame = primitives_frame.reset_index(drop=True)
    warm_pred = predict_layer1(layer1_models, frame)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for i, row in enumerate(frame.to_dict("records")):
        config_id, operation = str(row["config_id"]), str(row["operation"])
        per_device: dict[str, dict[str, float]] = {}
        for device in _DEVICES:
            e_warm = float(warm_pred[f"{device}_e_warm"][i])
            t_warm = float(warm_pred[f"{device}_t_warm"][i])
            e_startup = predict_startup(calibration, device, "e", operation)
            t_startup = predict_startup(calibration, device, "t", operation)
            probe = (probe_cache or {}).get((config_id, device))
            per_device[device] = compose_device_costs(
                e_warm, t_warm, e_startup, t_startup, probe=probe,
            )
        out[config_id] = per_device
    return out


def structured_y_for_rows(
    costs_by_config: Mapping[str, Mapping[str, Mapping[str, float]]], test: pd.DataFrame,
) -> np.ndarray:
    """``y_estructurado`` por fila de `test`, usando `horizon.edp_total_state`.

    Esta es la unica funcion que toca la formula de composicion, y la reusa
    de `horizon.py` sin reimplementarla (seccion 13.2, capa 3).
    """
    y = np.empty(len(test), dtype=float)
    for i, row in enumerate(test.to_dict("records")):
        config_id, state, k = str(row["config_id"]), str(row["resource_state"]), int(row["k"])
        per_device = costs_by_config[config_id]
        cpu_edp = edp_total_state(per_device["cpu"], device="cpu", state=state, k=k)[2]
        gpu_edp = edp_total_state(per_device["gpu"], device="gpu", state=state, k=k)[2]
        y[i] = math.log(gpu_edp / cpu_edp)
    return y


# --------------------------------------------------------------------------
# Evaluacion pareada, mismos pliegues y mismas metricas que r2.py
# --------------------------------------------------------------------------


def _structured_model_rows(
    fold_name: str, regime: str, train: pd.DataFrame, test: pd.DataFrame,
    primitives: pd.DataFrame, probe_cache: Mapping[tuple[str, str], tuple[float, float]],
    *, with_probe: bool, seed: int, k_grid: Sequence[int],
) -> list[dict[str, Any]]:
    assert_no_leakage(PRIMITIVE_FEATURES)
    train_configs = set(train["config_id"].astype(str))
    test_configs = set(test["config_id"].astype(str))
    primitives_train = primitives[primitives["config_id"].astype(str).isin(train_configs)]
    primitives_test = primitives[primitives["config_id"].astype(str).isin(test_configs)]
    if primitives_train.empty or primitives_test.empty:
        return []
    calibration = calibrate_startup(primitives_train)
    active_probe_cache = probe_cache if with_probe else {}
    records: list[dict[str, Any]] = []
    for family in REGRESSOR_FAMILIES:
        layer1_models = fit_layer1(primitives_train, family, seed=seed)
        costs_by_config = structured_costs_by_config(
            layer1_models, calibration, primitives_test, probe_cache=active_probe_cache,
        )
        y_pred_all = structured_y_for_rows(costs_by_config, test)
        devices_all, abstained_all = devices_from_predictions(
            y_pred_all, test["resource_state"].to_numpy()
        )
        for state in sorted(test["resource_state"].astype(str).unique()):
            state_mask = (test["resource_state"] == state).to_numpy()
            for k in k_grid:
                mask = state_mask & (test["k"] == k).to_numpy()
                if not mask.any():
                    continue
                metrics = evaluate_devices(test.loc[mask], devices_all[mask])
                records.append({
                    "fold": fold_name, "regime": regime, "resource_state": state, "k": int(k),
                    "method": "model_structured", "name": family, "with_probe": bool(with_probe),
                    "abstention_rate": float(abstained_all[mask].mean()),
                    "startup_mode_cpu_e": calibration["cpu"]["e"]["mode"],
                    "startup_mode_cpu_t": calibration["cpu"]["t"]["mode"],
                    "startup_mode_gpu_e": calibration["gpu"]["e"]["mode"],
                    "startup_mode_gpu_t": calibration["gpu"]["t"]["mode"],
                    **metrics,
                })
    return records


def evaluate_structured(
    folds: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidates: pd.DataFrame,
    *,
    run_regions: pd.DataFrame | None = None,
    seed: int = FROZEN_SEED,
    k_grid: Sequence[int] = K_GRID,
    with_probe_variants: Sequence[bool] = (False, True),
) -> pd.DataFrame:
    """Evalua el modelo estructurado (3 capas) sobre `folds`, sin baselines.

    Las baselines no se recalculan aqui: son identicas a las de `r2.py`
    (mismo `sizes.BASELINES`, mismos pliegues) y `three_way_blocking_rule`
    las toma directamente de `r2_results`, para no duplicar computo ni
    arriesgar una segunda implementacion que diverja.
    """
    primitives = build_primitives_dataset(candidates)
    probe_cache: dict[tuple[str, str], tuple[float, float]] = {}
    if run_regions is not None:
        from .compact import attach_probe_features
        from .horizon import build_horizon_dataset

        horizon_static = build_horizon_dataset(candidates, k_grid=k_grid)
        horizon_probe = attach_probe_features(horizon_static, run_regions)
        probe_cache = build_probe_cold_cache(horizon_probe)
    records: list[dict[str, Any]] = []
    for fold_name, train, test in folds:
        regime = fold_name.split("_", 1)[0]
        for with_probe in with_probe_variants:
            if with_probe and not probe_cache:
                continue
            records.extend(_structured_model_rows(
                fold_name, regime, train, test, primitives, probe_cache,
                with_probe=with_probe, seed=seed, k_grid=k_grid,
            ))
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Comparacion de tres vias: baseline / directo / estructurado (seccion 13.3)
# --------------------------------------------------------------------------


def select_final_structured_model(results: pd.DataFrame) -> dict[str, Any]:
    """Congela una familia estructurada y una variante de sondeo."""
    models = results[
        (results["method"] == "model_structured")
        & (results["regime"] == "extrapolation")
    ]
    if models.empty:
        return {"family": None, "with_probe": None, "reason": "sin_filas_de_extrapolacion"}
    aggregated = models.groupby(["name", "with_probe"], observed=True)[
        "edp_sum_ratio_vs_oracle"
    ].mean()
    best_key = aggregated.idxmin()
    return {
        "family": str(best_key[0]),
        "with_probe": bool(best_key[1]),
        "mean_edp_sum_ratio_vs_oracle_extrapolation": float(aggregated.loc[best_key]),
        "criterion": "mean_edp_sum_ratio_over_extrapolation_folds",
    }


def three_way_blocking_rule(
    direct_results: pd.DataFrame,
    structured_results: pd.DataFrame,
    *,
    baseline_selection: Mapping[str, Any] | None = None,
    direct_selection: Mapping[str, Any] | None = None,
    structured_selection: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compara tres politicas congeladas dentro del mismo pliegue de test.

    No se elige familia, variante de sondeo ni pliegue despues de observar
    cada rebanada. Las tres politicas se seleccionan una vez sobre el conjunto
    exploratorio y despues se enfrentan sobre el mismo
    `(fold, resource_state, k)`.
    """
    from . import r2

    baseline_selection = dict(baseline_selection or r2.select_final_baseline(direct_results))
    direct_selection = dict(direct_selection or r2.select_final_model(direct_results))
    structured_selection = dict(
        structured_selection or select_final_structured_model(structured_results)
    )
    if any(selection.get("name", selection.get("family")) is None for selection in (
        baseline_selection, direct_selection, structured_selection,
    )):
        return pd.DataFrame()

    combined = pd.concat([direct_results, structured_results], ignore_index=True, sort=False)
    records: list[dict[str, Any]] = []
    keys = combined[["fold", "regime", "resource_state", "k"]].drop_duplicates()
    for _, key in keys.iterrows():
        fold, regime = key["fold"], key["regime"]
        state, k = key["resource_state"], key["k"]
        subset = combined[
            (combined["fold"] == fold)
            & (combined["resource_state"] == state)
            & (combined["k"] == k)
        ]
        baseline_name = baseline_selection.get("by_regime_resource_state_k", {}).get(
            f"{regime}|{state}|{int(k)}", baseline_selection["name"],
        )
        baselines = subset[
            (subset["method"] == "baseline")
            & (subset["name"] == baseline_name)
        ]
        direct = subset[
            (subset["method"] == "model_regression")
            & (subset["name"] == direct_selection["family"])
            & r2._probe_mask(subset, bool(direct_selection["with_probe"]))
        ]
        structured = subset[
            (subset["method"] == "model_structured")
            & (subset["name"] == structured_selection["family"])
            & r2._probe_mask(subset, bool(structured_selection["with_probe"]))
        ]
        if baselines.empty or direct.empty or structured.empty:
            continue
        baseline_row, direct_row, structured_row = baselines.iloc[0], direct.iloc[0], structured.iloc[0]
        contenders = [("direct", direct_row), ("structured", structured_row)]
        winner_kind, winner_row = min(contenders, key=lambda item: item[1]["edp_sum_ratio_vs_oracle"])
        improvement = 100.0 * (
            1.0 - float(winner_row["edp_sum_ratio_vs_oracle"])
            / float(baseline_row["edp_sum_ratio_vs_oracle"])
        )
        records.append({
            "fold": fold, "regime": regime, "resource_state": state, "k": int(k),
            "selected_baseline": str(baseline_row["name"]),
            "selected_baseline_edp_sum_ratio_vs_oracle": float(baseline_row["edp_sum_ratio_vs_oracle"]),
            "selected_direct_name": (
                f"{direct_row['name']}{'+probe' if direct_row.get('with_probe') else ''}"
            ),
            "selected_direct_edp_sum_ratio_vs_oracle": float(direct_row["edp_sum_ratio_vs_oracle"]),
            "selected_structured_name": (
                f"{structured_row['name']}{'+probe' if structured_row.get('with_probe') else ''}"
            ),
            "selected_structured_edp_sum_ratio_vs_oracle": float(
                structured_row["edp_sum_ratio_vs_oracle"]
            ),
            "winner_formulation": winner_kind,
            "winner_edp_sum_ratio_vs_oracle": float(winner_row["edp_sum_ratio_vs_oracle"]),
            "n": int(winner_row["n"]),
            "oracle_edp_sum_js": float(
                winner_row["edp_sum_js"] / winner_row["edp_sum_ratio_vs_oracle"]
            ),
            "selected_baseline_edp_sum_js": float(baseline_row["edp_sum_js"]),
            "selected_direct_edp_sum_js": float(direct_row["edp_sum_js"]),
            "selected_structured_edp_sum_js": float(structured_row["edp_sum_js"]),
            "winner_edp_sum_js": float(winner_row["edp_sum_js"]),
            "improvement_pct_over_best_baseline": improvement,
            "beats_baseline_above_noise_floor": bool(improvement > NOISE_FLOOR_PCT),
        })
    return pd.DataFrame(records).sort_values(
        ["regime", "fold", "resource_state", "k"], kind="mergesort",
    ).reset_index(drop=True)


def aggregate_three_way_report(report: pd.DataFrame) -> pd.DataFrame:
    """Suma interpolacion disjunta y mantiene separados top1/top2."""
    if report.empty:
        return pd.DataFrame()
    work = report.copy()
    work["evaluation_scope"] = np.where(
        work["regime"] == "interpolation", "interpolation_all", work["fold"],
    )
    records: list[dict[str, Any]] = []
    for (scope, state, k), group in work.groupby(
        ["evaluation_scope", "resource_state", "k"], observed=True,
    ):
        oracle = float(group["oracle_edp_sum_js"].sum())
        baseline = float(group["selected_baseline_edp_sum_js"].sum())
        direct = float(group["selected_direct_edp_sum_js"].sum())
        structured_cost = float(group["selected_structured_edp_sum_js"].sum())
        winner_kind, winner = min(("direct", direct), ("structured", structured_cost), key=lambda x: x[1])
        improvement = 100.0 * (1.0 - winner / baseline)
        records.append({
            "evaluation_scope": scope,
            "resource_state": state,
            "k": int(k),
            "n": int(group["n"].sum()),
            "selected_baseline": str(group["selected_baseline"].iloc[0]),
            "selected_direct_name": str(group["selected_direct_name"].iloc[0]),
            "selected_structured_name": str(group["selected_structured_name"].iloc[0]),
            "selected_baseline_edp_sum_ratio_vs_oracle": baseline / oracle,
            "selected_direct_edp_sum_ratio_vs_oracle": direct / oracle,
            "selected_structured_edp_sum_ratio_vs_oracle": structured_cost / oracle,
            "winner_formulation": winner_kind,
            "winner_edp_sum_ratio_vs_oracle": winner / oracle,
            "improvement_pct_over_best_baseline": improvement,
            "beats_baseline_above_noise_floor": bool(improvement > NOISE_FLOOR_PCT),
        })
    return pd.DataFrame(records).sort_values(
        ["evaluation_scope", "resource_state", "k"], kind="mergesort",
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Latencia y tamano (comparables a `r2.measure_inference_latency`/`model_size_bytes`)
# --------------------------------------------------------------------------


def measure_structured_latency(
    layer1_models: Mapping[str, Any], calibration: Mapping[str, Any],
    sample_primitives: pd.DataFrame, *, warmups: int = 50, repeats: int = 200,
) -> dict[str, float]:
    """p50/p95/p99 de la capa 1+2+3 completa para UNA decision (una fila)."""
    import time

    row = sample_primitives.iloc[[0]]
    probe_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def _one_decision() -> None:
        costs = structured_costs_by_config(layer1_models, calibration, row, probe_cache=probe_cache)
        config_id = str(row["config_id"].iloc[0])
        per_device = costs[config_id]
        edp_total_state(per_device["cpu"], device="cpu", state="none_ready", k=1)
        edp_total_state(per_device["gpu"], device="gpu", state="none_ready", k=1)

    for _ in range(warmups):
        _one_decision()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        _one_decision()
        timings.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "latency_p50_us": float(np.percentile(timings, 50)),
        "latency_p95_us": float(np.percentile(timings, 95)),
        "latency_p99_us": float(np.percentile(timings, 99)),
    }


def structured_model_size_bytes(layer1_models: Mapping[str, Any], calibration: Mapping[str, Any]) -> int:
    import pickle
    return len(pickle.dumps(
        {"layer1": dict(layer1_models), "layer2": dict(calibration)},
        protocol=pickle.HIGHEST_PROTOCOL,
    ))


# --------------------------------------------------------------------------
# Orquestador reproducible (analogo a `r2.run_r2_analysis`)
# --------------------------------------------------------------------------


def run_structured_analysis(
    dataset_dir: "str | Path",
    output_dir: "str | Path | None" = None,
    *,
    seed: int = FROZEN_SEED,
    k_grid: Sequence[int] = K_GRID,
    direct_results: pd.DataFrame | None = None,
) -> dict[str, "Path"]:
    """Ejecuta el modelo estructurado sobre `selector_final_20260830` y lo compara
    con el modelo directo de `r2.py` en las mismas rebanadas (seccion 13.3).

    Si `direct_results` no se provee, se recalcula con `r2.evaluate_r2` sobre
    los mismos pliegues (determinista con la semilla congelada).
    """
    from pathlib import Path as _Path
    import json
    import hashlib

    from .sizes import extrapolation_folds, interpolation_folds

    dataset_dir = _Path(dataset_dir)
    output_dir = _Path(output_dir) if output_dir is not None else dataset_dir / "r2_structured"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_summary.csv", low_memory=False)
    run_regions_path = dataset_dir / "run_regions.csv"
    run_regions = pd.read_csv(run_regions_path, low_memory=False) if run_regions_path.is_file() else None

    primitives = build_primitives_dataset(candidates)
    layer1_r2 = layer1_grouped_cv_r2(primitives, "ridge", seed=seed)

    from .compact import attach_probe_features
    from .horizon import build_horizon_dataset

    horizon_static = build_horizon_dataset(candidates, k_grid=k_grid)
    horizon_probe = (
        attach_probe_features(horizon_static, run_regions)
        if run_regions is not None else horizon_static.copy()
    )
    interpolation = interpolation_folds(horizon_probe)
    extrapolation = extrapolation_folds(horizon_probe)
    folds = [*interpolation, *extrapolation]

    from . import r2 as _r2
    if direct_results is None:
        direct_results = _r2.evaluate_r2(folds, seed=seed, k_grid=k_grid)

    structured_results = evaluate_structured(
        folds, candidates, run_regions=run_regions, seed=seed, k_grid=k_grid,
    )
    baseline_selection = _r2.select_final_baseline(direct_results)
    direct_selection = _r2.select_final_model(direct_results)
    structured_selection = select_final_structured_model(structured_results)
    three_way = three_way_blocking_rule(
        direct_results,
        structured_results,
        baseline_selection=baseline_selection,
        direct_selection=direct_selection,
        structured_selection=structured_selection,
    )
    three_way_aggregated = aggregate_three_way_report(three_way)

    latency: dict[str, Any] = {}
    if extrapolation:
        _, train_fold, _ = extrapolation[-1]
        primitives_train = primitives[
            primitives["config_id"].astype(str).isin(set(train_fold["config_id"].astype(str)))
        ]
        if not primitives_train.empty:
            calibration = calibrate_startup(primitives_train)
            selected_family = str(structured_selection["family"])
            layer1_models = fit_layer1(primitives_train, selected_family, seed=seed)
            latency = {
                **measure_structured_latency(layer1_models, calibration, primitives_train),
                "model_size_bytes": structured_model_size_bytes(layer1_models, calibration),
                "fold_used": extrapolation[-1][0],
                "family": selected_family,
            }

    paths = {
        "primitives_dataset": output_dir / "primitives_dataset.csv",
        "structured_results": output_dir / "structured_results.csv",
        "three_way_blocking_rule": output_dir / "three_way_blocking_rule.csv",
        "three_way_blocking_rule_aggregated": output_dir / "three_way_blocking_rule_aggregated.csv",
        "summary": output_dir / "structured_summary.json",
    }
    primitives.to_csv(paths["primitives_dataset"], index=False)
    structured_results.to_csv(paths["structured_results"], index=False)
    three_way.to_csv(paths["three_way_blocking_rule"], index=False)
    three_way_aggregated.to_csv(paths["three_way_blocking_rule_aggregated"], index=False)
    n_slices = int(len(three_way))
    n_pass = int(three_way["beats_baseline_above_noise_floor"].sum()) if n_slices else 0
    n_structured_wins = int((three_way["winner_formulation"] == "structured").sum()) if n_slices else 0
    n_direct_wins = int((three_way["winner_formulation"] == "direct").sum()) if n_slices else 0
    paths["summary"].write_text(
        json.dumps({
            "input_sha256": {
                "candidate_summary.csv": hashlib.sha256(
                    (dataset_dir / "candidate_summary.csv").read_bytes()
                ).hexdigest(),
                "run_regions.csv": hashlib.sha256(run_regions_path.read_bytes()).hexdigest()
                if run_regions_path.is_file() else None,
            },
            "layer1_grouped_cv_r2_ridge": layer1_r2,
            "final_baseline_selection": baseline_selection,
            "final_direct_selection": direct_selection,
            "final_structured_selection": structured_selection,
            "n_folds": len(folds),
            "k_grid": list(k_grid),
            "three_way_total_slices": n_slices,
            "three_way_pass_count": n_pass,
            "three_way_structured_wins": n_structured_wins,
            "three_way_direct_wins": n_direct_wins,
            "three_way_aggregated_total_slices": int(len(three_way_aggregated)),
            "three_way_aggregated_pass_count": int(
                three_way_aggregated["beats_baseline_above_noise_floor"].sum()
            ) if not three_way_aggregated.empty else 0,
            "latency_and_size": latency,
            "interpretation_contract": {
                "k_is_supplied_not_predicted": True,
                "layer1_no_hyperparameter_search": True,
                "layer2_mode_chosen_by_grouped_cv_not_assumed": True,
                "layer3_reuses_horizon_edp_total_state_formula": True,
                "probe_substitutes_layer1_2_cold_prediction_only": True,
            },
        }, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return paths
