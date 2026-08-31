"""R3-A: política DVFS offline, previa al controlador en hardware.

La unidad independiente sigue siendo ``config_id``. Para cada estado se elige
primero el dispositivo ganador a REF y solo entonces se comparan frecuencias
de ese dispositivo. El modelo predice primitivas (tiempo y energía), compone
EDP y puede abstenerse; nunca aprende directamente la acción ganadora.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

import numpy as np
import pandas as pd

from .compact import RESOURCE_STATES, _boolean_mask
from .dataset import _static_descriptors
from .r2 import FROZEN_SEED, REGRESSOR_FAMILIES, _base_estimator, _preprocessor
from .sizes import assert_no_config_leak, extrapolation_folds, interpolation_folds


REGION_NOISE_PCT: Mapping[str, float] = {"cold": 5.76, "warm": 1.80}
DVFS_FAMILIES: tuple[str, ...] = ("power_law", "curve_physical", *REGRESSOR_FAMILIES)

# `tree`/`random_forest` heredan de r2.py un max_depth<=3/5 congelado en la
# seccion 4 del protocolo para el eje de dispositivo (R2): pocas categorias,
# margen enorme, la profundidad chica alcanza. R3 tiene ~40 acciones x 6
# operaciones -- un arbol de profundidad 3 no puede ni codificar las
# categorias. Verificado con datos reales (enmienda 2026-08-31-A): liberar la
# profundidad baja la mediana de error de 36.8% a 7.4% (p95 de 73.2% a
# 57.2%). Este override es local a R3, no cambia el limite congelado que usa
# R2 para el eje de dispositivo.
_DVFS_DEPTH_OVERRIDE: Mapping[str, dict[str, Any]] = {
    "tree": {"max_depth": None},
    "random_forest": {"max_depth": None},
}
DVFS_FEATURES: tuple[str, ...] = (
    "operation", "size", "log10_n", "flops_per_dispatch_analytic",
    "log10_flops_per_dispatch", "logical_bytes_per_dispatch",
    "log10_logical_bytes", "arithmetic_intensity_analytic",
    "resource_state", "device", "region", "frequency_action",
    "operation_frequency_action",
)


def _size_regimes(train: pd.DataFrame) -> dict[str, float]:
    """Mediana de tamaño por operación, calculada solo con TRAIN.

    No hay un umbral absoluto de tamaño valido entre operaciones: axpy solo
    tiene 2 tamaños (31623, 100000) y cholesky va de 64 a miles. El corte se
    hace relativo a la escala propia de cada operacion.
    """
    return train.groupby("operation")["size"].median().to_dict()


def _size_regime(operation: str, size: float, thresholds: Mapping[str, float]) -> str:
    threshold = thresholds.get(str(operation))
    if threshold is None:
        return "large"
    return "small" if float(size) < float(threshold) else "large"


class DVFSContractError(RuntimeError):
    """El dataset o una política DVFS viola el contrato de R3-A."""


@dataclass(frozen=True)
class CostModels:
    energy: Any
    time: Any
    uncertainty_pct: float
    # clave: (resource_state, device, size_regime) -- ver _size_regime.
    uncertainty_pct_by_context: Mapping[tuple[str, str, str], float]
    size_thresholds: Mapping[str, float]


class PowerLawCostModel:
    """Curva log(costo)=a+b*log(N) por operación×acción×región.

    Es la hipótesis física sencilla predeclarada en el plan. Los fallbacks se
    ajustan exclusivamente con entrenamiento y solo cubren casos de borde
    donde un pliegue no contiene dos tamaños de una curva.
    """

    def __init__(self) -> None:
        self.curves: dict[tuple[str, str, str], tuple[float, float]] = {}
        self.fallbacks: dict[tuple[str, str], float] = {}
        self.global_value = 0.0

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        work = x.copy()
        work["_target"] = np.asarray(y, dtype=float)
        self.global_value = float(work["_target"].median())
        self.fallbacks = {
            (str(operation), str(region)): float(group["_target"].median())
            for (operation, region), group in work.groupby(["operation", "region"], observed=True)
        }
        for key, group in work.groupby(
            ["operation", "frequency_action", "region"], observed=True,
        ):
            log_n = group["log10_n"].to_numpy(dtype=float)
            target = group["_target"].to_numpy(dtype=float)
            if len(np.unique(log_n)) >= 2:
                slope, intercept = np.polyfit(log_n, target, 1)
            else:
                slope, intercept = 0.0, float(np.median(target))
            self.curves[tuple(map(str, key))] = (float(intercept), float(slope))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        values: list[float] = []
        for row in x.to_dict("records"):
            key = (str(row["operation"]), str(row["frequency_action"]), str(row["region"]))
            curve = self.curves.get(key)
            if curve is None:
                value = self.fallbacks.get((key[0], key[2]), self.global_value)
            else:
                intercept, slope = curve
                value = intercept + slope * float(row["log10_n"])
            values.append(float(value))
        return np.asarray(values, dtype=float)


# Fraccion de frecuencia REAL observada (columnas `freq_khz_observed` en CPU
# y `gpu_sm_clock_mhz` en GPU de la telemetria de campana), normalizada al
# maximo medido (F0), sobre 16.320 filas de run_regions.csv. Reemplaza un
# supuesto anterior (fraccion DECLARADA en el manifiesto de campana + piso
# arbitrario 0.35): la fraccion declarada es la solicitada, no la alcanzada,
# y difieren sobre todo en F6 -- CPU real 0.267 vs. declarada 0.0; GPU real
# 0.149 vs. declarada 0.0 -- porque el hardware nunca llega al reloj minimo
# nominal. REF se mapea a 1.0 porque su fraccion real medida (CPU 0.994, GPU
# 1.0) coincide con F0 dentro del error de medicion, consistente con la
# verificacion previa de que el gobernador nativo corre al maximo bajo carga
# (razon de tiempo REF/F0 mediana 1,0001 en CPU, n=136).
CPU_FREQUENCY_FRACTION: Mapping[str, float] = {
    "REF": 1.0, "F0": 1.0, "F1": 0.881, "F2": 0.761, "F3": 0.631,
    "F4": 0.500, "F5": 0.386, "F6": 0.267,
}
GPU_FREQUENCY_FRACTION: Mapping[str, float] = {
    "REF": 1.0, "F0": 1.0, "F1": 0.862, "F2": 0.713, "F3": 0.574,
    "F4": 0.436, "F5": 0.287, "F6": 0.149,
}
CURVE_PARAMS: tuple[str, ...] = ("ta", "tb", "tc", "ea", "eb", "eg", "eh")
CURVE_STATIC_FEATURES: tuple[str, ...] = (
    "operation", "resource_state", "device", "log10_n",
    "flops_per_dispatch_analytic", "log10_flops_per_dispatch",
    "logical_bytes_per_dispatch", "log10_logical_bytes",
    "arithmetic_intensity_analytic",
)


def _relative_frequency(level: str, device: str = "cpu") -> float:
    table = GPU_FREQUENCY_FRACTION if str(device) == "gpu" else CPU_FREQUENCY_FRACTION
    fraction = table.get(str(level))
    if fraction is None:
        raise DVFSContractError(f"nivel de frecuencia desconocido: {level!r} ({device!r})")
    return fraction


def _device_host_frequency(frequency_action: str, device: str) -> tuple[float, float]:
    """(f_dispositivo, f_anfitrion) relativos, medidos en [minimo real, 1.0].

    El anfitrion de una accion GPU es CPU (controla el lanzamiento); su
    fraccion siempre sale de la tabla de CPU aunque el dispositivo que
    ejecuta sea GPU.
    """
    parts = str(frequency_action).split(":")
    if str(device) == "gpu":
        host_level, device_level = parts[1], parts[2]
        return _relative_frequency(device_level, "gpu"), _relative_frequency(host_level, "cpu")
    device_level = parts[1]
    value = _relative_frequency(device_level, "cpu")
    return value, value


def _fit_group_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Una fila por `decision_group_id` con los 7 parametros de la curva.

    t(f) = ta + tb/f_dev + tc/f_host ; E(f) = ea + eb/f_dev + eg*f_dev + eh/f_host.
    Ajuste por minimos cuadrados sobre las acciones medidas de ese grupo, no
    una prediccion todavia -- ver PhysicalCurveCostModel para la capa que
    predice estos parametros para grupos no vistos.
    """
    records: list[dict[str, Any]] = []
    for group_id, group in frame.groupby("decision_group_id", observed=True):
        device = str(group["device"].iloc[0])
        freqs = [
            _device_host_frequency(action, device)
            for action in group["frequency_action"].astype(str)
        ]
        f_dev = np.array([f[0] for f in freqs], dtype=float)
        f_host = np.array([f[1] for f in freqs], dtype=float)
        if len(np.unique(f_dev)) < 2 and len(np.unique(f_host)) < 2:
            continue
        t = group["time_s"].to_numpy(dtype=float)
        e = group["energy_j"].to_numpy(dtype=float)
        a_t = np.column_stack([np.ones_like(f_dev), 1.0 / f_dev, 1.0 / f_host])
        a_e = np.column_stack([np.ones_like(f_dev), 1.0 / f_dev, f_dev, 1.0 / f_host])
        coef_t, *_ = np.linalg.lstsq(a_t, t, rcond=None)
        coef_e, *_ = np.linalg.lstsq(a_e, e, rcond=None)
        row = group.iloc[0]
        record = {"decision_group_id": group_id, "config_id": str(row["config_id"])}
        for column in CURVE_STATIC_FEATURES:
            record[column] = row[column]
        record.update(zip(("ta", "tb", "tc"), (float(v) for v in coef_t)))
        record.update(zip(("ea", "eb", "eg", "eh"), (float(v) for v in coef_e)))
        records.append(record)
    return pd.DataFrame(records)


def _curve_param_pipeline(seed: int):
    from sklearn.pipeline import Pipeline

    categorical = ["operation", "resource_state", "device"]
    numeric = [c for c in CURVE_STATIC_FEATURES if c not in categorical]
    return Pipeline([
        ("preprocessor", _preprocessor(categorical, numeric, scale=True)),
        # Ridge es el sub-modelo de esta familia: verificado que domina a
        # RandomForest en el mismo experimento que motiva esta clase (9,89%
        # vs 9,42% de ahorro en gpu_ready, ver 6.5-bis del plan) y es mucho
        # mas barato de ajustar 7 veces por pliegue de calibracion.
        ("model", _base_estimator("ridge", seed)),
    ])


class PhysicalCurveCostModel:
    """Predice los 7 parámetros de la curva física, no 40 costos por acción.

    Hallazgo experimental (plan de reformulación, sección 6.5-bis): la misma
    evaluación, mismos datos y pliegues, sin compuerta, muestra que predecir
    los parámetros de `t(f)=ta+tb/f_dev+tc/f_host` y su análogo de energía
    -- en vez de 40 acciones categóricas sin relación entre sí -- multiplica
    por 2,5 el ahorro capturado en `gpu_ready` (3,89 % -> 9,89 % con Ridge).
    El ajuste de la forma física por `config_id` da R² mediano 0,94-0,98 en
    tiempo y 0,90-0,998 en energía.
    """

    def __init__(self, seed: int = FROZEN_SEED) -> None:
        self.seed = seed
        self._models: dict[str, Any] = {}
        self._fallback: dict[str, float] = {}

    def fit_curves(self, train: pd.DataFrame) -> "PhysicalCurveCostModel":
        curves = _fit_group_curves(train)
        if curves.empty:
            raise DVFSContractError(
                "no hay suficientes acciones por grupo para ajustar la curva fisica",
            )
        for param in CURVE_PARAMS:
            self._fallback[param] = float(curves[param].median())
            model = _curve_param_pipeline(self.seed)
            model.fit(curves[list(CURVE_STATIC_FEATURES)], curves[param])
            self._models[param] = model
        return self

    def _predict_params(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        static = x[list(CURVE_STATIC_FEATURES)]
        return {
            param: (
                self._models[param].predict(static) if param in self._models
                else np.full(len(x), self._fallback.get(param, 0.0))
            )
            for param in CURVE_PARAMS
        }

    def predict_log_ratio(self, x: pd.DataFrame, kind: str) -> np.ndarray:
        """log(costo(accion)/costo(REF)) reconstruido de forma analitica.

        No necesita el costo REF medido de la fila: REF tiene frecuencia
        relativa 1.0 por construccion (ver `_relative_frequency`), asi que
        `costo(REF)` sale de evaluar la misma curva predicha en f=1.0 -- es
        autoconsistente, no una segunda cantidad a predecir.
        """
        params = self._predict_params(x)
        freqs = [
            _device_host_frequency(action, device)
            for action, device in zip(x["frequency_action"].astype(str), x["device"].astype(str))
        ]
        f_dev = np.array([f[0] for f in freqs], dtype=float)
        f_host = np.array([f[1] for f in freqs], dtype=float)
        if kind == "time":
            value = params["ta"] + params["tb"] / f_dev + params["tc"] / f_host
            value_ref = params["ta"] + params["tb"] + params["tc"]
        elif kind == "energy":
            value = params["ea"] + params["eb"] / f_dev + params["eg"] * f_dev + params["eh"] / f_host
            value_ref = params["ea"] + params["eb"] + params["eg"] + params["eh"]
        else:  # pragma: no cover
            raise DVFSContractError(f"kind desconocido: {kind!r}")
        value = np.clip(value, 1e-12, None)
        value_ref = np.clip(value_ref, 1e-12, None)
        log_ratio = np.log(value / value_ref)
        # Salvaguarda numerica, no un resultado fisico: verificado con datos
        # reales que un grupo de calibracion fuera de muestra puede hacer que
        # el regresor de parametros extrapole un valor absurdo para un
        # config atipico (ej. cholesky_N256 en gpu_ready con reloj de host
        # F6), que al dividirse entre una fraccion de frecuencia chica
        # (minimo real 0.149 en GPU) explota el log-ratio. El rango medido
        # REAL de log_energy_ratio/log_time_ratio en las 40 acciones x 68
        # config_id del catalogo es factor 5,67x (energia) y 7,03x (tiempo);
        # un factor 8x por eje (energia y tiempo se recortan por separado y
        # el EDP los multiplica) ya es generoso frente a lo observado. Un
        # log-ratio fuera de ese rango es un artefacto de la regresion de
        # parametros, no una senal real, y se recorta.
        return np.clip(log_ratio, -math.log(8.0), math.log(8.0))


class _CurveEnergyModel:
    def __init__(self, curve: PhysicalCurveCostModel) -> None:
        self._curve = curve

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._curve.predict_log_ratio(x, "energy")


class _CurveTimeModel:
    def __init__(self, curve: PhysicalCurveCostModel) -> None:
        self._curve = curve

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._curve.predict_log_ratio(x, "time")


def _ref_action(device: str) -> str:
    return "cpu:REF" if device == "cpu" else "gpu:REF:REF"


def _eligible_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "config_id", "operation", "size", "device", "action_id", "region",
        "energy_mean", "time_mean", "edp_mean",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise DVFSContractError(f"candidate_summary: faltan columnas {missing}")
    work = candidates.copy()
    if "eligible_repetitions" in work:
        work = work[_boolean_mask(work["eligible_repetitions"])]
    for column in ("energy_mean", "time_mean", "edp_mean"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work[
        np.isfinite(work["energy_mean"])
        & np.isfinite(work["time_mean"])
        & np.isfinite(work["edp_mean"])
        & (work["energy_mean"] > 0)
        & (work["time_mean"] > 0)
        & (work["edp_mean"] > 0)
    ]
    duplicate = work.duplicated(["config_id", "action_id", "region"])
    if duplicate.any():
        offenders = work.loc[duplicate, "config_id"].astype(str).unique()[:5]
        raise DVFSContractError(f"acciones duplicadas: {list(offenders)}")
    return work


def build_dvfs_dataset(candidates: pd.DataFrame) -> pd.DataFrame:
    """Filas de acción relevantes tras elegir dispositivo a REF.

    Produce una fila por ``config_id × resource_state × acción`` del
    dispositivo que gana a REF en ese estado. Repetir las acciones entre
    estados no aumenta el tamaño efectivo: los pliegues agrupan siempre el
    ``config_id`` completo.
    """
    work = _eligible_candidates(candidates)
    records: list[dict[str, Any]] = []
    for config_id, group in work.groupby("config_id", observed=True):
        sample = group.iloc[0]
        operation, size = str(sample["operation"]), int(sample["size"])
        descriptors = _static_descriptors(operation, size)
        for state, (cpu_region, gpu_region) in RESOURCE_STATES.items():
            regions = {"cpu": cpu_region, "gpu": gpu_region}
            refs: dict[str, pd.Series] = {}
            for device in ("cpu", "gpu"):
                rows = group[
                    (group["device"] == device)
                    & (group["region"] == regions[device])
                    & (group["action_id"] == _ref_action(device))
                ]
                if len(rows) != 1:
                    refs = {}
                    break
                refs[device] = rows.iloc[0]
            if not refs:
                continue
            device = min(refs, key=lambda item: float(refs[item]["edp_mean"]))
            region = regions[device]
            reference = refs[device]
            ref_energy_j = float(reference["energy_mean"])
            ref_time_s = float(reference["time_mean"])
            actions = group[(group["device"] == device) & (group["region"] == region)]
            for row in actions.to_dict("records"):
                action = str(row["action_id"])
                energy_j = float(row["energy_mean"])
                time_s = float(row["time_mean"])
                records.append({
                    "config_id": str(config_id),
                    "decision_group_id": f"{config_id}:{state}",
                    "operation": operation,
                    "size": size,
                    **descriptors,
                    "resource_state": state,
                    "device": device,
                    "region": region,
                    "frequency_action": action,
                    "operation_frequency_action": f"{operation}:{action}",
                    "reference_action": str(reference["action_id"]),
                    "energy_j": energy_j,
                    "time_s": time_s,
                    "edp_js": float(row["edp_mean"]),
                    "ref_energy_j": ref_energy_j,
                    "ref_time_s": ref_time_s,
                    # Target = desvio log respecto a REF, no la magnitud absoluta.
                    # REF ya se mide sin error dentro del propio grupo de
                    # decision; predecir el desvio evita que el error absoluto
                    # de un config con EDP diminuto (p.ej. axpy N pequeno,
                    # EDP~1e-8) se traduzca en errores porcentuales de miles
                    # de veces cuando se divide por una magnitud casi nula.
                    "log_energy_ratio": math.log(energy_j) - math.log(ref_energy_j),
                    "log_time_ratio": math.log(time_s) - math.log(ref_time_s),
                })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise DVFSContractError("no hay estados con acciones CPU/GPU REF completas")
    for group_id, group in frame.groupby("decision_group_id", observed=True):
        if int((group["frequency_action"] == group["reference_action"]).sum()) != 1:
            raise DVFSContractError(f"{group_id}: REF ausente o duplicada")
        if group["device"].nunique() != 1 or group["region"].nunique() != 1:
            raise DVFSContractError(f"{group_id}: mezcla dispositivos o regiones")
    return frame.sort_values(
        ["operation", "size", "resource_state", "frequency_action"],
        kind="mergesort",
    ).reset_index(drop=True)


def configuration_frame(dvfs: pd.DataFrame) -> pd.DataFrame:
    """Una fila por configuración para construir pliegues sin fuga."""
    return dvfs[["config_id", "operation", "size"]].drop_duplicates().reset_index(drop=True)


def _fit_pair(train: pd.DataFrame, family: str, *, seed: int) -> tuple[Any, Any]:
    if family == "power_law":
        energy, time_model = PowerLawCostModel(), PowerLawCostModel()
        x = train[list(DVFS_FEATURES)]
        energy.fit(x, train["log_energy_ratio"].to_numpy(dtype=float))
        time_model.fit(x, train["log_time_ratio"].to_numpy(dtype=float))
        return energy, time_model

    if family == "curve_physical":
        curve = PhysicalCurveCostModel(seed=seed).fit_curves(train)
        return _CurveEnergyModel(curve), _CurveTimeModel(curve)

    from sklearn.pipeline import Pipeline

    categorical = [
        "operation", "resource_state", "device", "region",
        "frequency_action", "operation_frequency_action",
    ]
    numeric = [column for column in DVFS_FEATURES if column not in categorical]
    scale = family in ("ridge", "elasticnet", "huber")

    def pipeline():
        return Pipeline([
            ("preprocessor", _preprocessor(categorical, numeric, scale=scale)),
            ("model", _base_estimator(family, seed, _DVFS_DEPTH_OVERRIDE.get(family))),
        ])

    energy = pipeline()
    time_model = pipeline()
    x = train[list(DVFS_FEATURES)]
    energy.fit(x, train["log_energy_ratio"].to_numpy(dtype=float))
    time_model.fit(x, train["log_time_ratio"].to_numpy(dtype=float))
    return energy, time_model


def fit_cost_models(
    train: pd.DataFrame, family: str, *, seed: int = FROZEN_SEED, calibration_splits: int = 3,
) -> CostModels:
    """Ajusta tiempo/energía y calibra error EDP fuera de muestra por grupo."""
    if family not in DVFS_FAMILIES:
        raise DVFSContractError(f"familia desconocida: {family}")
    from sklearn.model_selection import GroupKFold

    configs = train["config_id"].astype(str).to_numpy()
    unique = len(set(configs))
    # Umbral de tamaño por operacion, fijado UNA vez con el train completo del
    # pliegue externo (no con cada pliegue interno de calibracion) para que
    # todas las particiones internas usen el mismo corte. No es fuga: no usa
    # ninguna etiqueta de prueba, solo la columna `size`.
    size_thresholds = _size_regimes(train)
    errors: list[float] = []
    contextual_errors: dict[tuple[str, str, str], list[float]] = {}
    splits = min(calibration_splits, unique)
    if splits >= 2:
        for train_idx, test_idx in GroupKFold(n_splits=splits).split(train, groups=configs):
            inner_train, inner_test = train.iloc[train_idx], train.iloc[test_idx]
            e_model, t_model = _fit_pair(inner_train, family, seed=seed)
            x = inner_test[list(DVFS_FEATURES)]
            pred_energy = inner_test["ref_energy_j"].to_numpy(dtype=float) * np.exp(e_model.predict(x))
            pred_time = inner_test["ref_time_s"].to_numpy(dtype=float) * np.exp(t_model.predict(x))
            predicted = pred_energy * pred_time
            actual = inner_test["edp_js"].to_numpy(dtype=float)
            fold_errors = 100.0 * np.abs(predicted / actual - 1.0)
            errors.extend(fold_errors.tolist())
            for state, device, operation, size, error in zip(
                inner_test["resource_state"].astype(str),
                inner_test["device"].astype(str),
                inner_test["operation"].astype(str),
                inner_test["size"], fold_errors,
            ):
                regime = _size_regime(operation, size, size_thresholds)
                contextual_errors.setdefault((state, device, regime), []).append(float(error))
    energy, time_model = _fit_pair(train, family, seed=seed)
    if errors:
        uncertainty = float(np.quantile(np.asarray(errors, dtype=float), 0.95))
    else:
        x = train[list(DVFS_FEATURES)]
        pred_energy = train["ref_energy_j"].to_numpy(dtype=float) * np.exp(energy.predict(x))
        pred_time = train["ref_time_s"].to_numpy(dtype=float) * np.exp(time_model.predict(x))
        predicted = pred_energy * pred_time
        actual = train["edp_js"].to_numpy(dtype=float)
        uncertainty = float(np.max(100.0 * np.abs(predicted / actual - 1.0)))
    by_context = {
        key: float(np.quantile(np.asarray(values, dtype=float), 0.95))
        for key, values in contextual_errors.items() if values
    }
    return CostModels(
        energy=energy, time=time_model, uncertainty_pct=uncertainty,
        uncertainty_pct_by_context=by_context, size_thresholds=size_thresholds,
    )


def predict_costs(models: CostModels, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    x = out[list(DVFS_FEATURES)]
    out["pred_energy_j"] = out["ref_energy_j"].to_numpy(dtype=float) * np.exp(models.energy.predict(x))
    out["pred_time_s"] = out["ref_time_s"].to_numpy(dtype=float) * np.exp(models.time.predict(x))
    out["pred_edp_js"] = out["pred_energy_j"] * out["pred_time_s"]
    out["size_regime"] = [
        _size_regime(operation, size, models.size_thresholds)
        for operation, size in zip(out["operation"].astype(str), out["size"])
    ]
    return out


def _net_edp(energy: np.ndarray, time_s: np.ndarray, switched: np.ndarray,
             overhead_energy_j: float, overhead_time_s: float) -> np.ndarray:
    return (
        energy + switched.astype(float) * overhead_energy_j
    ) * (
        time_s + switched.astype(float) * overhead_time_s
    )


def choose_actions(
    predicted: pd.DataFrame,
    *,
    model_uncertainty_pct: float,
    model_uncertainty_pct_by_context: Mapping[tuple[str, str, str], float] | None = None,
    overhead_energy_j: float = 0.0,
    overhead_time_s: float = 0.0,
) -> pd.DataFrame:
    """Elige acción o se abstiene conservando REF.

    El conjunto equivalente usa el máximo entre piso de ruido regional y el
    error p95 fuera de muestra del modelo. El overhead se añade únicamente a
    acciones distintas de REF, tanto para la política como para el oráculo.
    """
    if overhead_energy_j < 0 or overhead_time_s < 0:
        raise DVFSContractError("el overhead no puede ser negativo")
    records: list[dict[str, Any]] = []
    for group_id, group in predicted.groupby("decision_group_id", observed=True):
        group = group.copy()
        reference_action = str(group["reference_action"].iloc[0])
        switched = group["frequency_action"].astype(str).to_numpy() != reference_action
        group["pred_net_edp_js"] = _net_edp(
            group["pred_energy_j"].to_numpy(dtype=float),
            group["pred_time_s"].to_numpy(dtype=float), switched,
            overhead_energy_j, overhead_time_s,
        )
        group["actual_net_edp_js"] = _net_edp(
            group["energy_j"].to_numpy(dtype=float),
            group["time_s"].to_numpy(dtype=float), switched,
            overhead_energy_j, overhead_time_s,
        )
        best_pred = float(group["pred_net_edp_js"].min())
        region = str(group["region"].iloc[0])
        # size_regime lo agrega predict_costs; si el llamador arma el grupo a
        # mano (pruebas), se asume "large" -- el fallback de _size_regime.
        size_regime = str(group["size_regime"].iloc[0]) if "size_regime" in group.columns else "large"
        context = (
            str(group["resource_state"].iloc[0]), str(group["device"].iloc[0]), size_regime,
        )
        contextual = (model_uncertainty_pct_by_context or {}).get(
            context, model_uncertainty_pct,
        )
        uncertainty = max(float(REGION_NOISE_PCT[region]), float(contextual))
        equivalent = group[group["pred_net_edp_js"] <= best_pred * (1.0 + uncertainty / 100.0)]
        if reference_action in set(equivalent["frequency_action"].astype(str)):
            selected_action, abstained = reference_action, True
        else:
            selected_action = str(group.loc[group["pred_net_edp_js"].idxmin(), "frequency_action"])
            abstained = False
        selected = group[group["frequency_action"].astype(str) == selected_action].iloc[0]
        reference = group[group["frequency_action"].astype(str) == reference_action].iloc[0]
        oracle = group.loc[group["actual_net_edp_js"].idxmin()]
        records.append({
            "decision_group_id": str(group_id),
            "config_id": str(group["config_id"].iloc[0]),
            "operation": str(group["operation"].iloc[0]),
            "size": int(group["size"].iloc[0]),
            "resource_state": str(group["resource_state"].iloc[0]),
            "device": str(group["device"].iloc[0]),
            "region": region,
            "selected_action": selected_action,
            "reference_action": reference_action,
            "oracle_action": str(oracle["frequency_action"]),
            "equivalent_actions": "|".join(sorted(equivalent["frequency_action"].astype(str))),
            "abstained": bool(abstained),
            "combined_uncertainty_pct": uncertainty,
            "selected_edp_js": float(selected["actual_net_edp_js"]),
            "reference_edp_js": float(reference["actual_net_edp_js"]),
            "oracle_edp_js": float(oracle["actual_net_edp_js"]),
        })
    return pd.DataFrame(records)


def _constant_table(train: pd.DataFrame, overhead_energy_j: float, overhead_time_s: float) -> dict[tuple[str, str], str]:
    table: dict[tuple[str, str], str] = {}
    for (state, device), group in train.groupby(["resource_state", "device"], observed=True):
        common = set.intersection(*(
            set(part["frequency_action"].astype(str))
            for _, part in group.groupby("decision_group_id", observed=True)
        ))
        if not common:
            continue
        reference = _ref_action(str(device))
        totals: dict[str, float] = {}
        for action in common:
            rows = group[group["frequency_action"].astype(str) == action]
            switched = np.full(len(rows), action != reference)
            totals[action] = float(_net_edp(
                rows["energy_j"].to_numpy(dtype=float), rows["time_s"].to_numpy(dtype=float),
                switched, overhead_energy_j, overhead_time_s,
            ).sum())
        table[(str(state), str(device))] = min(totals, key=totals.get)
    return table


def _fixed_decisions(test: pd.DataFrame, mode: str, *, constant_table: Mapping[tuple[str, str], str] | None,
                     overhead_energy_j: float, overhead_time_s: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_id, group in test.groupby("decision_group_id", observed=True):
        reference_action = str(group["reference_action"].iloc[0])
        switched = group["frequency_action"].astype(str).to_numpy() != reference_action
        group = group.copy()
        group["actual_net_edp_js"] = _net_edp(
            group["energy_j"].to_numpy(dtype=float), group["time_s"].to_numpy(dtype=float),
            switched, overhead_energy_j, overhead_time_s,
        )
        oracle = group.loc[group["actual_net_edp_js"].idxmin()]
        if mode == "oracle":
            selected_action = str(oracle["frequency_action"])
        elif mode == "constant":
            key = (str(group["resource_state"].iloc[0]), str(group["device"].iloc[0]))
            selected_action = str((constant_table or {}).get(key, reference_action))
            if selected_action not in set(group["frequency_action"].astype(str)):
                selected_action = reference_action
        else:
            selected_action = reference_action
        selected = group[group["frequency_action"].astype(str) == selected_action].iloc[0]
        reference = group[group["frequency_action"].astype(str) == reference_action].iloc[0]
        records.append({
            "decision_group_id": str(group_id), "config_id": str(group["config_id"].iloc[0]),
            "operation": str(group["operation"].iloc[0]), "size": int(group["size"].iloc[0]),
            "resource_state": str(group["resource_state"].iloc[0]),
            "device": str(group["device"].iloc[0]), "region": str(group["region"].iloc[0]),
            "selected_action": selected_action, "reference_action": reference_action,
            "oracle_action": str(oracle["frequency_action"]),
            "equivalent_actions": selected_action, "abstained": selected_action == reference_action,
            "combined_uncertainty_pct": np.nan,
            "selected_edp_js": float(selected["actual_net_edp_js"]),
            "reference_edp_js": float(reference["actual_net_edp_js"]),
            "oracle_edp_js": float(oracle["actual_net_edp_js"]),
        })
    return pd.DataFrame(records)


def _metrics(decisions: pd.DataFrame, *, fold: str, method: str, name: str) -> list[dict[str, Any]]:
    regime = "extrapolation" if fold.startswith("extrapolation") else "interpolation"
    records: list[dict[str, Any]] = []
    for state in (*RESOURCE_STATES, "all"):
        group = decisions if state == "all" else decisions[decisions["resource_state"] == state]
        if group.empty:
            continue
        selected = float(group["selected_edp_js"].sum())
        oracle = float(group["oracle_edp_js"].sum())
        reference = float(group["reference_edp_js"].sum())
        records.append({
            "fold": fold, "regime": regime, "resource_state": state,
            "method": method, "name": name, "n": int(len(group)),
            "edp_sum_js": selected, "edp_sum_ratio_vs_oracle": selected / oracle,
            "edp_sum_ratio_vs_ref": selected / reference,
            "savings_vs_ref_pct": 100.0 * (reference - selected) / reference,
            "oracle_savings_captured_pct": (
                100.0 * (reference - selected) / (reference - oracle)
                if reference > oracle else 100.0
            ),
            "action_accuracy": float((group["selected_action"] == group["oracle_action"]).mean()),
            "abstention_rate": float(group["abstained"].mean()),
            "uncertainty_pct": float(group["combined_uncertainty_pct"].max())
            if group["combined_uncertainty_pct"].notna().any() else np.nan,
        })
    return records


def evaluate_dvfs(
    dvfs: pd.DataFrame,
    *,
    families: Sequence[str] = DVFS_FAMILIES,
    seed: int = FROZEN_SEED,
    overhead_energy_j: float = 0.0,
    overhead_time_s: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluación agrupada por tamaño; devuelve métricas y decisiones modelo."""
    configs = configuration_frame(dvfs)
    folds = [*interpolation_folds(configs), *extrapolation_folds(configs)]
    records: list[dict[str, Any]] = []
    model_decisions: list[pd.DataFrame] = []
    for fold, train_configs, test_configs in folds:
        assert_no_config_leak(train_configs, test_configs)
        train_ids = set(train_configs["config_id"].astype(str))
        test_ids = set(test_configs["config_id"].astype(str))
        train = dvfs[dvfs["config_id"].astype(str).isin(train_ids)].copy()
        test = dvfs[dvfs["config_id"].astype(str).isin(test_ids)].copy()
        constant = _constant_table(train, overhead_energy_j, overhead_time_s)
        for mode, name in (("ref", "always_ref"), ("constant", "best_constant_train"), ("oracle", "oracle")):
            decisions = _fixed_decisions(
                test, mode, constant_table=constant,
                overhead_energy_j=overhead_energy_j, overhead_time_s=overhead_time_s,
            )
            records.extend(_metrics(decisions, fold=fold, method="baseline", name=name))
        for family in families:
            models = fit_cost_models(train, family, seed=seed)
            predicted = predict_costs(models, test)
            decisions = choose_actions(
                predicted, model_uncertainty_pct=models.uncertainty_pct,
                model_uncertainty_pct_by_context=models.uncertainty_pct_by_context,
                overhead_energy_j=overhead_energy_j, overhead_time_s=overhead_time_s,
            )
            decisions.insert(0, "family", family)
            decisions.insert(0, "fold", fold)
            model_decisions.append(decisions)
            records.extend(_metrics(decisions, fold=fold, method="model", name=family))
    return pd.DataFrame(records), pd.concat(model_decisions, ignore_index=True)


def select_dvfs_policy(results: pd.DataFrame) -> dict[str, Any]:
    """Congela una familia y una baseline globales usando extrapolación/all."""
    scope = results[(results["regime"] == "extrapolation") & (results["resource_state"] == "all")]
    models = scope[scope["method"] == "model"].groupby("name", observed=True)["edp_sum_ratio_vs_oracle"].mean()
    baselines = scope[(scope["method"] == "baseline") & (scope["name"] != "oracle")].groupby(
        "name", observed=True,
    )["edp_sum_ratio_vs_oracle"].mean()
    if models.empty or baselines.empty:
        raise DVFSContractError("faltan modelos o baselines en extrapolación")
    family, baseline = str(models.idxmin()), str(baselines.idxmin())
    model_ratio, baseline_ratio = float(models[family]), float(baselines[baseline])
    improvement = 100.0 * (baseline_ratio - model_ratio) / baseline_ratio
    return {
        "family": family, "baseline": baseline,
        "model_ratio": model_ratio, "baseline_ratio": baseline_ratio,
        "model_improvement_pct": improvement,
        "adopt_model": bool(improvement > 3.11),
    }


def run_dvfs_analysis(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    seed: int = FROZEN_SEED,
    overhead_energy_j: float = 0.0,
    overhead_time_s: float = 0.0,
) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / "r3_dvfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = dataset_dir / "candidate_summary.csv"
    if not source.is_file():
        raise DVFSContractError(f"falta {source}")
    candidates = pd.read_csv(source, low_memory=False)
    dvfs = build_dvfs_dataset(candidates)
    results, decisions = evaluate_dvfs(
        dvfs, seed=seed, overhead_energy_j=overhead_energy_j, overhead_time_s=overhead_time_s,
    )
    selection = select_dvfs_policy(results)
    paths = {
        "dataset": output_dir / "dvfs_dataset.csv",
        "results": output_dir / "dvfs_results.csv",
        "decisions": output_dir / "dvfs_model_decisions.csv",
        "summary": output_dir / "dvfs_summary.json",
    }
    dvfs.to_csv(paths["dataset"], index=False)
    results.to_csv(paths["results"], index=False)
    decisions.to_csv(paths["decisions"], index=False)
    paths["summary"].write_text(json.dumps({
        "effective_config_count": int(dvfs["config_id"].nunique()),
        "decision_group_count": int(dvfs["decision_group_id"].nunique()),
        "overhead_energy_j": overhead_energy_j,
        "overhead_time_s": overhead_time_s,
        "overhead_status": "measured" if overhead_energy_j or overhead_time_s else "not_measured_upper_bound",
        "selection": selection,
        "contract": {
            "device_selected_at_ref_before_dvfs": True,
            "time_and_energy_predicted_separately": True,
            "config_id_is_independent_unit": True,
            "combined_uncertainty_includes_model_p95_and_region_noise": True,
            "ref_preferred_inside_equivalent_set": True,
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths
