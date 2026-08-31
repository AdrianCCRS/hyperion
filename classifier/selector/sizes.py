"""Particiones por tamano y baselines de dispositivo (secciones 8.1 y 9).

Todo en este modulo opera sobre el dataset compacto de `compact.py`: una fila
por `config_id` x `resource_state`, con el target
``y = log(EDP_GPU_REF / EDP_CPU_REF)``.

Dos reglas que gobiernan el modulo entero:

1. **La unidad de particion es `config_id`.** Ningun `config_id` puede
   aparecer en entrenamiento y prueba a la vez, aunque sus filas pertenezcan a
   estados de recurso distintos: los tres estados comparten las mismas
   mediciones fisicas subyacentes.
2. **Toda baseline con parametros se estima solo con entrenamiento.** Los
   umbrales de tamano, de intensidad aritmetica y la tabla de cruce por
   operacion se ajustan dentro de `fit`, nunca sobre el conjunto de prueba.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence
import math

import numpy as np
import pandas as pd

from .compact import NOISE_FLOOR_PCT, CompactDatasetError


# --------------------------------------------------------------------------
# Particiones por tamano (seccion 8.1)
# --------------------------------------------------------------------------


def _sizes_by_operation(frame: pd.DataFrame) -> dict[str, list[int]]:
    return {
        str(operation): sorted(int(size) for size in group["size"].unique())
        for operation, group in frame.groupby("operation", observed=True)
    }


def _split_by_configs(frame: pd.DataFrame, test_configs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_test = frame["config_id"].astype(str).isin(test_configs)
    return frame[~is_test].copy(), frame[is_test].copy()


def assert_no_config_leak(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = set(train["config_id"].astype(str)) & set(test["config_id"].astype(str))
    if shared:
        raise CompactDatasetError(f"fuga de config_id entre train y test: {sorted(shared)[:5]}")


def interpolation_folds(
    frame: pd.DataFrame, *, n_folds: int = 3,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Tamanos INTERNOS retenidos, con extremos siempre en entrenamiento.

    Para cada operacion se reservan el tamano menor y el mayor en
    entrenamiento y los intermedios se reparten en `n_folds` pliegues de forma
    intercalada (posicion modulo `n_folds`), no en bloques contiguos: repartir
    por bloques dejaria un pliegue con solo los tamanos pequenos y otro con
    solo los grandes, que es extrapolacion disfrazada de interpolacion.

    Garantiza por construccion que cada tamano de prueba tiene un tamano menor
    y uno mayor de la misma operacion en entrenamiento, que es la definicion de
    interpolacion del plan.
    """
    if n_folds < 2:
        raise CompactDatasetError(f"n_folds debe ser >= 2: {n_folds}")
    sizes = _sizes_by_operation(frame)
    fold_sizes: list[dict[str, list[int]]] = [{} for _ in range(n_folds)]
    for operation, ordered in sizes.items():
        interior = ordered[1:-1]
        for position, size in enumerate(interior):
            fold_sizes[position % n_folds].setdefault(operation, []).append(size)
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for index, assignment in enumerate(fold_sizes):
        test_configs = {
            str(row["config_id"])
            for row in frame.to_dict("records")
            if int(row["size"]) in assignment.get(str(row["operation"]), [])
        }
        if not test_configs:
            continue
        train, test = _split_by_configs(frame, test_configs)
        assert_no_config_leak(train, test)
        folds.append((f"interpolation_fold{index}", train, test))
    return folds


def extrapolation_folds(
    frame: pd.DataFrame, *, n_largest: int = 2,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Entrenar con los tamanos menores, probar en el extremo superior.

    Esta es la prueba que justifica (o no) la campana de tamanos grandes: si
    un modelo entrenado sin los tamanos mayores acierta en ellos, medir
    tamanos aun mayores aporta poco; si falla, la campana esta justificada.

    Se genera un pliegue por valor de `1..n_largest` para poder observar como
    se degrada la extrapolacion al alejarse del rango de entrenamiento.
    """
    if n_largest < 1:
        raise CompactDatasetError(f"n_largest debe ser >= 1: {n_largest}")
    sizes = _sizes_by_operation(frame)
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for depth in range(1, n_largest + 1):
        held_out = {
            operation: set(ordered[-depth:]) for operation, ordered in sizes.items()
        }
        test_configs = {
            str(row["config_id"])
            for row in frame.to_dict("records")
            if int(row["size"]) in held_out.get(str(row["operation"]), set())
        }
        train, test = _split_by_configs(frame, test_configs)
        if train.empty or test.empty:
            continue
        assert_no_config_leak(train, test)
        folds.append((f"extrapolation_top{depth}", train, test))
    return folds


# --------------------------------------------------------------------------
# Baselines de dispositivo (seccion 9)
# --------------------------------------------------------------------------


def _edp_of(frame: pd.DataFrame, devices: Sequence[str]) -> np.ndarray:
    cpu = frame["cpu_ref_edp_js"].to_numpy(dtype=float)
    gpu = frame["gpu_ref_edp_js"].to_numpy(dtype=float)
    return np.where(np.asarray(devices) == "gpu", gpu, cpu)


def _best_constant_device(train: pd.DataFrame) -> str:
    """Dispositivo constante que minimiza el EDP total de entrenamiento.

    Se usa la SUMA de EDP y no el conteo de victorias porque una politica
    constante se paga en energia, no en votos: ganar 40 configuraciones
    baratas no compensa perder 28 caras por dos ordenes de magnitud.
    """
    return "gpu" if train["gpu_ref_edp_js"].sum() < train["cpu_ref_edp_js"].sum() else "cpu"


def _fit_threshold(train: pd.DataFrame, column: str) -> tuple[float, str]:
    """Umbral escalar sobre `column` que minimiza el EDP total de entrenamiento.

    Devuelve (umbral, lado_gpu) donde `lado_gpu` in {"above", "below"} indica
    de que lado del umbral se elige GPU. Se prueba explicitamente el sentido
    de la desigualdad en lugar de asumirlo: la intensidad aritmetica y el
    tamano no tienen por que orientarse igual.
    """
    values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("inf"), "above"
    candidates = np.unique(finite)
    midpoints = (candidates[:-1] + candidates[1:]) / 2.0 if len(candidates) > 1 else candidates
    # +-inf permiten que el umbral degenere a "siempre CPU" o "siempre GPU"
    # cuando eso es lo mejor que hace un solo corte.
    grid = np.concatenate(([-np.inf], midpoints, [np.inf]))
    best = (float("inf"), float("inf"), "above")
    for side in ("above", "below"):
        for threshold in grid:
            picks = np.where(
                (values > threshold) if side == "above" else (values < threshold),
                "gpu", "cpu",
            )
            total = float(_edp_of(train, picks).sum())
            if total < best[0]:
                best = (total, float(threshold), side)
    return best[1], best[2]


def _apply_threshold(frame: pd.DataFrame, column: str, threshold: float, side: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.where(
        (values > threshold) if side == "above" else (values < threshold), "gpu", "cpu",
    )


def _fit_operation_crossover(train: pd.DataFrame) -> dict[str, tuple[float, str]]:
    """Un umbral de tamano por operacion, ajustado solo con entrenamiento."""
    table: dict[str, tuple[float, str]] = {}
    for operation, group in train.groupby("operation", observed=True):
        table[str(operation)] = _fit_threshold(group, "log10_n")
    return table


def _stay_on_ready_device(frame: pd.DataFrame) -> np.ndarray:
    """Permanecer en el dispositivo ya preparado; CPU cuando no hay ninguno.

    `none_ready` no tiene dispositivo preparado, asi que esta baseline degenera
    ahi a "siempre CPU". Es deliberado: es la politica segura que el plan fija
    para ese estado (seccion 6.3), no un relleno arbitrario.
    """
    state = frame["resource_state"].astype(str).to_numpy()
    return np.where(state == "gpu_ready", "gpu", "cpu")


# --------------------------------------------------------------------------
# Baselines de horizonte K (enmienda 2026-08-30-A, seccion 12.4)
# --------------------------------------------------------------------------

#: Columnas de costo `cold`/`warm` que produce `horizon.build_horizon_dataset`.
#: Solo estan presentes en el dataset de horizonte, nunca en el compacto
#: (K=1) -- por eso `_fit_k_break_even_table` degrada con seguridad cuando
#: faltan en vez de fallar.
_HORIZON_COST_COLUMNS = tuple(
    f"cost_{device}_{field}"
    for device in ("cpu", "gpu")
    for field in ("e_cold", "t_cold", "e_warm", "t_warm")
)


def _interpolate_log_k_break_even(points: Sequence[tuple[float, float]], x: float) -> float:
    """Interpola ``log10(K_break_even)`` sobre ``log10(size)`` de entrenamiento.

    Los puntos con `K_break_even` infinito (sin cruce) se excluyen del ajuste:
    no aportan magnitud, solo la ausencia de una. Fuera del rango observado se
    extrapola de forma constante (`np.interp` recorta al extremo mas cercano)
    en vez de proyectar una tendencia exponencial sin respaldo.
    """
    finite = sorted((lx, k) for lx, k in points if np.isfinite(k) and k > 0)
    if not finite:
        return float("inf")
    xs = np.array([p[0] for p in finite], dtype=float)
    ys = np.log10(np.array([p[1] for p in finite], dtype=float))
    return float(10.0 ** np.interp(x, xs, ys))


def _fit_k_break_even_table(train: pd.DataFrame):
    """Tabla empirica de `K_break_even` por operacion, ajustada solo en train.

    Requiere las columnas de costo `cold`/`warm` que solo trae el dataset de
    horizonte (`horizon.build_horizon_dataset`). Sin ellas -- por ejemplo
    sobre el dataset compacto K=1 -- no hay forma de reconstruir un cruce, asi
    que se degrada a `_stay_on_ready_device` en lugar de fallar: sigue siendo
    una politica segura y valida, solo que no usa la informacion de horizonte.
    """
    if not all(column in train.columns for column in _HORIZON_COST_COLUMNS):
        return _stay_on_ready_device
    if train.empty:
        return _stay_on_ready_device
    from .horizon import switch_k_for_state

    state = str(train["resource_state"].iloc[0])
    unique = train.drop_duplicates("config_id")
    table: dict[str, list[tuple[float, float]]] = {}
    for operation, group in unique.groupby("operation", observed=True):
        points: list[tuple[float, float]] = []
        for row in group.to_dict("records"):
            cpu = {field: row[f"cost_cpu_{field}"] for field in ("e_cold", "t_cold", "e_warm", "t_warm")}
            gpu = {field: row[f"cost_gpu_{field}"] for field in ("e_cold", "t_cold", "e_warm", "t_warm")}
            k_be = switch_k_for_state(cpu, gpu, state)
            points.append((math.log10(float(row["size"])), k_be))
        table[str(operation)] = points

    def predict(test: pd.DataFrame) -> np.ndarray:
        devices = []
        for row in test.to_dict("records"):
            points = table.get(str(row["operation"]), [])
            k_pred = _interpolate_log_k_break_even(points, math.log10(float(row["size"])))
            k_row = float(row["k"]) if "k" in row and pd.notna(row.get("k")) else 1.0
            devices.append("gpu" if np.isfinite(k_pred) and k_row >= k_pred else "cpu")
        return np.array(devices, dtype=object)

    return predict


#: Nombre -> funcion ``fit(train) -> predict(test) -> devices``.
BASELINES: dict[str, Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]] = {
    "always_cpu_ref": lambda train: (lambda test: np.full(len(test), "cpu", dtype=object)),
    "always_gpu_ref": lambda train: (lambda test: np.full(len(test), "gpu", dtype=object)),
    "stay_on_ready_device": lambda train: _stay_on_ready_device,
    "best_constant_device_train": lambda train: (
        lambda test, device=_best_constant_device(train):
        np.full(len(test), device, dtype=object)
    ),
    "size_threshold_train": lambda train: (
        lambda test, fitted=_fit_threshold(train, "log10_n"):
        _apply_threshold(test, "log10_n", *fitted)
    ),
    "intensity_threshold_train": lambda train: (
        lambda test, fitted=_fit_threshold(train, "arithmetic_intensity_analytic"):
        _apply_threshold(test, "arithmetic_intensity_analytic", *fitted)
    ),
    "operation_crossover_table_train": lambda train: (
        lambda test, table=_fit_operation_crossover(train): np.array([
            _apply_threshold(
                test.iloc[[position]], "log10_n",
                *table.get(str(row["operation"]), (float("inf"), "above")),
            )[0]
            for position, row in enumerate(test.to_dict("records"))
        ], dtype=object)
    ),
    "oracle": lambda train: (
        lambda test: np.where(
            test["gpu_ref_edp_js"].to_numpy(dtype=float)
            < test["cpu_ref_edp_js"].to_numpy(dtype=float),
            "gpu", "cpu",
        )
    ),
    # -- Enmienda 2026-08-30-A, seccion 12.4 -- las tres baselines de horizonte.
    # `stay_on_ready_device_k` es identica a `stay_on_ready_device`: permanecer
    # en el dispositivo preparado no depende de K, solo del estado. Se declara
    # aparte porque semanticamente es "la baseline que la politica de la
    # seccion 2 representa realmente" bajo la formulacion de horizonte, no un
    # duplicado accidental.
    "stay_on_ready_device_k": lambda train: _stay_on_ready_device,
    "k_break_even_table_train": lambda train: _fit_k_break_even_table(train),
    # `oracle_k` reutiliza la misma comparacion que `oracle`: sobre el dataset
    # de horizonte, `cpu_ref_edp_js`/`gpu_ref_edp_js` YA son EDP_total(d, K)
    # (ver `horizon.build_horizon_dataset`), asi que "elegir el EDP total mas
    # bajo" es exactamente `argmin_d EDP_total(d, K | estado)` de la seccion
    # 12.1. En K=1 sobre el dataset compacto coincide numericamente con
    # `oracle`.
    "oracle_k": lambda train: (
        lambda test: np.where(
            test["gpu_ref_edp_js"].to_numpy(dtype=float)
            < test["cpu_ref_edp_js"].to_numpy(dtype=float),
            "gpu", "cpu",
        )
    ),
}


def evaluate_devices(test: pd.DataFrame, devices: Sequence[str]) -> dict[str, float]:
    """Metricas de una decision de dispositivo sobre `test` (seccion 10.2)."""
    devices = np.asarray(devices, dtype=object)
    truth = test["device_label"].astype(str).to_numpy()
    chosen_edp = _edp_of(test, devices)
    oracle_edp = np.minimum(
        test["cpu_ref_edp_js"].to_numpy(dtype=float),
        test["gpu_ref_edp_js"].to_numpy(dtype=float),
    )
    worst_edp = np.maximum(
        test["cpu_ref_edp_js"].to_numpy(dtype=float),
        test["gpu_ref_edp_js"].to_numpy(dtype=float),
    )
    # Regret relativo: cuantas veces mas EDP se gasta frente al oraculo. Se
    # usa la razon y no la diferencia porque los EDP absolutos abarcan mas de
    # diez ordenes de magnitud entre axpy pequeno y cholesky grande, y una
    # media de diferencias quedaria dominada por una sola configuracion.
    regret_ratio = chosen_edp / oracle_edp
    correct = devices == truth
    # Porcentaje del ahorro del oraculo capturado, medido sobre la peor
    # eleccion posible como referencia comun.
    savings_room = worst_edp - oracle_edp
    captured = np.where(savings_room > 0, (worst_edp - chosen_edp) / savings_room, 1.0)

    true_positive = int(((devices == "gpu") & (truth == "gpu")).sum())
    false_positive = int(((devices == "gpu") & (truth == "cpu")).sum())
    false_negative = int(((devices == "cpu") & (truth == "gpu")).sum())
    true_negative = int(((devices == "cpu") & (truth == "cpu")).sum())
    recall_gpu = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    recall_cpu = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else float("nan")
    precision_gpu = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    denominator = math.sqrt(
        float(true_positive + false_positive) * float(true_positive + false_negative)
        * float(true_negative + false_positive) * float(true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative) / denominator
        if denominator > 0 else float("nan")
    )
    balanced = np.nanmean([recall_gpu, recall_cpu])

    return {
        "n": int(len(test)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced),
        "mcc": float(mcc),
        "precision_migrate_gpu": float(precision_gpu),
        "recall_migrate_gpu": float(recall_gpu),
        "tp_gpu": true_positive, "fp_gpu": false_positive,
        "fn_gpu": false_negative, "tn_cpu": true_negative,
        "regret_ratio_mean": float(regret_ratio.mean()),
        "regret_ratio_median": float(np.median(regret_ratio)),
        "regret_ratio_p95": float(np.percentile(regret_ratio, 95)),
        "regret_ratio_max": float(regret_ratio.max()),
        "oracle_savings_captured_pct": float(100.0 * np.mean(captured)),
        # Agregacion por suma de EDP: mide el costo de la politica sobre el
        # conjunto, complementaria a la media de razones (seccion 10.4).
        "edp_sum_js": float(chosen_edp.sum()),
        "edp_sum_ratio_vs_oracle": float(chosen_edp.sum() / oracle_edp.sum()),
    }


def run_baselines(
    folds: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
    *,
    baselines: dict[str, Any] | None = None,
    by_resource_state: bool = True,
) -> pd.DataFrame:
    """Evalua todas las baselines sobre todos los pliegues.

    Cuando `by_resource_state` es verdadero (por defecto) cada estado se
    ajusta y evalua por separado: los tres estados tienen distribuciones de
    etiqueta muy distintas -- `cpu_ready` es constante y `gpu_ready` es la
    unica tarea con dos clases -- y promediarlos oculta esa asimetria.
    """
    baselines = baselines or BASELINES
    records: list[dict[str, Any]] = []
    for fold_name, train, test in folds:
        if by_resource_state:
            states = sorted(set(test["resource_state"].astype(str)))
            slices = [
                (state, train[train["resource_state"] == state], test[test["resource_state"] == state])
                for state in states
            ]
        else:
            slices = [("all", train, test)]
        for state, train_slice, test_slice in slices:
            if train_slice.empty or test_slice.empty:
                continue
            for name, factory in baselines.items():
                predict = factory(train_slice)
                devices = predict(test_slice)
                records.append({
                    "fold": fold_name,
                    "regime": fold_name.split("_", 1)[0],
                    "resource_state": state,
                    "baseline": name,
                    **evaluate_devices(test_slice, devices),
                })
    return pd.DataFrame(records)


def baseline_headroom_report(results: pd.DataFrame) -> pd.DataFrame:
    """Brecha descriptiva entre la mejor baseline y el oraculo.

    El CV mediano de acciones individuales se conserva solo como referencia
    de cribado. No es la incertidumbre de la perdida agregada entre politicas
    y, por tanto, esta tabla no decide por si sola si existe senal aprendible.
    Esa conclusion requiere R2: modelo y baseline evaluados en los mismos
    pliegues externos, con incertidumbre sobre su diferencia.
    """
    records: list[dict[str, Any]] = []
    for (regime, state), group in results.groupby(["regime", "resource_state"], observed=True):
        aggregated = group.groupby("baseline", observed=True).agg(
            regret_ratio_mean=("regret_ratio_mean", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            edp_sum_ratio_vs_oracle=("edp_sum_ratio_vs_oracle", "mean"),
        )
        without_oracle = aggregated.drop(index="oracle", errors="ignore")
        if without_oracle.empty:
            continue
        best = without_oracle["edp_sum_ratio_vs_oracle"].idxmin()
        oracle_headroom = float(
            100.0 * (1.0 - 1.0 / without_oracle.loc[best, "edp_sum_ratio_vs_oracle"])
        )
        pass_possible = oracle_headroom >= NOISE_FLOOR_PCT
        records.append({
            "regime": regime,
            "resource_state": state,
            "best_baseline": str(best),
            "best_baseline_edp_ratio_vs_oracle": float(
                without_oracle.loc[best, "edp_sum_ratio_vs_oracle"]
            ),
            "best_baseline_balanced_accuracy": float(
                without_oracle.loc[best, "balanced_accuracy"]
            ),
            "oracle_headroom_over_best_baseline_pct": oracle_headroom,
            "above_individual_action_cv_reference": pass_possible,
            # El oraculo es una cota superior: si ni el puede mejorar la
            # baseline por el umbral preregistrado, ningun modelo puede pasar
            # la regla bloqueante. Esto es una conclusion bajo el protocolo
            # congelado, no un intervalo estadistico sobre politicas.
            "strict_frozen_protocol_pass_possible": pass_possible,
            "inference_status": (
                "requires_r2_model_comparison" if pass_possible
                else "ruled_out_by_oracle_upper_bound_under_frozen_rule"
            ),
        })
    return pd.DataFrame(records)


def learnable_signal_report(results: pd.DataFrame) -> pd.DataFrame:
    """Alias de compatibilidad; no afirma que la senal sea aprendible."""
    return baseline_headroom_report(results)
