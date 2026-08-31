"""Target de dispositivo condicionado por `resource_state` x horizonte `K`.

Implementa la enmienda **2026-08-30-A** del protocolo congelado
(`docs/general/protocolo_congelado_confirmatorio_20260830.md`, seccion 12).

La formulacion inicial del protocolo (seccion 1) fija el target como
``y = log(EDP_GPU_REF / EDP_CPU_REF)`` leyendo cada dispositivo en la region
que ve segun el estado. Esa formulacion es correcta **solo para K = 1**: mide
el despacho siguiente, no el horizonte. La politica real del agente es

```text
decision(estado, K) = argmin_d  EDP_total(d, K | estado)
```

con el termino de arranque pagado unicamente cuando el dispositivo destino no
esta inicializado en ese estado (seccion 12.1):

```text
d ya inicializado:   E_total = K * E_warm(d)
                     T_total = K * T_warm(d)
d no inicializado:   E_total = E_cold(d) + (K-1) * E_warm(d)
                     T_total = T_cold(d) + (K-1) * T_warm(d)
EDP_total(d, K)    = E_total(d, K) * T_total(d, K)
```

`K` es **entrada conocida** (seccion 12.1; seccion 7.1 del plan la enumera
entre las caracteristicas del modelo estatico como "horizonte esperado K
cuando aplique"). Su estimacion en linea es la Fase E2 y NO se implementa
aqui.

Comprobacion de consistencia que ancla el modulo: en ``K = 1`` la formula de
arriba se reduce exactamente a las regiones de `compact.RESOURCE_STATES`
(``K * warm`` con K=1 es ``warm``; ``cold + 0 * warm`` es ``cold``), de modo
que ``y(estado, 1)`` reproduce el target congelado en la seccion 1. Hay un
test que lo verifica fila a fila.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

import numpy as np
import pandas as pd

from .compact import (
    NOISE_FLOOR_PCT,
    CompactDatasetError,
    STATIC_FEATURES,
    _ref_lookup,
    ref_configurations,
)
from .dataset import _static_descriptors

#: Rejilla de horizontes congelada en la seccion 12.5. Se fija aqui para
#: impedir que se elija despues de ver los resultados.
K_GRID: tuple[int, ...] = (1, 2, 3, 5, 10, 30, 100, 1000)

#: Dispositivo ya inicializado en cada estado de recurso. `none_ready` no
#: tiene ninguno: los dos pagan arranque.
READY_DEVICE: Mapping[str, str | None] = {
    "none_ready": None,
    "cpu_ready": "cpu",
    "gpu_ready": "gpu",
}

#: Caracteristicas del modelo de horizonte: las nueve estaticas congeladas en
#: la seccion 3.1 mas el horizonte, que es entrada conocida.
HORIZON_FEATURES: tuple[str, ...] = (*STATIC_FEATURES, "k", "log10_k")


def edp_total_state(
    costs: Mapping[str, float], *, device: str, state: str, k: int,
) -> tuple[float, float, float]:
    """``(E_total, T_total, EDP_total)`` de `device` en `state` para horizonte `k`.

    `costs` debe traer ``e_cold``, ``t_cold``, ``e_warm`` y ``t_warm`` de ese
    dispositivo. El arranque se paga solo si el dispositivo no esta ya
    inicializado en el estado (seccion 12.1).
    """
    if k < 1:
        raise CompactDatasetError(f"horizonte K debe ser >= 1: {k}")
    if state not in READY_DEVICE:
        raise CompactDatasetError(f"estado de recurso desconocido: {state}")
    if device not in ("cpu", "gpu"):
        raise CompactDatasetError(f"dispositivo desconocido: {device}")
    if READY_DEVICE[state] == device:
        energy = k * float(costs["e_warm"])
        time = k * float(costs["t_warm"])
    else:
        energy = float(costs["e_cold"]) + (k - 1) * float(costs["e_warm"])
        time = float(costs["t_cold"]) + (k - 1) * float(costs["t_warm"])
    return energy, time, energy * time


def device_costs(candidates: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """`config_id` -> device -> costos ``e_cold``/``t_cold``/``e_warm``/``t_warm``.

    Solo se leen las dos acciones REF, que son las que definen el target. Los
    valores son las medias sobre repeticiones; la banda de sensibilidad se
    calcula aparte (`horizon_sensitivity`) sobre los extremos marginales.
    """
    lookup = _ref_lookup(candidates)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for config_id in ref_configurations(candidates):
        per_device: dict[str, dict[str, float]] = {}
        for device in ("cpu", "gpu"):
            cold = lookup[(config_id, device, "cold")]
            warm = lookup[(config_id, device, "warm")]
            per_device[device] = {
                "e_cold": float(cold["energy_mean"]), "t_cold": float(cold["time_mean"]),
                "e_warm": float(warm["energy_mean"]), "t_warm": float(warm["time_mean"]),
            }
        out[config_id] = per_device
    return out


def build_horizon_dataset(
    candidates: pd.DataFrame,
    *,
    resource_states: Sequence[str] = tuple(READY_DEVICE),
    k_grid: Sequence[int] = K_GRID,
) -> pd.DataFrame:
    """Una fila por `config_id` x `resource_state` x `K` con el target de horizonte.

    Columnas de decision:

    - ``y_log_edp_ratio_k`` = ``log(EDP_total(gpu,K|estado) / EDP_total(cpu,K|estado))``;
    - ``device_label`` = ``"gpu"`` si ``y < 0`` si no ``"cpu"``;
    - ``cpu_ref_edp_js`` / ``gpu_ref_edp_js``: **EDP total del horizonte**, no
      el EDP de un despacho. Conservan el nombre del dataset compacto a
      proposito, para que las ocho baselines congeladas en la seccion 6 y
      `sizes.evaluate_devices` operen sin cambios sobre esta tabla; en ``K=1``
      coinciden numericamente con las del dataset compacto.

    Ninguna de esas columnas es utilizable como entrada: todas caen bajo
    `compact.LEAKAGE_PATTERNS` y el chequeo bloqueante las rechaza.
    """
    unknown = sorted(set(resource_states) - set(READY_DEVICE))
    if unknown:
        raise CompactDatasetError(f"estados de recurso desconocidos: {unknown}")
    horizons = [int(k) for k in k_grid]
    if not horizons or min(horizons) < 1:
        raise CompactDatasetError(f"rejilla de K invalida: {list(k_grid)}")
    lookup = _ref_lookup(candidates)
    costs = device_costs(candidates)
    records: list[dict[str, Any]] = []
    for config_id, per_device in costs.items():
        sample = lookup[(config_id, "cpu", "cold")]
        operation, size = str(sample["operation"]), int(sample["size"])
        descriptors = _static_descriptors(operation, size)
        for state in resource_states:
            for k in horizons:
                cpu_e, cpu_t, cpu_edp = edp_total_state(
                    per_device["cpu"], device="cpu", state=state, k=k,
                )
                gpu_e, gpu_t, gpu_edp = edp_total_state(
                    per_device["gpu"], device="gpu", state=state, k=k,
                )
                if not (cpu_edp > 0 and gpu_edp > 0):
                    raise CompactDatasetError(
                        f"EDP total no positivo en {config_id}/{state}/K={k}"
                    )
                y = math.log(gpu_edp / cpu_edp)
                record: dict[str, Any] = {
                    "config_id": config_id,
                    "operation": operation,
                    "size": size,
                    "family": str(sample.get("family", "")),
                    "resource_state": state,
                    "k": k,
                    "log10_k": math.log10(k),
                    "ready_device": READY_DEVICE[state] or "none",
                    "y_log_edp_ratio_k": y,
                    "device_label": "gpu" if y < 0 else "cpu",
                    "cpu_ref_edp_js": cpu_edp,
                    "gpu_ref_edp_js": gpu_edp,
                    "oracle_ref_edp_js": min(cpu_edp, gpu_edp),
                    "cpu_total_energy_j": cpu_e, "cpu_total_time_s": cpu_t,
                    "gpu_total_energy_j": gpu_e, "gpu_total_time_s": gpu_t,
                    "device_margin_pct": (
                        100.0 * abs(gpu_edp - cpu_edp) / min(cpu_edp, gpu_edp)
                    ),
                }
                record.update(descriptors)
                # Costos crudos: los necesitan `oracle_k` y la tabla empirica
                # de K_break_even. Estan prohibidos como entrada del modelo
                # (prefijos `cost_*` se excluyen explicitamente en
                # `horizon_feature_columns`).
                for device in ("cpu", "gpu"):
                    for field, value in per_device[device].items():
                        record[f"cost_{device}_{field}"] = value
                record["above_noise_floor"] = int(
                    record["device_margin_pct"] >= NOISE_FLOOR_PCT
                )
                records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise CompactDatasetError("ninguna configuracion tiene las dos acciones REF completas")
    return frame.sort_values(
        ["config_id", "resource_state", "k"], kind="mergesort",
    ).reset_index(drop=True)


def horizon_feature_columns(frame: pd.DataFrame, *, with_probe: bool = False) -> list[str]:
    """Caracteristicas auditadas del modelo de horizonte, sin fuga por construccion."""
    from .compact import assert_no_leakage

    columns = [column for column in HORIZON_FEATURES if column in frame]
    if with_probe:
        columns += sorted(
            column for column in frame.columns
            if str(column).startswith("probe_") and column != "probe_device"
        )
        if "probe_device" in frame:
            columns.append("probe_device")
    forbidden = [column for column in columns if str(column).startswith("cost_")]
    if forbidden:
        raise CompactDatasetError(f"costos medidos usados como entrada: {forbidden}")
    assert_no_leakage(columns)
    return columns


# --------------------------------------------------------------------------
# K de cambio de dispositivo por estado (seccion 12.5)
# --------------------------------------------------------------------------


def _gpu_wins(cpu: Mapping[str, float], gpu: Mapping[str, float], state: str, k: int) -> bool:
    return (
        edp_total_state(gpu, device="gpu", state=state, k=k)[2]
        < edp_total_state(cpu, device="cpu", state=state, k=k)[2]
    )


def switch_k_for_state(
    cpu: Mapping[str, float],
    gpu: Mapping[str, float],
    state: str,
    *,
    horizon: int = 10 ** 9,
) -> float:
    """Menor `K` en el que GPU pasa a ganar en `state`; ``inf`` si nunca.

    Ancla analitica: cuando ``K -> inf`` el termino de arranque se diluye y la
    razon tiende a ``EDP_warm(GPU)/EDP_warm(CPU)`` **en los tres estados**. Por
    tanto, el conjunto asintotico es el mismo para `none_ready`, `cpu_ready` y
    `gpu_ready`: las configuraciones en que GPU gana en caliente. Esa identidad
    es la comprobacion de consistencia interna que exige la seccion 12.2.

    Si GPU ya gana en ``K = 1`` se devuelve ``1``: es un cruce en el sentido
    de la politica (``argmin`` cambia de dispositivo respecto a CPU), no
    necesariamente un punto de amortizacion.
    """
    warm_gpu = float(gpu["e_warm"]) * float(gpu["t_warm"])
    warm_cpu = float(cpu["e_warm"]) * float(cpu["t_warm"])
    if _gpu_wins(cpu, gpu, state, 1):
        return 1.0
    if not (warm_gpu < warm_cpu):
        return float("inf")
    high = 2
    while high <= horizon and not _gpu_wins(cpu, gpu, state, high):
        high *= 2
    if high > horizon:
        return float("inf")
    low = high // 2
    while low + 1 < high:
        middle = (low + high) // 2
        if _gpu_wins(cpu, gpu, state, middle):
            high = middle
        else:
            low = middle
    return float(high)


def state_switch_map(
    candidates: pd.DataFrame,
    *,
    resource_states: Sequence[str] = tuple(READY_DEVICE),
    horizon: int = 10 ** 9,
) -> pd.DataFrame:
    """`K` de cambio de dispositivo por `config_id` y estado, con su banda.

    La banda (`switch_k_low`/`switch_k_high`) es la misma envolvente
    rectangular de extremos marginales que usa `compact.amortization_map`: no
    es un intervalo de confianza. Se reporta porque la seccion 12.5 prohibe
    presentar `K_break_even` como entero exacto.
    """
    lookup = _ref_lookup(candidates)
    records: list[dict[str, Any]] = []
    for config_id in ref_configurations(candidates):
        sample = lookup[(config_id, "cpu", "cold")]
        variants: dict[str, dict[str, dict[str, float]]] = {}
        for statistic, gpu_stat, cpu_stat in (
            ("mean", "mean", "mean"),
            ("low", "min", "max"),   # escenario mas favorable a GPU
            ("high", "max", "min"),  # escenario mas desfavorable a GPU
        ):
            per_device: dict[str, dict[str, float]] = {}
            for device in ("cpu", "gpu"):
                statistic_name = gpu_stat if device == "gpu" else cpu_stat
                cold = lookup[(config_id, device, "cold")]
                warm = lookup[(config_id, device, "warm")]
                per_device[device] = {
                    "e_cold": float(cold[f"energy_{statistic_name}"]),
                    "t_cold": float(cold[f"time_{statistic_name}"]),
                    "e_warm": float(warm[f"energy_{statistic_name}"]),
                    "t_warm": float(warm[f"time_{statistic_name}"]),
                }
            variants[statistic] = per_device
        for state in resource_states:
            mean = variants["mean"]
            record = {
                "config_id": config_id,
                "operation": str(sample["operation"]),
                "size": int(sample["size"]),
                "resource_state": state,
                "switch_k": switch_k_for_state(mean["cpu"], mean["gpu"], state, horizon=horizon),
                "switch_k_low": switch_k_for_state(
                    variants["low"]["cpu"], variants["low"]["gpu"], state, horizon=horizon,
                ),
                "switch_k_high": switch_k_for_state(
                    variants["high"]["cpu"], variants["high"]["gpu"], state, horizon=horizon,
                ),
                "gpu_wins_warm": int(
                    mean["gpu"]["e_warm"] * mean["gpu"]["t_warm"]
                    < mean["cpu"]["e_warm"] * mean["cpu"]["t_warm"]
                ),
            }
            record["switches_within_horizon"] = int(np.isfinite(record["switch_k"]))
            records.append(record)
    return pd.DataFrame(records).sort_values(
        ["operation", "size", "resource_state"], kind="mergesort",
    ).reset_index(drop=True)


def horizon_migration_summary(switch_map: pd.DataFrame) -> pd.DataFrame:
    """Cuantas configuraciones cambian de dispositivo en algun `K`, por estado.

    Reproduce la tabla de evidencia de la seccion 12.2. La comprobacion de
    consistencia (los tres estados convergen al mismo conjunto asintotico) se
    expresa como ``asymptotic_gpu_wins``: debe ser identico en los tres
    estados e igual al numero de configuraciones con GPU ganadora en caliente.
    """
    records: list[dict[str, Any]] = []
    for state, group in switch_map.groupby("resource_state", observed=True):
        ready = READY_DEVICE[str(state)]
        finite = group["switches_within_horizon"].astype(bool)
        # "Migrar" significa terminar en un dispositivo distinto del que la
        # politica de K=1 usaria en ese estado.
        at_k1 = group["switch_k"] == 1.0
        records.append({
            "resource_state": str(state),
            "ready_device": ready or "none",
            "n_configs": int(len(group)),
            "gpu_wins_at_k1": int(at_k1.sum()),
            "gpu_wins_at_some_k": int(finite.sum()),
            "migrates_to_gpu_after_k1": int((finite & ~at_k1).sum()),
            "never_gpu": int((~finite).sum()),
            "asymptotic_gpu_wins": int(group["gpu_wins_warm"].sum()),
        })
    return pd.DataFrame(records)
