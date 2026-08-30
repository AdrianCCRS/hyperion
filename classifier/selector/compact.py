"""Dataset compacto del selector: una fila por `config_id` x `resource_state`.

Reformulacion aprobada en `docs/general/plan_reformulacion_selector_tamanos_20260830.md`
(Fase R1). Sustituye la formulacion "elegir una de 40 acciones" por tres
preguntas separadas:

1. amortizacion (`amortization_map`): cuantos despachos `K` hacen falta para
   que pagar el arranque de GPU salga a cuenta;
2. dispositivo (`build_compact_dataset`): regresion sobre
   ``y = log(EDP_GPU_REF / EDP_CPU_REF)``;
3. frecuencia (`dvfs_headroom`): cuanto EDP adicional queda por ganar
   actuando la frecuencia DESPUES de haber elegido bien el dispositivo.

Diferencia operativa central con `dataset.build_strategy_a`/`build_strategy_c`:
aquellos exigen las 40 acciones completas por `config_id`
(`_complete_candidate_slice`) y descartan en silencio cualquier configuracion
que solo tenga las acciones REF. El dataset compacto exige unicamente
``REF_ACTIONS`` -- las dos acciones que definen el target -- de modo que las
configuraciones medidas solo a REF (campanas suplementarias de tamanos
grandes) son ciudadanos de primera clase sin romper el constructor de 40
acciones, que sigue siendo valido para el barrido de frecuencias del
catalogo base.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np
import pandas as pd

from .dataset import DatasetContractError, _static_descriptors

CPU_REF_ACTION = "cpu:REF"
GPU_REF_ACTION = "gpu:REF:REF"
REF_ACTIONS = (CPU_REF_ACTION, GPU_REF_ACTION)

#: Region que ve cada dispositivo en cada estado de recurso. Un dispositivo
#: ya preparado (`*_ready`) responde con su costo `warm`; el otro tiene que
#: pagar su arranque, es decir su costo `cold`. `none_ready` paga los dos.
RESOURCE_STATES: Mapping[str, tuple[str, str]] = {
    # estado: (region que ve la CPU, region que ve la GPU)
    "none_ready": ("cold", "cold"),
    "cpu_ready": ("warm", "cold"),
    "gpu_ready": ("cold", "warm"),
}

#: Descriptores analiticos que un selector puede conocer ANTES de ejecutar.
STATIC_FEATURES = (
    "operation",
    "size",
    "log10_n",
    "flops_per_dispatch_analytic",
    "log10_flops_per_dispatch",
    "logical_bytes_per_dispatch",
    "log10_logical_bytes",
    "arithmetic_intensity_analytic",
    "resource_state",
)

#: Columnas prohibidas como entrada (seccion 7.3 del plan). El chequeo se
#: hace por subcadena porque las familias completas (`edp_*`, `probe_edp_*`,
#: cualquier `*margin*`) son igual de fugadoras que las columnas exactas.
LEAKAGE_PATTERNS = (
    "edp_",
    "_edp",
    "margin",
    "is_optimal",
    "optimum_stability",
    "winner",
    "best_action",
    "action_id",
    "run_id",
    "repetition",
    "device_label",
    "y_log_edp_ratio",
    "cpu_ref_",
    "gpu_ref_",
)

#: El piso de ruido medido entre repeticiones de la misma accion (CV mediano
#: ~3.11 %, calibracion de 2026-08-30). Un margen porcentual por debajo de
#: esto no se distingue de jitter de medicion.
NOISE_FLOOR_PCT = 3.11
DEFAULT_Z_SCORE = 1.96


class CompactDatasetError(RuntimeError):
    """El resumen de candidatos no permite construir el dataset compacto."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise CompactDatasetError(f"{what}: faltan columnas {missing}")


def _boolean_mask(series: pd.Series) -> pd.Series:
    """Interpreta booleanos serializados sin convertir ``"False"`` en True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _ref_slice(candidates: pd.DataFrame, *, require_eligible: bool = True) -> pd.DataFrame:
    """Filas REF de CPU y GPU, una por `config_id` x `device` x `region`."""
    _require_columns(
        candidates,
        ("config_id", "operation", "size", "device", "action_id", "region",
         "edp_mean", "energy_mean", "time_mean"),
        "candidate_summary",
    )
    work = candidates[candidates["action_id"].isin(REF_ACTIONS)].copy()
    if require_eligible and "eligible_repetitions" in work:
        work = work[_boolean_mask(work["eligible_repetitions"])]
    duplicated = work.duplicated(["config_id", "action_id", "region"])
    if duplicated.any():
        offenders = work.loc[duplicated, "config_id"].unique()[:3]
        raise CompactDatasetError(f"filas REF duplicadas para {list(offenders)}")
    return work


def ref_configurations(candidates: pd.DataFrame) -> list[str]:
    """`config_id` que tienen las dos acciones REF en `cold` y en `warm`.

    Este es el unico requisito de completitud del dataset compacto: no exige
    las 40 acciones, por lo que admite las configuraciones medidas solo a REF.
    """
    work = _ref_slice(candidates)
    needed = {(action, region) for action in REF_ACTIONS for region in ("cold", "warm")}
    complete: list[str] = []
    for config_id, group in work.groupby("config_id", observed=True):
        present = set(zip(group["action_id"], group["region"]))
        if needed <= present:
            complete.append(str(config_id))
    return sorted(complete)


def _ref_lookup(candidates: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(config_id, device, region) -> fila REF, como diccionario plano."""
    work = _ref_slice(candidates)
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in work.to_dict("records"):
        key = (str(row["config_id"]), str(row["device"]), str(row["region"]))
        lookup[key] = row
    return lookup


def build_compact_dataset(
    candidates: pd.DataFrame,
    *,
    resource_states: Sequence[str] = tuple(RESOURCE_STATES),
    z_score: float = DEFAULT_Z_SCORE,
) -> pd.DataFrame:
    """Una fila por `config_id` x `resource_state` con el target de dispositivo.

    El target es ``y = log(EDP_GPU_REF / EDP_CPU_REF)`` evaluado en la region
    que corresponde a cada dispositivo segun el estado (ver `RESOURCE_STATES`):
    ``y < 0`` favorece GPU, ``y > 0`` favorece CPU y ``y ~= 0`` es zona de
    abstencion. Se conserva la magnitud porque el regret de equivocarse es
    proporcional a ``|y|``, no constante como en una clasificacion binaria.
    """
    unknown = sorted(set(resource_states) - set(RESOURCE_STATES))
    if unknown:
        raise CompactDatasetError(f"estados de recurso desconocidos: {unknown}")
    if z_score <= 0:
        raise CompactDatasetError(f"z_score debe ser positivo: {z_score}")
    lookup = _ref_lookup(candidates)
    records: list[dict[str, Any]] = []
    for config_id in ref_configurations(candidates):
        sample = lookup[(config_id, "cpu", "cold")]
        operation, size = str(sample["operation"]), int(sample["size"])
        for state in resource_states:
            cpu_region, gpu_region = RESOURCE_STATES[state]
            cpu = lookup[(config_id, "cpu", cpu_region)]
            gpu = lookup[(config_id, "gpu", gpu_region)]
            cpu_edp, gpu_edp = float(cpu["edp_mean"]), float(gpu["edp_mean"])
            if not (cpu_edp > 0 and gpu_edp > 0):
                raise CompactDatasetError(
                    f"EDP REF no positivo en {config_id}/{state}: cpu={cpu_edp}, gpu={gpu_edp}"
                )
            y = math.log(gpu_edp / cpu_edp)
            record: dict[str, Any] = {
                "config_id": config_id,
                "operation": operation,
                "size": size,
                "family": str(sample.get("family", "")),
                "resource_state": state,
                "cpu_region": cpu_region,
                "gpu_region": gpu_region,
                "y_log_edp_ratio": y,
                "device_label": "gpu" if y < 0 else "cpu",
                # Auditoria y metricas de regret -- NO son features (ver
                # LEAKAGE_PATTERNS: los prefijos cpu_ref_/gpu_ref_ estan
                # explicitamente prohibidos como entrada del modelo).
                "cpu_ref_edp_js": cpu_edp,
                "gpu_ref_edp_js": gpu_edp,
                "cpu_ref_energy_j": float(cpu["energy_mean"]),
                "gpu_ref_energy_j": float(gpu["energy_mean"]),
                "cpu_ref_time_s": float(cpu["time_mean"]),
                "gpu_ref_time_s": float(gpu["time_mean"]),
                "oracle_ref_edp_js": min(cpu_edp, gpu_edp),
            }
            record.update(_static_descriptors(operation, size))
            record["device_margin_pct"] = 100.0 * abs(gpu_edp - cpu_edp) / min(cpu_edp, gpu_edp)
            # Si el resumen conserva dispersion y numero de repeticiones,
            # propagamos el error estandar sobre log(GPU/CPU). Es una
            # aproximacion normal (no un IC exacto) y no supone independencia
            # entre las 204 filas: cada fila sigue siendo config_id x estado.
            cpu_std = float(cpu.get("edp_std", np.nan))
            gpu_std = float(gpu.get("edp_std", np.nan))
            cpu_n = int(cpu.get("n_repetitions", 0) or 0)
            gpu_n = int(gpu.get("n_repetitions", 0) or 0)
            if (
                np.isfinite(cpu_std) and cpu_std >= 0 and cpu_n > 0
                and np.isfinite(gpu_std) and gpu_std >= 0 and gpu_n > 0
            ):
                log_se = math.sqrt(
                    (cpu_std / math.sqrt(cpu_n) / cpu_edp) ** 2
                    + (gpu_std / math.sqrt(gpu_n) / gpu_edp) ** 2
                )
                ci_low, ci_high = y - z_score * log_se, y + z_score * log_se
                separated = ci_high < 0 or ci_low > 0
                method = "normal_approx_log_ratio"
            else:
                log_se = ci_low = ci_high = float("nan")
                separated = record["device_margin_pct"] >= NOISE_FLOOR_PCT
                method = "noise_floor_fallback"
            record.update({
                "y_log_edp_ratio_se_normal_approx": log_se,
                "y_log_edp_ratio_ci_low_normal_approx": ci_low,
                "y_log_edp_ratio_ci_high_normal_approx": ci_high,
                "device_decision_separated": int(separated),
                "device_decision_uncertainty_method": method,
            })
            records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise CompactDatasetError("ninguna configuracion tiene las dos acciones REF completas")
    return frame.sort_values(["config_id", "resource_state"], kind="mergesort").reset_index(drop=True)


def _probe_action_for(device: str) -> str:
    return CPU_REF_ACTION if device == "cpu" else GPU_REF_ACTION


def attach_probe_features(
    compact: pd.DataFrame,
    run_regions: pd.DataFrame,
) -> pd.DataFrame:
    """Variante con sondeo (seccion 7.2): una unica ejecucion real observada.

    El sondeo lo produce el dispositivo ya preparado, asi que solo existe para
    `cpu_ready` y `gpu_ready`; `none_ready` no ha ejecutado nada todavia y sus
    columnas de sondeo quedan ausentes, marcadas con sus indicadores
    ``*_missing``. Se toma la PRIMERA repeticion, no el promedio de las tres:
    en despliegue no existen tres repeticiones retrospectivas.
    """
    from .dataset import CPU_TELEMETRY, GPU_TELEMETRY

    _require_columns(
        run_regions,
        ("config_id", "action_id", "region", "repetition", "run_id",
         "time_per_dispatch_s", "energy_per_dispatch_j", "region_to_sampling_ratio"),
        "run_regions",
    )
    out = compact.copy()
    probe_columns: list[str] = []
    for device in ("cpu", "gpu"):
        telemetry = CPU_TELEMETRY if device == "cpu" else GPU_TELEMETRY
        work = run_regions[
            (run_regions["region"] == "cold")
            & (run_regions["action_id"] == _probe_action_for(device))
        ].copy()
        if work.empty:
            continue
        first = (
            work.sort_values(["config_id", "repetition", "run_id"], kind="mergesort")
            .groupby("config_id", observed=True, as_index=False)
            .first()
        )
        first["avg_power_w"] = first["energy_per_dispatch_j"] / first["time_per_dispatch_s"]
        # Si la region fria dura menos que el intervalo de muestreo, tiempo y
        # energia siguen siendo integraciones validas, pero no se fabrica una
        # observacion puntual de telemetria desde una fraccion de ventana.
        low_resolution = first["region_to_sampling_ratio"] < 1.0
        for column in telemetry:
            if column in first:
                first.loc[low_resolution, column] = np.nan
        numeric = [
            "time_per_dispatch_s", "energy_per_dispatch_j", "avg_power_w",
            "region_to_sampling_ratio", *telemetry,
        ]
        numeric = [column for column in numeric if column in first]
        renamed = first[["config_id", *numeric]].rename(
            columns={column: f"probe_{column}" for column in numeric}
        )
        state = f"{device}_ready"
        mask = out["resource_state"] == state
        merged = out.loc[mask, ["config_id"]].merge(renamed, on="config_id", how="left")
        merged.index = out.index[mask]
        for column in renamed.columns:
            if column == "config_id":
                continue
            if column not in out:
                out[column] = np.nan
                probe_columns.append(column)
            out.loc[mask, column] = merged[column]
    out["probe_device"] = out["resource_state"].map({
        "cpu_ready": "cpu", "gpu_ready": "gpu",
    })
    out["probe_device_missing"] = out["probe_device"].isna().astype(int)
    for column in probe_columns:
        out[f"{column}_missing"] = out[column].isna().astype(int)
    return out


def leaking_columns(feature_columns: Iterable[str]) -> list[str]:
    """Columnas de `feature_columns` prohibidas por la seccion 7.3 del plan."""
    return sorted(
        column for column in feature_columns
        if any(pattern in str(column) for pattern in LEAKAGE_PATTERNS)
    )


def assert_no_leakage(feature_columns: Iterable[str]) -> None:
    offenders = leaking_columns(feature_columns)
    if offenders:
        raise CompactDatasetError(f"caracteristicas con fuga de la etiqueta: {offenders}")


def feature_columns(frame: pd.DataFrame, *, with_probe: bool = False) -> list[str]:
    """Lista auditada de caracteristicas utilizables, sin fuga por construccion."""
    columns = [column for column in STATIC_FEATURES if column in frame]
    if with_probe:
        columns += sorted(
            column for column in frame.columns
            if str(column).startswith("probe_") and column != "probe_device"
        )
        if "probe_device" in frame:
            columns.append("probe_device")
    assert_no_leakage(columns)
    return columns


# --------------------------------------------------------------------------
# Mapa de amortizacion K_break_even (seccion 6.2)
# --------------------------------------------------------------------------


def edp_total(e_cold: float, e_warm: float, t_cold: float, t_warm: float, k: int) -> float:
    """``EDP_total(d, K) = (E_cold + (K-1) E_warm) * (T_cold + (K-1) T_warm)``."""
    if k < 1:
        raise CompactDatasetError(f"horizonte K debe ser >= 1: {k}")
    return (e_cold + (k - 1) * e_warm) * (t_cold + (k - 1) * t_warm)


def break_even_k(
    cpu: Mapping[str, float],
    gpu: Mapping[str, float],
    *,
    horizon: int = 10 ** 9,
) -> float:
    """Menor `K` entero con ``EDP_total(GPU, K) < EDP_total(CPU, K)``.

    Devuelve ``inf`` si no existe dentro del horizonte. Nota analitica que
    ancla la implementacion: cuando ``K -> inf`` el termino frio se vuelve
    despreciable y la razon tiende a ``EDP_warm(GPU) / EDP_warm(CPU)``. Por
    por tanto, para una GPU que pierde inicialmente, **`K_break_even` solo
    puede ser finito si la GPU gana en caliente**. El caso transitorio en que
    GPU gana en K=1 pero pierde asintoticamente se conserva y se marca aparte.
    La busqueda se corta con ese criterio antes de barrer, en lugar de
    depender de que el horizonte sea "suficientemente grande".
    """
    warm_gpu = gpu["e_warm"] * gpu["t_warm"]
    warm_cpu = cpu["e_warm"] * cpu["t_warm"]
    if not (warm_gpu < warm_cpu):
        # La GPU no gana ni con arranque amortizado a cero: no hay cruce.
        # (Se comprueba K=1 igualmente: podria ganar en frio y perder en
        # caliente, en cuyo caso el cruce existe pero no es un break-even.)
        if edp_total(gpu["e_cold"], gpu["e_warm"], gpu["t_cold"], gpu["t_warm"], 1) < \
           edp_total(cpu["e_cold"], cpu["e_warm"], cpu["t_cold"], cpu["t_warm"], 1):
            return 1.0
        return float("inf")

    def gpu_wins(k: int) -> bool:
        return edp_total(gpu["e_cold"], gpu["e_warm"], gpu["t_cold"], gpu["t_warm"], k) < \
               edp_total(cpu["e_cold"], cpu["e_warm"], cpu["t_cold"], cpu["t_warm"], k)

    if gpu_wins(1):
        return 1.0
    # Busqueda exponencial y luego binaria: el cruce es unico una vez que la
    # GPU domina en caliente (las dos curvas son cuadraticas en K y la de GPU
    # tiene coeficiente principal menor).
    high = 2
    while high <= horizon and not gpu_wins(high):
        high *= 2
    if high > horizon:
        return float("inf")
    low = high // 2
    while low + 1 < high:
        middle = (low + high) // 2
        if gpu_wins(middle):
            high = middle
        else:
            low = middle
    return float(high)


def _cost_triplet(row: Mapping[str, Any], statistic: str) -> tuple[float, float]:
    """(energia, tiempo) por despacho segun `statistic` in {mean, min, max}."""
    return float(row[f"energy_{statistic}"]), float(row[f"time_{statistic}"])


def low_resolution_configs(run_regions: pd.DataFrame) -> set[str]:
    """`config_id` cuya region fria REF dura menos que el intervalo de muestreo.

    `candidate_summary` agrega y pierde `region_to_sampling_ratio`, asi que la
    marca se deriva de `run_regions`. Sin `run_regions` no se puede afirmar
    nada sobre resolucion, y `amortization_map` lo declara como desconocido en
    lugar de reportar un falso "resolucion nominal".
    """
    if run_regions is None or run_regions.empty:
        return set()
    _require_columns(run_regions, ("config_id", "action_id", "region",
                                   "region_to_sampling_ratio"), "run_regions")
    cold_ref = run_regions[
        (run_regions["region"] == "cold") & (run_regions["action_id"].isin(REF_ACTIONS))
    ]
    ratio = pd.to_numeric(cold_ref["region_to_sampling_ratio"], errors="coerce")
    return set(cold_ref.loc[ratio < 1.0, "config_id"].astype(str).unique())


def amortization_map(
    candidates: pd.DataFrame,
    run_regions: pd.DataFrame | None = None,
    *,
    horizon: int = 10 ** 9,
) -> pd.DataFrame:
    """Mapa `K_break_even` CPU->GPU por operacion y tamano, con banda.

    La banda es una sensibilidad rectangular con los extremos marginales de
    energia y tiempo observados en las 3 repeticiones. No presupone que el
    minimo de energia y el minimo de tiempo procedan de la misma repeticion,
    por lo que se reporta como envolvente conservadora, no como intervalo de
    confianza:

    - ``k_break_even_low``: escenario mas favorable a GPU (GPU en su
      extremos marginales mas baratos, CPU en los mas caros);
    - ``k_break_even_high``: escenario mas desfavorable a GPU.

    Se reporta ademas ``cold_low_resolution``: cuando la region fria dura
    menos que el intervalo de muestreo, ``E_cold``/``T_cold`` son
    estimaciones de baja resolucion y `K_break_even` hereda esa
    incertidumbre. Ese caso no se corrige aqui; se marca.
    """
    lookup = _ref_lookup(candidates)
    low_resolution = low_resolution_configs(run_regions) if run_regions is not None else None
    records: list[dict[str, Any]] = []
    for config_id in ref_configurations(candidates):
        sample = lookup[(config_id, "cpu", "cold")]
        costs: dict[tuple[str, str], dict[str, float]] = {}
        for device in ("cpu", "gpu"):
            cold, warm = lookup[(config_id, device, "cold")], lookup[(config_id, device, "warm")]
            for statistic, cold_stat, warm_stat in (
                ("mean", "mean", "mean"),
                # Favorable a GPU / desfavorable a CPU y viceversa: se aplica
                # el extremo optimista al dispositivo GPU y el pesimista al CPU.
                ("low", "min" if device == "gpu" else "max", "min" if device == "gpu" else "max"),
                ("high", "max" if device == "gpu" else "min", "max" if device == "gpu" else "min"),
            ):
                e_cold, t_cold = _cost_triplet(cold, cold_stat)
                e_warm, t_warm = _cost_triplet(warm, warm_stat)
                costs[(device, statistic)] = {
                    "e_cold": e_cold, "t_cold": t_cold,
                    "e_warm": e_warm, "t_warm": t_warm,
                }
        record: dict[str, Any] = {
            "config_id": config_id,
            "operation": str(sample["operation"]),
            "size": int(sample["size"]),
        }
        for statistic, suffix in (("mean", ""), ("low", "_low"), ("high", "_high")):
            record[f"k_break_even{suffix}"] = break_even_k(
                costs[("cpu", statistic)], costs[("gpu", statistic)], horizon=horizon,
            )
        cpu_mean, gpu_mean = costs[("cpu", "mean")], costs[("gpu", "mean")]
        record.update({
            "cpu_e_cold_j": cpu_mean["e_cold"], "cpu_t_cold_s": cpu_mean["t_cold"],
            "cpu_e_warm_j": cpu_mean["e_warm"], "cpu_t_warm_s": cpu_mean["t_warm"],
            "gpu_e_cold_j": gpu_mean["e_cold"], "gpu_t_cold_s": gpu_mean["t_cold"],
            "gpu_e_warm_j": gpu_mean["e_warm"], "gpu_t_warm_s": gpu_mean["t_warm"],
            "warm_edp_ratio_gpu_over_cpu": (
                (gpu_mean["e_warm"] * gpu_mean["t_warm"])
                / (cpu_mean["e_warm"] * cpu_mean["t_warm"])
            ),
            "gpu_wins_warm": int(
                gpu_mean["e_warm"] * gpu_mean["t_warm"] < cpu_mean["e_warm"] * cpu_mean["t_warm"]
            ),
            # None = desconocido (no se paso run_regions), no "resolucion ok".
            "cold_low_resolution": (
                None if low_resolution is None else int(config_id in low_resolution)
            ),
        })
        records.append(record)
    frame = pd.DataFrame(records)
    # Coherencia con la nota analitica: finito <=> GPU gana en caliente. La
    # unica excepcion legitima es una GPU que ya gana en el primer despacho
    # (K=1) aunque pierda asintoticamente; se excluye del chequeo en vez de
    # contarse como inconsistencia, y se marca aparte porque en ese caso el
    # "K minimo" no es un punto de amortizacion sino su opuesto.
    finite = np.isfinite(frame["k_break_even"])
    frame["gpu_wins_first_dispatch"] = (frame["k_break_even"] == 1.0).astype(int)
    expected_finite = frame["gpu_wins_warm"].astype(bool) | (frame["k_break_even"] == 1.0)
    inconsistent = frame[finite != expected_finite]
    frame.attrs["analytic_inconsistencies"] = inconsistent["config_id"].tolist()
    return frame.sort_values(["operation", "size"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Headroom de DVFS despues de elegir dispositivo (seccion 11.5)
# --------------------------------------------------------------------------


def dvfs_headroom(
    candidates: pd.DataFrame,
    *,
    resource_states: Sequence[str] = tuple(RESOURCE_STATES),
) -> pd.DataFrame:
    """Ganancia maxima de EDP por actuar la frecuencia tras elegir dispositivo.

    Por `config_id` y estado de recurso: se fija primero el dispositivo
    ganador a REF (la misma decision que produce ``y`` en
    `build_compact_dataset`) y luego se busca la mejor accion *de ese mismo
    dispositivo*, evaluada en la region que ese dispositivo ve en ese estado.
    La diferencia es el headroom que la capa DVFS puede capturar.

    El calculo es por estado, no por region, a proposito: comparar CPU-warm
    contra GPU-warm no corresponde a ningun estado alcanzable del agente (los
    dos dispositivos no pueden estar calientes a la vez en el contrato
    cold/warm vigente), y hacerlo cambia el ganador en configuraciones reales.

    ``dvfs_headroom_pct`` es una **cota superior**: excluye el costo de
    actuacion de la frecuencia, que se mide por separado. Si la cota no supera
    `NOISE_FLOOR_PCT`, no hay margen que la capa DVFS pueda capturar.
    """
    unknown = sorted(set(resource_states) - set(RESOURCE_STATES))
    if unknown:
        raise CompactDatasetError(f"estados de recurso desconocidos: {unknown}")
    _require_columns(candidates, ("config_id", "operation", "size", "device",
                                  "action_id", "region", "edp_mean"), "candidate_summary")
    work = candidates.copy()
    if "eligible_repetitions" in work:
        work = work[_boolean_mask(work["eligible_repetitions"])]
    work = work[np.isfinite(pd.to_numeric(work["edp_mean"], errors="coerce"))]
    lookup = _ref_lookup(candidates)
    records: list[dict[str, Any]] = []
    for config_id, group in work.groupby("config_id", observed=True):
        config_id = str(config_id)
        if (config_id, "cpu", "cold") not in lookup:
            continue
        sample = lookup[(config_id, "cpu", "cold")]
        for state in resource_states:
            regions = dict(zip(("cpu", "gpu"), RESOURCE_STATES[state]))
            try:
                ref_edp_by_device = {
                    device: float(lookup[(config_id, device, regions[device])]["edp_mean"])
                    for device in ("cpu", "gpu")
                }
            except KeyError:
                continue
            winner = min(ref_edp_by_device, key=ref_edp_by_device.get)
            ref_edp = ref_edp_by_device[winner]
            same_device = group[
                (group["device"] == winner) & (group["region"] == regions[winner])
            ]
            if same_device.empty:
                continue
            best_same_device = same_device.loc[same_device["edp_mean"].idxmin()]
            records.append({
                "config_id": config_id,
                "operation": str(sample["operation"]),
                "size": int(sample["size"]),
                "resource_state": state,
                "winner_device_at_ref": winner,
                "winner_region": regions[winner],
                "edp_ref_winner_js": ref_edp,
                "edp_best_same_device_js": float(best_same_device["edp_mean"]),
                "best_action_same_device": str(best_same_device["action_id"]),
                "n_actions_same_device": int(len(same_device)),
                "dvfs_headroom_pct": (
                    100.0 * (ref_edp - float(best_same_device["edp_mean"])) / ref_edp
                ),
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise CompactDatasetError("sin configuraciones con acciones REF completas")
    frame["above_noise_floor"] = (frame["dvfs_headroom_pct"] >= NOISE_FLOOR_PCT).astype(int)
    return frame.sort_values(
        ["operation", "size", "resource_state"], kind="mergesort",
    ).reset_index(drop=True)
