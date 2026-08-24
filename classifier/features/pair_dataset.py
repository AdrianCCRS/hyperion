"""Construcción del dataset por PAR (carga, estado de frecuencia) --
la arquitectura de `Estrategia_GPU_Fase2.md` §4 / `Estrategia_CPU_Fase2.md`
§3 (rediseño tras el Anexo M), común a ambos ejes.

La unidad ya no es la ventana: es la corrida completa a un nivel de
frecuencia candidato, descrita por características observadas en la
corrida del MISMO kernel y la MISMA repetición pero al nivel de
REFERENCIA. El objetivo es predecir, para ese par, cuánto cambian energía
y tiempo respecto de la referencia:

    energy_ratio = energia(nivel) / energia(f_ref)
    time_ratio   = tiempo(nivel)  / tiempo(f_ref)

GARANTÍA ANTI-FUGA ESTRUCTURAL. `build_pair_dataset()` solo copia
`feature_cols` de la fila de referencia -- nunca de la fila candidata. Aun
así hace falta un guardarraíl explícito para el caso en que alguien pase,
por descuido, una columna que ES el objetivo (o se deriva de él) como si
fuera una característica: `assert_no_target_leak()` existe para eso, y la
prueba V5/C6 de ambas estrategias consiste en confirmar que falla
ruidosamente cuando se le fuerza esa columna.
"""
from __future__ import annotations

import pandas as pd

# Nombres que NUNCA deben aparecer en `feature_cols`: son el objetivo o se
# derivan directamente de la corrida candidata, no de la de referencia.
FORBIDDEN_TARGET_COLUMNS = {
    "energy_j", "elapsed_s", "energy_ratio", "time_ratio",
}

# Reutiliza la misma prohibición de train_phase.py: cualquier cosa que
# entre en el cálculo de la intensidad operacional es la fuente de la
# etiqueta de régimen y no debe usarse como predictor, en esta
# arquitectura igual que en la de ventana.
FORBIDDEN_LABEL_SOURCE_COLUMNS = {
    "operational_intensity", "operational_intensity_uncore_real",
    "i_ridge_used", "flops_measured_window", "bytes_moved_window",
    "bytes_moved_uncore_real", "uncore_cas_count_read_interval",
    "uncore_cas_count_write_interval", "phase_label_uncore_real",
    "phase_label_hint",
}


def assert_no_target_leak(feature_cols: list[str]) -> None:
    """V5/C6: falla ruidosamente si `feature_cols` contiene el objetivo o
    su fuente. Llamar SIEMPRE antes de entrenar, no solo en pruebas."""
    leaked = (set(feature_cols) & FORBIDDEN_TARGET_COLUMNS) | (
        set(feature_cols) & FORBIDDEN_LABEL_SOURCE_COLUMNS
    )
    if leaked:
        raise AssertionError(
            f"fuga de objetivo/etiqueta en feature_cols: {sorted(leaked)}"
        )


# Características CPU: las mismas siete de train_phase.py (ninguna
# requiere uncore), promediadas sobre la corrida de referencia completa.
CPU_FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]

# Características GPU: telemetría NVML disponible en un solo nivel de
# referencia (Estrategia_GPU_Fase2.md §4) -- utilización, utilización de
# memoria y su cociente como proxy de memory-boundness.
GPU_FEATURES = ["gpu_util_pct", "gpu_mem_util_pct"]


def _elapsed_and_grouped_mean(
    df: pd.DataFrame,
    feature_cols: list[str],
    run_keys: list[str],
) -> pd.DataFrame:
    work = df.copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    t_start = pd.to_numeric(work["t_start_ns"], errors="coerce")
    t_end = pd.to_numeric(work["t_end_ns"], errors="coerce")
    grouped = work.groupby(run_keys, observed=True)
    elapsed_s = ((t_end.groupby([work[k] for k in run_keys]).transform("max")
                  - t_start.groupby([work[k] for k in run_keys]).transform("min"))
                 / 1e9)
    work["_elapsed_s"] = elapsed_s
    agg = grouped.agg(**{c: (c, "mean") for c in feature_cols})
    agg["elapsed_s"] = grouped["_elapsed_s"].first()
    return agg.reset_index()


def aggregate_cpu_runs(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    run_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Colapsa `windows.csv` de CPU a una fila por corrida: energía RAPL
    (pkg+dram) sumada sobre ventanas con `energy_valid == 1`, duración por
    rango de timestamps, y la media de `feature_cols`.

    Espejo de `cpu_policy_headroom.read_run()`, pero vectorizado sobre un
    DataFrame ya cargado en memoria en vez de leer directorio por
    directorio -- para operar sobre el dataset exportado en
    `local_datasets/`, que trae todas las corridas en un único CSV.
    """
    feature_cols = feature_cols or CPU_FEATURES
    run_keys = run_keys or ["kernel_ref", "repetition", "freq_level_id"]
    assert_no_target_leak(feature_cols)

    base = _elapsed_and_grouped_mean(df, feature_cols, run_keys)

    valid = df[df["energy_valid"].astype(str) == "1"].copy()
    valid["_energy_uj"] = (
        pd.to_numeric(valid["pkg_delta_uj"], errors="coerce")
        + pd.to_numeric(valid["dram_delta_uj"], errors="coerce")
    )
    energy = (
        valid.groupby(run_keys, observed=True)["_energy_uj"]
        .sum().div(1e6).rename("energy_j").reset_index()
    )
    out = base.merge(energy, on=run_keys, how="inner")
    return out[(out["elapsed_s"] > 0) & (out["energy_j"] > 0)].reset_index(drop=True)


def aggregate_gpu_runs(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    run_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Colapsa `windows.csv` de GPU a una fila por corrida: energía de GPU
    (NVML, `gpu_energy_delta_mj`) sumada sobre filas `quality_status ==
    "gpu_telemetry"` con `gpu_energy_valid == 1` -- la métrica primaria
    tras la corrección del Anexo M (energía de acelerador, no del nodo) --,
    duración por rango de timestamps sobre TODAS las filas de la corrida,
    y la media de `feature_cols` restringida a las filas de telemetría GPU
    (las CPU-passthrough no tienen `gpu_util_pct`).

    Espejo de `gpu_oracle_headroom.read_run()`.
    """
    feature_cols = feature_cols or GPU_FEATURES
    run_keys = run_keys or ["kernel_ref", "repetition", "gpu_freq_level_id"]
    assert_no_target_leak(feature_cols)

    if "run_accepted" in df.columns:
        df = df[df["run_accepted"].astype(str).str.lower() == "true"]

    gpu_rows = df[df["quality_status"] == "gpu_telemetry"].copy()
    for col in feature_cols:
        gpu_rows[col] = pd.to_numeric(gpu_rows[col], errors="coerce")
    gpu_features = (
        gpu_rows.groupby(run_keys, observed=True)
        .agg(**{c: (c, "mean") for c in feature_cols})
        .reset_index()
    )

    valid_energy = gpu_rows[gpu_rows["gpu_energy_valid"].astype(str) == "1"].copy()
    valid_energy["gpu_energy_delta_mj"] = pd.to_numeric(
        valid_energy["gpu_energy_delta_mj"], errors="coerce"
    )
    energy = (
        valid_energy.groupby(run_keys, observed=True)["gpu_energy_delta_mj"]
        .sum().div(1e3).rename("energy_j").reset_index()
    )

    t_start = pd.to_numeric(df["t_start_ns"], errors="coerce")
    t_end = pd.to_numeric(df["t_end_ns"], errors="coerce")
    keys = [df[k] for k in run_keys]
    elapsed = (
        (t_end.groupby(keys).transform("max") - t_start.groupby(keys).transform("min"))
        .div(1e9).rename("elapsed_s")
    )
    elapsed_by_run = (
        df.assign(_elapsed_s=elapsed)
        .groupby(run_keys, observed=True)["_elapsed_s"].first()
        .rename("elapsed_s").reset_index()
    )

    out = gpu_features.merge(energy, on=run_keys, how="inner").merge(
        elapsed_by_run, on=run_keys, how="inner"
    )
    out = out.rename(columns={"gpu_freq_level_id": "freq_level_id"} if "gpu_freq_level_id" in run_keys else {})
    return out[(out["elapsed_s"] > 0) & (out["energy_j"] > 0)].reset_index(drop=True)


def build_pair_dataset(
    runs: pd.DataFrame,
    feature_cols: list[str],
    ref_level: str,
    kernel_col: str = "kernel_ref",
    rep_col: str = "repetition",
    level_col: str = "freq_level_id",
) -> pd.DataFrame:
    """A partir de una tabla de una fila por corrida (`aggregate_*_runs`),
    produce una fila por (kernel, repetición, nivel candidato) con las
    características tomadas de la corrida de referencia DE ESA MISMA
    repetición, y los objetivos `energy_ratio`/`time_ratio`.

    Emparejar por repetición y no por el promedio del kernel conserva el
    ruido correlacionado entre referencia y candidato (misma variación de
    fondo del nodo en ese instante), en vez de mezclarlo con el de otra
    repetición.

    Corridas candidatas sin referencia emparejable (repetición ausente al
    nivel de referencia) se DESCARTAN, no se rellenan -- mismo principio
    de no ocultar pérdida de datos que rige el resto del instrumento. El
    conteo descartado se devuelve como atributo `.attrs["dropped_no_ref"]`
    del DataFrame resultante, para que quien lo llame decida si es
    aceptable en vez de perderlo en silencio.
    """
    assert_no_target_leak(feature_cols)

    ref = runs[runs[level_col] == ref_level].set_index([kernel_col, rep_col])
    cand = runs[runs[level_col] != ref_level].reset_index(drop=True)

    ref_key = list(zip(cand[kernel_col], cand[rep_col]))
    has_ref = pd.Series(ref_key, index=cand.index).isin(ref.index)
    dropped = int((~has_ref).sum())

    cand = cand[has_ref].reset_index(drop=True)
    matched_ref = ref.loc[list(zip(cand[kernel_col], cand[rep_col]))].reset_index(drop=True)

    out = cand.copy()
    for col in feature_cols:
        out[f"ref_{col}"] = matched_ref[col].to_numpy()
    out["ref_energy_j"] = matched_ref["energy_j"].to_numpy()
    out["ref_elapsed_s"] = matched_ref["elapsed_s"].to_numpy()
    out["energy_ratio"] = out["energy_j"].to_numpy() / out["ref_energy_j"].to_numpy()
    out["time_ratio"] = out["elapsed_s"].to_numpy() / out["ref_elapsed_s"].to_numpy()

    out.attrs["dropped_no_ref"] = dropped
    return out
