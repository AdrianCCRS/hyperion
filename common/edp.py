"""Cálculo de EDP (Energy-Delay Product) desde windows.csv, compartido
entre Fase 3 (derivador de política, §3.5) y Fase 4 (reporte de evaluación,
§5.2) -- movido aquí desde fase3_daemon/policy/derive_policy_table.py
durante la construcción de Fase 4, siguiendo el mismo criterio que ya se
aplicó a common/stats.py: código que necesitan 2+ fases no se duplica.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_windows(windows_csv_paths: list[Path]) -> pd.DataFrame:
    """Carga y concatena varios windows.csv, filtrando SOLO filas usables:
    quality_status="ok" en CPU, "gpu_telemetry" con etiqueta válida en GPU
    -- nunca corridas rechazadas. Añade una columna `device` derivada
    (`"gpu"` si quality_status == "gpu_telemetry", `"cpu"` en otro caso).

    En CPU, además exige `frequency_quality_status` en {"valid",
    "not_applicable_native"} -- mismo filtro que ya aplica
    `fase2_clasificador/training/train_phase.py::load()` y que
    `validation.py` establece como el par "el reloj efectivo de esta
    ventana quedó verificado" (§ARC-174, clasificación por ventana). Sin
    este filtro, una ventana cuyo `freq_level_id` dice "F4" pero cuyo
    reloj real todavía no había convergido (`observation_unreliable`/
    `observation_unverified_grace`) contaminaría el EDP agregado de ese
    nivel sin que nada lo señale -- exactamente el tipo de dato no
    verificado que el proyecto se niega a usar en cualquier otro punto.
    No se aplica a GPU: esa columna queda vacía en filas GPU (ver
    `postprocess.py`, "Vacías para filas GPU" en `REQUIRED_OUTPUT_COLUMNS`)
    -- la calidad de la frecuencia GPU se controla aparte, con el reloj
    realmente observado (`gpu_sm_clock_mhz`, ver `median_observed_frequency`)
    y el filtro de asentamiento de `fase3_daemon/policy/derive_policy_table.py`.
    """
    frames = [pd.read_csv(p, low_memory=False) for p in windows_csv_paths]
    df = pd.concat(frames, ignore_index=True)

    is_gpu_row = df["quality_status"] == "gpu_telemetry"
    cpu_ok = (
        (df["quality_status"] == "ok")
        & df["phase_label_train"].notna() & (df["phase_label_train"] != "")
        & df["frequency_quality_status"].isin(["valid", "not_applicable_native"])
    )
    gpu_ok = is_gpu_row & df["phase_label_train"].notna() & (df["phase_label_train"] != "")
    df = df[cpu_ok | gpu_ok].copy()

    df["device"] = np.where(is_gpu_row.loc[df.index], "gpu", "cpu")
    return df


def _valid_mask(df: pd.DataFrame, column: str) -> pd.Series:
    """Máscara booleana de `column`, tratando su AUSENCIA como "todo
    válido" -- pero, a diferencia de `df.get(column, True)`, sin el bug de
    devolver un `bool` de Python plano (que no tiene `.astype`) cuando la
    columna falta: siempre devuelve una Series alineada con `df.index`.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    return df[column].astype(bool)


def compute_window_edp(df: pd.DataFrame) -> pd.Series:
    """EDP (Joule-segundo) de cada ventana: energía_J * tiempo_s.

    CPU: energía = pkg_delta_uj + dram_delta_uj (RAPL paquete + memoria,
    µJ -> J). GPU: energía = gpu_energy_delta_mj (mJ -> J). Tiempo, en
    ambos casos: delta_t_ns -> s. Filas sin energía válida (`energy_valid
    == False` en CPU, `gpu_energy_valid == False` en GPU -- ARC-95, el
    mismo bit de validez que CPU pero para el delta de energía GPU) quedan
    NaN -- se descartan aguas abajo, nunca se fabrica un EDP con energía
    no verificada. Requiere que `df` tenga la columna `device` (ver
    `load_windows`).
    """
    time_s = df["delta_t_ns"].astype(float) / 1e9
    is_gpu = df["device"] == "gpu"

    cpu_energy_j = (
        df["pkg_delta_uj"].astype(float).fillna(0.0)
        + df["dram_delta_uj"].astype(float).fillna(0.0)
    ) / 1e6
    cpu_energy_j = cpu_energy_j.where(_valid_mask(df, "energy_valid"), np.nan)

    gpu_energy_j = df.get("gpu_energy_delta_mj", pd.Series(np.nan, index=df.index)).astype(float) / 1e3
    gpu_energy_j = gpu_energy_j.where(_valid_mask(df, "gpu_energy_valid"), np.nan)

    energy_j = np.where(is_gpu, gpu_energy_j, cpu_energy_j)
    return pd.Series(energy_j, index=df.index) * time_s


def median_observed_frequency(df: pd.DataFrame, device: str, label: str, level: str,
                               level_col: str | None = None) -> float | None:
    """Mediana de la frecuencia/reloj REALMENTE observado (nunca el
    solicitado) para (device, phase_label_train, freq_level_id) -- mismo
    principio de verificación por relectura que el resto del proyecto
    aplica a la actuación de frecuencia (nunca confiar en lo solicitado).
    """
    level_col = level_col or ("gpu_freq_level_id" if device == "gpu" else "freq_level_id")
    observed_col = "gpu_sm_clock_mhz" if device == "gpu" else "freq_khz_observed"
    subset = df[(df["device"] == device) & (df["phase_label_train"] == label) & (df[level_col] == level)]
    values = pd.to_numeric(subset.get(observed_col), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())
