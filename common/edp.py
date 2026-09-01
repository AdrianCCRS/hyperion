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
    """
    frames = [pd.read_csv(p, low_memory=False) for p in windows_csv_paths]
    df = pd.concat(frames, ignore_index=True)

    is_gpu_row = df["quality_status"] == "gpu_telemetry"
    cpu_ok = (df["quality_status"] == "ok") & df["phase_label_train"].notna() & (df["phase_label_train"] != "")
    gpu_ok = is_gpu_row & df["phase_label_train"].notna() & (df["phase_label_train"] != "")
    df = df[cpu_ok | gpu_ok].copy()

    df["device"] = np.where(is_gpu_row.loc[df.index], "gpu", "cpu")
    return df


def compute_window_edp(df: pd.DataFrame) -> pd.Series:
    """EDP (Joule-segundo) de cada ventana: energía_J * tiempo_s.

    CPU: energía = pkg_delta_uj + dram_delta_uj (RAPL paquete + memoria,
    µJ -> J). GPU: energía = gpu_energy_delta_mj (mJ -> J). Tiempo, en
    ambos casos: delta_t_ns -> s. Filas sin energía válida (energy_valid
    == False en CPU) quedan NaN -- se descartan aguas abajo, nunca se
    fabrica un EDP con energía no verificada. Requiere que `df` tenga la
    columna `device` (ver `load_windows`).
    """
    time_s = df["delta_t_ns"].astype(float) / 1e9
    is_gpu = df["device"] == "gpu"

    cpu_energy_j = (
        df["pkg_delta_uj"].astype(float).fillna(0.0)
        + df["dram_delta_uj"].astype(float).fillna(0.0)
    ) / 1e6
    cpu_energy_j = cpu_energy_j.where(df.get("energy_valid", True).astype(bool), np.nan)

    gpu_energy_j = df.get("gpu_energy_delta_mj", pd.Series(np.nan, index=df.index)).astype(float) / 1e3

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
