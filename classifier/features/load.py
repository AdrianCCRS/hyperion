"""Carga y filtrado de ``windows.csv`` producidos por
``orchestrator/postprocess.py``.

Cada corrida escribe su propio ``windows.csv`` bajo
``<campaign_dir>/<run_id>/windows.csv`` -- este módulo los concatena y aplica
el filtro de "ventana entrenable" que le corresponde a cada dispositivo, en
espejo exacto de los criterios que ``orchestrator/validation.py`` ya usa para
decidir si una corrida se acepta (ver ``validate_windows()``), pero aplicados
ventana por ventana en vez de exigirlos a nivel de corrida completa.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

# Mismo piso que orchestrator/validation.py::_GPU_UTIL_NOISE_FLOOR_PCT --
# duplicado aquí a propósito (import directo del paquete de Fase 1
# introduciría una dependencia real entre classifier/ y orchestrator/ por un
# solo número; si ese piso cambia, actualizar ambos lados es más barato que
# acoplar los dos paquetes).
_GPU_UTIL_NOISE_FLOOR_PCT = 5.0


def load_campaign_windows(campaign_dir: str | Path, run_id_glob: str = "*") -> pd.DataFrame:
    """Concatena ``windows.csv`` de todas las corridas bajo ``campaign_dir``
    cuyo nombre de directorio matchea ``run_id_glob``. Por defecto ('*')
    incluye todo lo que exista bajo ``campaign_dir`` -- si el directorio
    mezcla corridas de telemetría con ``__baseline``/``_calibration`` (como
    la campaña CPU original), acotar el glob explícitamente
    (p.ej. ``'*__rep[0-9][0-9]'`` sin ``__baseline``) o filtrar después por
    ``kernel_ref``/``freq_level_id``."""
    campaign_dir = Path(campaign_dir)
    frames = []
    for windows_path in sorted(campaign_dir.glob(f"{run_id_glob}/windows.csv")):
        frames.append(pd.read_csv(windows_path))
    if not frames:
        raise FileNotFoundError(f"ningún windows.csv encontrado bajo {campaign_dir}/{run_id_glob}")
    return pd.concat(frames, ignore_index=True)


def load_run_matrix(campaign_dir: str | Path, run_id_template: str, **axes: list) -> pd.DataFrame:
    """Construye explícitamente el producto cartesiano de ``axes`` (p.ej.
    ``kernel=[...], level=[...], rep=range(1, 11)``), lo interpola en
    ``run_id_template`` (una f-string con esos mismos nombres, p.ej.
    ``"{campaign_id}__{kernel}__{level}__rep{rep:02d}"``) y concatena el
    ``windows.csv`` de cada corrida que exista.

    Deliberadamente NO usa un glob sobre el directorio -- la campaña CPU
    mezcla, en el mismo ``campaign_dir``, las corridas de telemetría reales
    con sus pares ``__baseline`` (``perf_enabled=False``, sin telemetría
    real) y con las corridas de calibración (``rep00``); listar la matriz
    explícita evita arrastrar cualquiera de las dos por accidente. Una
    corrida ausente se omite con una advertencia, no revienta la carga
    completa -- útil para trabajar con una matriz todavía incompleta."""
    campaign_dir = Path(campaign_dir)
    keys = list(axes.keys())
    frames = []
    missing = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        run_id = run_id_template.format(**dict(zip(keys, combo)))
        windows_path = campaign_dir / run_id / "windows.csv"
        if windows_path.exists():
            frames.append(pd.read_csv(windows_path))
        else:
            missing.append(run_id)
    if not frames:
        raise FileNotFoundError(
            f"ninguna corrida de la matriz tiene windows.csv bajo {campaign_dir}"
        )
    if missing:
        preview = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        print(f"[load_run_matrix] {len(missing)} corridas sin windows.csv, omitidas: {preview}")
    return pd.concat(frames, ignore_index=True)


def filter_cpu_trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Ventanas de CPU utilizables para entrenar: calidad general ``ok``,
    frecuencia clasificada como válida o no aplicable (gobernador nativo,
    ver ARC-174), y con etiqueta de fase asignada."""
    mask = (
        (df["quality_status"] == "ok")
        & df["frequency_quality_status"].isin(["valid", "not_applicable_native"])
        & df["phase_label_train"].notna()
        & (df["phase_label_train"] != "")
    )
    return df.loc[mask].copy()


def filter_gpu_trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Ventanas de GPU utilizables para entrenar: telemetría de GPU con
    utilización en o por encima del piso de ruido del sensor (5%, el mismo
    piso que ``validate_windows()`` ya exige) y con etiqueta de fase
    asignada."""
    mask = (
        (df["quality_status"] == "gpu_telemetry")
        & (pd.to_numeric(df["gpu_util_pct"], errors="coerce") >= _GPU_UTIL_NOISE_FLOOR_PCT)
        & df["phase_label_train"].notna()
        & (df["phase_label_train"] != "")
    )
    return df.loc[mask].copy()
