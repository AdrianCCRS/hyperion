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


def filter_gpu_trainable(
    df: pd.DataFrame,
    *,
    idle_power_mw_by_level: dict[str, float] | None = None,
    active_power_margin_mw: dict[str, float] | float | None = None,
) -> pd.DataFrame:
    """Ventanas de GPU utilizables para entrenar, en espejo del criterio
    que ``validate_windows()`` usa para aceptar la corrida (ARC-185).

    Por defecto (sin los dos parámetros de potencia) usa el piso de
    utilización de siempre. Si se pasan ``idle_power_mw_by_level`` y
    ``active_power_margin_mw``, cambia al criterio invariante a la
    frecuencia -- potencia sobre la línea de reposo medida por nivel de
    reloj -- que reemplaza al piso de utilización porque
    ``gpu_util_pct`` es una fracción de TIEMPO y crece con reloj más
    lento aunque el kernel esté genuinamente ocioso (medido: rodinia_lud
    pasa de 0.0 % en F0 a 3.5 % en F4). Sin línea de reposo para el nivel
    de una fila, esa fila se descarta (fail-closed, no se asume 0).

    ARC-189: ``active_power_margin_mw`` DEBE variar por nivel -- el
    exceso de potencia de trabajo GPU real escala casi tanto con el reloj
    como la propia potencia de reposo (medido: ~9.5-12.7 W de exceso
    mínimo en ventanas claramente activas a F0, contra ~1.3-3.8 W del
    MISMO régimen de actividad a F4, en tres kernels de referencia
    distintos). Un margen único calibrado en F0 rechaza F4 completo. Se
    acepta un ``float`` suelto solo por compatibilidad -- ver
    ``validate_windows`` para el mismo razonamiento."""
    base_mask = (
        (df["quality_status"] == "gpu_telemetry")
        & df["phase_label_train"].notna()
        & (df["phase_label_train"] != "")
    )
    if idle_power_mw_by_level is not None and active_power_margin_mw is not None:
        idle = df["gpu_freq_level_id"].map(idle_power_mw_by_level)
        margin = (
            df["gpu_freq_level_id"].map(active_power_margin_mw)
            if isinstance(active_power_margin_mw, dict)
            else active_power_margin_mw
        )
        power = pd.to_numeric(df["gpu_power_mw"], errors="coerce")
        signal_mask = idle.notna() & margin.notna() if isinstance(margin, pd.Series) else idle.notna()
        signal_mask = signal_mask & ((power - idle) >= margin)
    else:
        signal_mask = pd.to_numeric(df["gpu_util_pct"], errors="coerce") >= _GPU_UTIL_NOISE_FLOOR_PCT
    return df.loc[base_mask & signal_mask].copy()
