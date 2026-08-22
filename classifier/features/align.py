"""Alineación de fases entre corridas a distinta frecuencia.

El problema que resuelve: el mismo kernel corrido a 3200 MHz y a 800 MHz
produce dos series de ventanas de longitudes distintas. La ventana 50 de
una NO es el mismo punto del programa que la ventana 50 de la otra -- la
lenta está estirada. Para comparar "el mismo momento" entre frecuencias
hace falta una coordenada de progreso que no dependa del reloj.

``delta_instructions`` sirve: un kernel determinista retira las mismas
instrucciones vaya rápido o lento. Verificado sobre la campaña real
(compuerta 1, ARC-175): la desviación del conteo total entre niveles de
frecuencia es de 0.34% en el peor caso y <0.1% en la mayoría, contra un
criterio de +-2%.

En GPU esta coordenada NO existe: las filas de telemetría de GPU son
passthrough (ARC-70) y no traen PMU de CPU. Allí hay que usar
``add_time_progress()``, que es más débil porque asume que el trabajo se
reparte uniformemente en el tiempo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columnas que identifican una corrida única dentro de una campaña.
RUN_KEYS = ("kernel_ref", "freq_level_id", "repetition")

# Columnas que identifican una celda alineada: el mismo tramo lógico del
# programa, del mismo kernel y repetición, a un nivel de frecuencia dado.
CELL_KEYS = ("kernel_ref", "repetition", "progress_bin", "freq_level_id")


def add_instruction_progress(
    df: pd.DataFrame,
    instr_col: str = "delta_instructions",
    run_keys: tuple[str, ...] = RUN_KEYS,
) -> pd.DataFrame:
    """Añade ``progress`` en [0,1]: fracción de instrucciones del programa ya
    retiradas al final de cada ventana.

    Se calcula por corrida (``run_keys``). El progreso es la suma acumulada
    de ``instr_col`` dividida por el total de esa corrida, así que la última
    ventana de cada corrida vale exactamente 1.0.

    Una corrida cuyo total de instrucciones sea 0 o NaN queda con
    ``progress`` NaN en todas sus filas -- no se inventa un progreso
    uniforme, porque eso mezclaría silenciosamente una coordenada válida con
    una fabricada.
    """
    out = df.copy()
    instr = pd.to_numeric(out[instr_col], errors="coerce")
    grouped = instr.groupby([out[k] for k in run_keys])
    cumulative = grouped.cumsum()
    total = grouped.transform("sum")
    out["progress"] = np.where(total > 0, cumulative / total, np.nan)
    return out


def add_time_progress(
    df: pd.DataFrame,
    start_col: str = "t_start_ns",
    end_col: str = "t_end_ns",
    run_keys: tuple[str, ...] = RUN_KEYS,
) -> pd.DataFrame:
    """Variante de respaldo para GPU: progreso como fracción de tiempo
    transcurrido dentro de la corrida.

    Es más débil que ``add_instruction_progress`` porque supone que el
    trabajo avanza uniformemente en el tiempo -- justo lo que deja de ser
    cierto al cambiar la frecuencia. Usar solo donde no haya conteo de
    instrucciones, y decirlo explícitamente en los resultados.
    """
    out = df.copy()
    start = pd.to_numeric(out[start_col], errors="coerce")
    end = pd.to_numeric(out[end_col], errors="coerce")
    keys = [out[k] for k in run_keys]
    t0 = start.groupby(keys).transform("min")
    t1 = end.groupby(keys).transform("max")
    span = t1 - t0
    out["progress"] = np.where(span > 0, (end - t0) / span, np.nan)
    return out


def assign_progress_bins(df: pd.DataFrame, n_bins: int = 100) -> pd.DataFrame:
    """Discretiza ``progress`` en ``n_bins`` tramos iguales, numerados 0..n-1.

    El extremo ``progress == 1.0`` cae en el último bin (no en uno
    inexistente n), que es el motivo del ``clip``: sin él, la última ventana
    de cada corrida -- que siempre vale exactamente 1.0 -- se perdería.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins debe ser positivo, recibido {n_bins}")
    out = df.copy()
    raw = np.floor(pd.to_numeric(out["progress"], errors="coerce") * n_bins)
    out["progress_bin"] = raw.clip(upper=n_bins - 1).astype("Int64")
    return out


def aggregate_cells(
    df: pd.DataFrame,
    feature_cols: list[str],
    energy_col: str = "pkg_delta_uj",
    duration_col: str = "delta_t_ns",
    cell_keys: tuple[str, ...] = CELL_KEYS,
) -> pd.DataFrame:
    """Colapsa las ventanas de cada celda alineada en una sola fila.

    Las *features* se promedian (describen un estado del sistema) mientras
    que energía y duración se SUMAN (son cantidades extensivas del tramo).
    Confundir las dos cosas es el error clásico aquí: promediar la energía
    haría que un tramo largo y uno corto pesaran igual.

    Devuelve, por celda: la media de cada feature, ``energy_uj`` y
    ``duration_ns`` totales, y ``n_windows``.
    """
    work = df.dropna(subset=["progress_bin"]).copy()
    for col in [*feature_cols, energy_col, duration_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    agg: dict[str, tuple[str, str]] = {c: (c, "mean") for c in feature_cols}
    agg["energy_uj"] = (energy_col, "sum")
    agg["duration_ns"] = (duration_col, "sum")
    agg["n_windows"] = (duration_col, "size")

    return work.groupby(list(cell_keys), observed=True).agg(**agg).reset_index()


def fit_alpha(durations_by_freq: dict[float, float], f_ref_mhz: float) -> tuple[float, float]:
    """Ajusta la ley clásica de escalado DVFS a un conjunto de tiempos.

        T(f) / T(f_ref) = (1 - alpha) + alpha * (f_ref / f)

    ``alpha`` es la fracción del tiempo que es SENSIBLE a la frecuencia:
    0.0 significa que el reloj no afecta (todo el tiempo es espera a
    memoria) y 1.0 que el tiempo escala perfectamente con el reloj.

    ``durations_by_freq`` mapea MHz -> duración. Devuelve ``(alpha, r2)``.
    El ajuste es por mínimos cuadrados sin intercepto sobre ``y-1 = a*(x-1)``
    con ``x = f_ref/f``, que impone exactamente lo que la física dice: a la
    frecuencia de referencia el tiempo relativo vale 1 por construcción, así
    que la recta no puede tener un término libre.
    """
    t_ref = durations_by_freq.get(f_ref_mhz)
    if not t_ref or t_ref <= 0:
        raise ValueError(f"falta la duración de referencia a {f_ref_mhz} MHz")

    points = [
        (f_ref_mhz / mhz, duration / t_ref)
        for mhz, duration in sorted(durations_by_freq.items())
        if mhz > 0 and duration and duration > 0
    ]
    if len(points) < 2:
        raise ValueError("hacen falta al menos dos frecuencias para ajustar alpha")

    numerator = sum((x - 1.0) * (y - 1.0) for x, y in points)
    denominator = sum((x - 1.0) ** 2 for x, _ in points)
    alpha = numerator / denominator if denominator else float("nan")

    predicted = [(1.0 - alpha) + alpha * x for x, _ in points]
    observed = [y for _, y in points]
    mean_observed = sum(observed) / len(observed)
    ss_res = sum((y - p) ** 2 for y, p in zip(observed, predicted))
    ss_tot = sum((y - mean_observed) ** 2 for y in observed)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return alpha, r2
