"""Construcción de los objetivos de entrenamiento de Fase 2.

Dos objetivos, que responden a preguntas distintas:

``b`` -- **qué está pasando ahora**. Grado de acotamiento continuo en [0,1],
con 0 = compute_bound y 1 = memory_bound. No es una etiqueta nueva: es la
magnitud continua que la etiqueta binaria de Fase 1 ya estaba
discretizando, expresada sin tirar la información de "qué tan lejos del
umbral".

``alpha`` -- **qué pasaría si cambio la frecuencia**. Vive en
``align.fit_alpha()`` porque necesita varias frecuencias del mismo tramo,
no una sola ventana.

Sobre el ridge: ``i_ridge_used`` NO es constante, depende de la frecuencia
a la que se tomó la telemetría (cae de 8.733 a 2.992 FLOP/byte entre 3200 y
800 MHz, porque el pico de cómputo baja con el reloj y el ancho de banda
casi no). Por eso ``b`` siempre debe calcularse con el ridge de la MISMA
fila, nunca con uno fijo de la campaña -- es lo que hace que el score sea
coherente con lo que el agente observará en ejecución.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Umbral que separa las dos clases. No es configurable a propósito: 0.5
# corresponde exactamente a OI == I_ridge, que es la frontera que Fase 1 ya
# usa (postprocess.py::_finalize_operational_intensity). Moverlo rompería
# la equivalencia con la etiqueta binaria.
DECISION_THRESHOLD = 0.5


def boundedness_score(
    operational_intensity: pd.Series | np.ndarray,
    i_ridge: pd.Series | np.ndarray,
    k: float = 1.0,
) -> np.ndarray:
    """``b = sigma(-k * log10(OI / I_ridge))`` en [0,1].

    0 = compute_bound, 1 = memory_bound -- la convención pedida.

    Propiedad que lo hace una generalización estricta y no un reemplazo
    arbitrario: en ``OI == I_ridge`` da exactamente 0.5, así que umbralizar
    en 0.5 reproduce la etiqueta binaria de Fase 1 fila por fila
    (ver ``agreement_with_binary_label``).

    ``k`` solo controla lo abrupta que es la transición; no mueve la
    frontera. Calibrarlo con ``calibrate_k`` en vez de elegirlo a dedo.
    Valores no positivos o no finitos de OI o del ridge dan NaN: el
    logaritmo no está definido y fabricar un 0.5 los haría pasar por
    "justo en la frontera", que es la peor confusión posible aquí.
    """
    oi = pd.to_numeric(pd.Series(operational_intensity), errors="coerce").to_numpy(dtype=float)
    ridge = pd.to_numeric(pd.Series(i_ridge), errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(oi) & np.isfinite(ridge) & (oi > 0) & (ridge > 0)
    result = np.full(oi.shape, np.nan, dtype=float)
    z = -k * np.log10(oi[valid] / ridge[valid])
    result[valid] = 1.0 / (1.0 + np.exp(-z))
    return result


def calibrate_k(
    operational_intensity: pd.Series | np.ndarray,
    i_ridge: pd.Series | np.ndarray,
) -> float:
    """``k = 1 / sd(log10(OI/I_ridge))``.

    Hace que una desviación estándar de distancia al ridge caiga en
    ``b ~ 0.27`` / ``0.73``: la sigmoide usa su zona sensible justo donde
    los datos tienen dispersión, en vez de saturar a 0 y 1 con todo.
    """
    oi = pd.to_numeric(pd.Series(operational_intensity), errors="coerce").to_numpy(dtype=float)
    ridge = pd.to_numeric(pd.Series(i_ridge), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(oi) & np.isfinite(ridge) & (oi > 0) & (ridge > 0)
    if not valid.any():
        raise ValueError("no hay filas con OI y ridge positivos para calibrar k")
    sd = float(np.std(np.log10(oi[valid] / ridge[valid])))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("la dispersión de log10(OI/ridge) es nula: k no es calibrable")
    return 1.0 / sd


def binary_from_score(scores: np.ndarray) -> np.ndarray:
    """Devuelve las etiquetas de Fase 1 a partir de ``b``.

    ``>`` y no ``>=``: en el empate exacto (``b == 0.5``, o sea
    ``OI == I_ridge``) Fase 1 clasifica como compute_bound, porque su
    condición es ``memory_bound if OI < ridge``. Invertir esto desalinearía
    la equivalencia justo en la frontera.
    """
    out = np.full(scores.shape, None, dtype=object)
    finite = np.isfinite(scores)
    out[finite & (scores > DECISION_THRESHOLD)] = "memory_bound"
    out[finite & (scores <= DECISION_THRESHOLD)] = "compute_bound"
    return out


def agreement_with_binary_label(
    df: pd.DataFrame,
    oi_col: str = "operational_intensity_uncore_real",
    ridge_col: str = "i_ridge_used",
    label_col: str = "phase_label_train",
    k: float = 1.0,
) -> tuple[float, int]:
    """Fracción de filas en que umbralizar ``b`` reproduce ``phase_label_train``.

    Es la prueba de que el target continuo generaliza al binario en vez de
    reemplazarlo. Debe dar 1.0; cualquier desacuerdo señala que el ridge de
    la fila no es el que Fase 1 usó para etiquetarla. Devuelve
    ``(concordancia, filas_comparadas)``.
    """
    scores = boundedness_score(df[oi_col], df[ridge_col], k=k)
    derived = binary_from_score(scores)
    original = df[label_col].to_numpy()

    comparable = np.array([
        s is not None and isinstance(o, str) and o != ""
        for s, o in zip(derived, original)
    ])
    if not comparable.any():
        return float("nan"), 0
    matches = derived[comparable] == original[comparable]
    return float(matches.mean()), int(comparable.sum())


def add_targets(
    df: pd.DataFrame,
    oi_col: str = "operational_intensity_uncore_real",
    ridge_col: str = "i_ridge_used",
    k: float | None = None,
) -> pd.DataFrame:
    """Añade la columna ``b`` al DataFrame, calibrando ``k`` si no se da."""
    out = df.copy()
    if k is None:
        k = calibrate_k(out[oi_col], out[ridge_col])
    out["b"] = boundedness_score(out[oi_col], out[ridge_col], k=k)
    return out
