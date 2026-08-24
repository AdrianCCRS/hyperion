"""Protocolo de validación de Fase 2.

El dataset tiene 9.95 M de ventanas entrenables pero **solo 9 kernels**. Un
split aleatorio pone ventanas de la misma corrida en entrenamiento y en
prueba, y el modelo alcanza exactitud cercana a 1.0 reconociendo el kernel
en vez del régimen de ejecución. El número sería espectacular y no
significaría nada.

Por eso el único protocolo honesto aquí es **leave-one-kernel-out**: el
kernel de prueba no aparece en entrenamiento en ninguna de sus
repeticiones ni niveles de frecuencia. Y la dispersión entre pliegues
importa tanto como la media: si un kernel se desploma, eso ES el resultado,
no un detalle que se promedia.

Este módulo se escribe ANTES que cualquier entrenamiento a propósito, para
que ningún número del trabajo llegue a existir fuera del protocolo.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def leave_one_kernel_out(
    df: pd.DataFrame,
    kernel_col: str = "kernel_ref",
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Genera ``(idx_train, idx_test, kernel_excluido)`` por cada kernel.

    Los índices son posicionales sobre ``df`` tal como se recibe. El orden
    de los pliegues es el alfabético de los kernels, para que dos
    ejecuciones den los mismos pliegues sin depender del orden de las filas.
    """
    kernels = sorted(df[kernel_col].dropna().unique())
    if len(kernels) < 2:
        raise ValueError(
            f"hacen falta al menos 2 kernels para LOKO, hay {len(kernels)}"
        )
    values = df[kernel_col].to_numpy()
    positions = np.arange(len(df))
    for kernel in kernels:
        held_out = values == kernel
        yield positions[~held_out], positions[held_out], kernel


def assert_no_kernel_leak(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    kernel_col: str = "kernel_ref",
) -> None:
    """Falla si algún kernel aparece en ambos lados del split.

    Guardarraíl explícito: es exactamente el error que produce métricas
    infladas y que este módulo existe para impedir.
    """
    train_kernels = set(df.iloc[idx_train][kernel_col].unique())
    test_kernels = set(df.iloc[idx_test][kernel_col].unique())
    shared = train_kernels & test_kernels
    if shared:
        raise AssertionError(f"fuga de kernel entre train y test: {sorted(shared)}")


def fold_summary(scores: dict[str, float]) -> dict[str, float]:
    """Resume los resultados por pliegue: media, desviación, peor y mejor.

    Devuelve también ``worst_kernel`` porque en LOKO con 9 pliegues el
    kernel que peor generaliza es un resultado en sí mismo, no ruido a
    promediar.
    """
    if not scores:
        raise ValueError("no hay pliegues que resumir")
    values = np.array(list(scores.values()), dtype=float)
    worst_kernel = min(scores, key=lambda k: scores[k])
    best_kernel = max(scores, key=lambda k: scores[k])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "worst_kernel": worst_kernel,
        "best_kernel": best_kernel,
        "n_folds": len(scores),
    }


def edp_loss(chosen_edp: np.ndarray, oracle_edp: np.ndarray) -> float:
    """EDP alcanzado siguiendo el modelo ÷ EDP del óptimo con oráculo.

    Es la métrica que de verdad importa para la decisión de frecuencia:
    1.0 significa que el modelo eligió tan bien como el óptimo, 1.10 que
    gastó un 10% más de EDP del necesario.

    Se prefiere al "acierto del argmin" porque un modelo que falla el
    argmin pero elige una frecuencia casi tan buena es aceptable, mientras
    que uno que acierta a menudo pero cuando falla lo hace catastróficamente
    no lo es -- y la exactitud del argmin no distingue esos dos casos.
    """
    chosen = np.asarray(chosen_edp, dtype=float)
    oracle = np.asarray(oracle_edp, dtype=float)
    if chosen.shape != oracle.shape:
        raise ValueError("chosen_edp y oracle_edp deben tener la misma forma")
    valid = np.isfinite(chosen) & np.isfinite(oracle) & (oracle > 0)
    if not valid.any():
        return float("nan")
    return float(chosen[valid].sum() / oracle[valid].sum())


def trivial_baselines(
    edp_by_level: pd.DataFrame,
    max_level: str,
) -> dict[str, float]:
    """EDP loss de las líneas base tontas, que son obligatorias.

    ``edp_by_level`` tiene una fila por decisión y una columna por nivel de
    frecuencia, con el EDP que se habría obtenido en cada uno.

    Con los datos de CPU actuales el óptimo es la frecuencia máxima en 9 de
    9 kernels, así que "siempre al máximo" logra EDP loss = 1.0 y cualquier
    modelo aprendido tiene que empatarlo o ganarle para justificar su
    existencia. Reportar la métrica del modelo sin esta comparación haría
    pasar por logro lo que es el comportamiento por defecto.
    """
    oracle = edp_by_level.min(axis=1).to_numpy()
    out = {
        "siempre_maxima": edp_loss(edp_by_level[max_level].to_numpy(), oracle),
        "oraculo": 1.0,
    }
    n_levels = edp_by_level.shape[1]
    if n_levels:
        # "al azar" = media sobre niveles, el valor esperado de elegir uno
        # cualquiera con probabilidad uniforme.
        out["al_azar"] = edp_loss(
            edp_by_level.mean(axis=1).to_numpy(), oracle
        )
    return out


def honest_constant_baseline(edp_by_level: pd.DataFrame) -> dict[str, object]:
    """V6/C7: EDP loss de "la mejor frecuencia constante única", elegida
    de forma honesta -- para cada kernel dejado fuera, la constante se
    calcula únicamente con los kernels de entrenamiento del pliegue LOKO
    correspondiente.

    Este es el rival que de verdad hay que vencer, no ``siempre_maxima``:
    es la línea base que ``gpu_policy_headroom.py``/``cpu_policy_headroom.py``
    ya calculan mirando TODO el conjunto -- lo cual hace trampa si se usa
    para evaluar un modelo, porque incorpora información del propio kernel
    de prueba. Aquí se recalcula por pliegue para que la comparación con un
    modelo entrenado bajo LOKO sea justa.

    ``edp_by_level`` tiene una fila por kernel (índice = nombre del
    kernel) y una columna por nivel de frecuencia, con el EDP que ese
    kernel habría obtenido en cada uno.
    """
    kernels = list(edp_by_level.index)
    if len(kernels) < 2:
        raise ValueError(f"hacen falta al menos 2 kernels, hay {len(kernels)}")

    oracle = edp_by_level.min(axis=1)
    chosen = pd.Series(index=kernels, dtype=float)
    chosen_level = {}
    for test_kernel in kernels:
        train = edp_by_level.drop(index=test_kernel)
        level = train.mean(axis=0).idxmin()
        chosen[test_kernel] = edp_by_level.loc[test_kernel, level]
        chosen_level[test_kernel] = level

    return {
        "edp_loss": edp_loss(chosen.to_numpy(), oracle.to_numpy()),
        "chosen_level_by_fold": chosen_level,
        "n_distinct_levels": len(set(chosen_level.values())),
    }
