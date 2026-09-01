"""Protocolo de validación de Fase 2.

El dataset original de este módulo (9 kernels: NPB bt/mg/cg/sp/ft/lu,
dgemm_n2048, rodinia_lavamd_omp, rajaperf_polybench_3mm_omp) solo tenía 9.95
M de ventanas entrenables repartidas en **9 kernels**. Un split aleatorio
pone ventanas de la misma corrida en entrenamiento y en prueba, y el modelo
alcanza exactitud cercana a 1.0 reconociendo el kernel en vez del régimen de
ejecución. El número sería espectacular y no significaría nada.

Por eso el protocolo honesto aquí es **agrupado**: el kernel (o, tras la
fusión del catálogo ampliado en la reconstrucción de 4 fases, la FAMILIA
algorítmica) de prueba no aparece en entrenamiento en ninguna de sus
repeticiones, tamaños de problema ni niveles de frecuencia. Y la dispersión
entre pliegues importa tanto como la media: si un kernel/familia se
desploma, eso ES el resultado, no un detalle que se promedia.

⚠️ **Por qué existe `leave_one_familia_out` además de `leave_one_kernel_out`**
(añadido en la reconstrucción en 4 fases, no en la versión original de este
archivo): el catálogo fusionado (`fase1_telemetria/catalog/catalog.yaml`,
232 entradas) tiene familias como `dual_gemm_*` con el MISMO algoritmo
barrido sobre ~16 tamaños de problema distintos en cada dispositivo. Dejar
un solo tamaño de `dual_gemm` fuera de entrenamiento (lo que hacía
`leave_one_kernel_out` con `kernel_col="kernel_ref"`) no prueba
generalización a un algoritmo nuevo, solo a un tamaño nuevo del mismo
algoritmo ya visto en entrenamiento -- sobreestima la generalización real.
`leave_one_familia_out` agrupa por `derive_kernel_family(kernel_ref)`
en vez de por `kernel_ref` crudo. Para el dataset original de 9 kernels
(ninguno comparte familia) el resultado es idéntico a `leave_one_kernel_out`
-- no es un protocolo distinto, es el mismo protocolo con la unidad de
agrupación correcta una vez que el catálogo tiene familias con size-sweeps.

Este módulo se escribe ANTES que cualquier entrenamiento a propósito, para
que ningún número del trabajo llegue a existir fuera del protocolo.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
import pandas as pd

# Sub-suites conocidas de RAJAPerf que el plan de realineación (§2.1.1)
# trata como familias separadas ("RAJAPerf stream/lcals/polybench/basic"),
# no como una sola familia "rajaperf" -- son algoritmos distintos entre sí,
# solo comparten el arnés de medición de RAJAPerf.
_RAJAPERF_SUBSUITES = ("stream", "lcals", "polybench", "basic")

# Patrones de kernel_ref -> familia algorítmica, en el orden en que se
# prueban (el primero que matchea gana). Cada patrón captura el nombre del
# algoritmo y descarta explícitamente lo que NO define una familia distinta:
# el dispositivo (cpu/gpu) y el tamaño de problema (N<size>, class B/C de
# NPB, resolución de dwt2d, parámetro p de phasic).
_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # dual_<algo>_(cpu|gpu)_N<size> -> dual_<algo> (mismo algoritmo,
    # cpu/gpu y tamaño son variantes, no familias nuevas).
    (re.compile(r"^dual_(?P<algo>[a-z0-9]+)_(cpu|gpu)_N\d+$"), "dual_{algo}"),
    # npb_<name> / npb_<name>_c (clase B vs C del mismo problema) -> npb_<name>.
    (re.compile(r"^npb_(?P<name>[a-z]+)(_c)?$"), "npb_{name}"),
    # (cpu_|gpu_)?rajaperf_<subsuite>_<resto> -> rajaperf_<subsuite>, solo
    # para las 4 sub-suites que el plan trata como familias con nombre.
    (
        re.compile(
            r"^(cpu_|gpu_)?rajaperf_(?P<subsuite>" + "|".join(_RAJAPERF_SUBSUITES) + r")_.+$"
        ),
        "rajaperf_{subsuite}",
    ),
    # Kernels RAJAPerf-CUDA sin sub-suite reconocible en el nombre
    # (heat_3d, jacobi_2d, reduce3_int, indexlist_3loop) -> una familia
    # propia del backend CUDA de RAJAPerf, no fusionada con las CPU.
    (re.compile(r"^gpu_rajaperf_.+$"), "rajaperf_cuda"),
    # rodinia_<nombre>[_omp][_s<res>] -> rodinia_<nombre>. Unifica
    # deliberadamente la variante CPU-OpenMP (p.ej. rodinia_lavamd_omp) con
    # su contraparte GPU (rodinia_lavamd): mismo algoritmo, distinto
    # dispositivo, exactamente el criterio de familia de este módulo.
    (re.compile(r"^rodinia_(?P<name>[a-z0-9]+?)(_omp)?(_s\d+)?$"), "rodinia_{name}"),
    # (gpu_)?phasic_p<valor> -> phasic (el parámetro p es una variante de
    # carga, no un algoritmo distinto).
    (re.compile(r"^(gpu_)?phasic_p\d+$"), "phasic"),
    # DGEMM: calibración cuBLAS, N4096 y la versión CPU/OpenBLAS son el
    # mismo algoritmo (producto matriz-matriz denso).
    (re.compile(r"^(gpu_)?dgemm(_.+)?$"), "dgemm"),
    (re.compile(r"^gpu_ert_probe.*$|^ert_probe$"), "ert"),
    (re.compile(r"^gpu_stream_bw$|^stream_official$"), "stream"),
    (re.compile(r"^cpu_gap_(?P<name>[a-z]+)$"), "gap_{name}"),
)


def derive_kernel_family(kernel_ref: str) -> str:
    """Familia algorítmica de un ``kernel_ref``, para agrupar validación.

    Reglas (ver ``_FAMILY_PATTERNS`` arriba para el detalle exacto de cada
    una): variantes de tamaño de problema, clase NPB (B/C), dispositivo
    (cpu/gpu) y parámetros de carga NUNCA definen una familia nueva -- solo
    el algoritmo en sí. Si ningún patrón conocido matchea, la familia es el
    propio ``kernel_ref`` sin cambios (kernels de un solo tamaño/variante,
    como ``ptrchase``, ``cpu_hpcg``, ``cpu_lulesh``, ``cpu_cholmod`` --
    correcto ahí porque no tienen ninguna variante de la que separarse).
    """
    for pattern, template in _FAMILY_PATTERNS:
        match = pattern.match(kernel_ref)
        if match:
            return template.format(**match.groupdict())
    return kernel_ref


def leave_one_kernel_out(
    df: pd.DataFrame,
    kernel_col: str = "kernel_ref",
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Genera ``(idx_train, idx_test, kernel_excluido)`` por cada kernel.

    Los índices son posicionales sobre ``df`` tal como se recibe. El orden
    de los pliegues es el alfabético de los kernels, para que dos
    ejecuciones den los mismos pliegues sin depender del orden de las filas.

    ⚠️ Sobre un dataset con familias de tamaños (p.ej. ``dual_gemm_*``),
    esta función SOBREESTIMA la generalización -- usar
    ``leave_one_familia_out`` en su lugar. Se conserva porque sigue siendo
    correcta para un dataset sin familias repetidas (el caso original de 9
    kernels de este módulo) y porque ``leave_one_familia_out`` la reutiliza
    internamente.
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


def leave_one_familia_out(
    df: pd.DataFrame,
    kernel_col: str = "kernel_ref",
    family_fn=derive_kernel_family,
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Genera ``(idx_train, idx_test, familia_excluida)`` por cada familia
    algorítmica (§2.1.1/§2.6 de ``Plan_Detallado_Realineacion_Hyperion.md``).

    Reutiliza ``leave_one_kernel_out`` sobre una columna auxiliar de familia
    derivada con ``family_fn`` (por defecto ``derive_kernel_family``), en
    vez de reimplementar el mismo bucle -- el contrato (índices
    posicionales, orden alfabético reproducible) es idéntico, solo cambia
    la unidad de agrupación.
    """
    with_family = df.copy()
    with_family["_familia"] = df[kernel_col].map(family_fn)
    yield from leave_one_kernel_out(with_family, kernel_col="_familia")


def assert_no_familia_leak(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    kernel_col: str = "kernel_ref",
    family_fn=derive_kernel_family,
) -> None:
    """Equivalente a ``assert_no_kernel_leak`` pero por familia algorítmica.

    Falla también si dos ``kernel_ref`` DISTINTOS de la misma familia (p.ej.
    ``dual_gemm_cpu_N64`` en train y ``dual_gemm_gpu_N96`` en test) quedan
    repartidos entre los dos lados -- ese es exactamente el escenario de
    fuga que ``leave_one_familia_out`` existe para evitar.
    """
    train_families = {family_fn(k) for k in df.iloc[idx_train][kernel_col].unique()}
    test_families = {family_fn(k) for k in df.iloc[idx_test][kernel_col].unique()}
    shared = train_families & test_families
    if shared:
        raise AssertionError(f"fuga de familia algorítmica entre train y test: {sorted(shared)}")


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
