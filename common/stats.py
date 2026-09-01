"""Pruebas de significancia estadística compartidas entre Fase 3 y Fase 4.

Por qué vive en common/ y no en una sola fase: el derivador de la tabla de
política de Fase 3 (fase3_daemon/policy/derive_policy_table.py, §3.5 del
plan de realineación) necesita una prueba pareada por kernel (Wilcoxon o
bootstrap) para decidir si el EDP de un nivel de frecuencia candidato mejora
de forma defendible sobre REF -- y Fase 4 (fase4_evaluacion/, §5.2) necesita
exactamente el mismo tipo de prueba para decidir si el agente mejora sobre
cada gobernador nativo. Duplicar esta lógica en las dos fases arriesgaría
que la definición de "mejora estadísticamente defendible" divergiera entre
la política que se despliega y la evaluación que la juzga.

Hallazgo de la auditoría exclusiva de código que motivó este módulo: no
existía ni un solo uso de `scipy.stats` en todo el repositorio (ninguna de
las dos ramas de origen) -- el análisis de EDP existente
(`fase2_clasificador/eval/protocol.py`) compara magnitudes (razones de EDP)
pero nunca produce un p-valor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignificanceResult:
    """Resultado de una prueba pareada entre dos condiciones (p.ej. EDP a
    REF vs. EDP a un nivel candidato, o EDP del agente vs. un gobernador).
    """
    test_name: str
    statistic: float
    p_value: float
    n_pairs: int
    significant: bool
    alpha: float

    def __bool__(self) -> bool:
        return self.significant


def paired_significance_test(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    alpha: float = 0.05,
    force_test: str | None = None,
) -> SignificanceResult:
    """Prueba pareada entre ``baseline`` y ``candidate`` (misma longitud,
    un par por kernel/operación), eligiendo la prueba según la distribución
    de las diferencias -- nunca t-test pareado por defecto sin comprobar el
    supuesto de normalidad que exige.

    Criterio: con menos de 8 pares no hay suficiente potencia para un test
    de normalidad fiable (Shapiro-Wilk), así que se usa directamente
    Wilcoxon (no paramétrico, sin ese supuesto). Con 8 o más pares, se
    corre Shapiro-Wilk sobre las diferencias; si no se puede rechazar
    normalidad (p >= alpha), se usa t-test pareado (más potente cuando el
    supuesto se sostiene); si se rechaza, Wilcoxon. ``force_test``
    (``"wilcoxon"``, ``"ttest"`` o ``"mannwhitney"``) salta esta selección
    automática -- para cuando el capítulo de resultados necesita reportar
    una prueba específica de forma consistente entre kernels, en vez de que
    la prueba elegida varíe de uno a otro según su propia distribución.
    """
    from scipy import stats

    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline y candidate deben tener la misma forma (pares)")
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    baseline, candidate = baseline[valid], candidate[valid]
    n_pairs = len(baseline)
    if n_pairs < 2:
        raise ValueError(f"hacen falta al menos 2 pares válidos, hay {n_pairs}")

    differences = candidate - baseline
    if np.allclose(differences, 0.0):
        # Wilcoxon lanza ValueError si todas las diferencias son cero
        # ("wilcoxon: all differences are zero") -- caso legítimo (p.ej.
        # comparar REF contra sí mismo), no un error de uso.
        return SignificanceResult(
            test_name="sin_diferencia", statistic=0.0, p_value=1.0,
            n_pairs=n_pairs, significant=False, alpha=alpha,
        )

    test_name = force_test
    if test_name is None:
        if n_pairs >= 8:
            _, normality_p = stats.shapiro(differences)
            test_name = "ttest" if normality_p >= alpha else "wilcoxon"
        else:
            test_name = "wilcoxon"

    if test_name == "ttest":
        statistic, p_value = stats.ttest_rel(candidate, baseline)
    elif test_name == "wilcoxon":
        statistic, p_value = stats.wilcoxon(candidate, baseline)
    elif test_name == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(candidate, baseline)
    else:
        raise ValueError(f"force_test desconocido: {test_name!r}")

    return SignificanceResult(
        test_name=test_name, statistic=float(statistic), p_value=float(p_value),
        n_pairs=n_pairs, significant=bool(p_value < alpha), alpha=alpha,
    )
