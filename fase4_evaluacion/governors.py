"""Conmutación real de gobernador nativo de Linux para el escenario 1 de
§5.1 del plan de realineación ("Gobernador nativo de Linux (ondemand/
schedutil...)").

Hallazgo de la auditoría exclusiva de código que motivó este módulo: antes
de esta reconstrucción, `orchestrator/freqctl.py` (ahora `common/hpc/
freqctl.py`) tenía un modo `native_governor` que solo significa "dejar el
`scaling_governor` que el nodo ya tenía puesto" -- nunca conmutar hacia
`ondemand`/`schedutil` explícitamente y correr el catálogo bajo cada uno.
Cero apariciones de esas dos cadenas en todo el repositorio, en cualquiera
de las dos ramas de origen.

Construido sobre `common.hpc.freqctl.set_governor()`/`read_governors()`
(nuevos, añadidos en esta reconstrucción como funciones aditivas del mismo
módulo, con la misma disciplina de escritura+verificación por relectura,
reintentos ante rechazo transitorio bajo carga -- ARC-108 -- que el resto
de `freqctl.py` ya tenía) -- no reimplementa esa disciplina aquí.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.hpc import freqctl

logger = logging.getLogger(__name__)

# Los 3 escenarios de §5.1: 2 gobernadores nativos reactivos + performance
# (frecuencia fija de alto rendimiento, tratado aquí también como una
# conmutación de gobernador -- más simple y más verificable por relectura
# que reutilizar el camino de apply_frequency(native_governor) existente,
# que asume "lo que ya estuviera puesto").
SCENARIO_GOVERNORS: tuple[str, ...] = ("ondemand", "schedutil", "performance")


class GovernorNotAvailableError(RuntimeError):
    """El gobernador pedido no está en scaling_available_governors del nodo."""


def available_governors(cpus: Iterable[int], env: Any) -> set[str]:
    """Lee `scaling_available_governors` (lista separada por espacios,
    mismo archivo sysfs para todos los CPUs de una misma política cpufreq
    en la práctica, pero se lee por CPU delegado para no asumirlo) --
    intersección entre todos los CPUs pedidos, nunca la unión: un
    gobernador solo cuenta como disponible si lo está en TODOS los CPUs
    que la evaluación va a tocar.
    """
    control_paths = getattr(env, "frequency_control_paths", {}) or {}
    per_cpu_sets: list[set[str]] = []
    for cpu in cpus:
        governor_path_str = control_paths.get(cpu, {}).get("scaling_governor")
        if not governor_path_str:
            per_cpu_sets.append(set())
            continue
        available_path = Path(governor_path_str).parent / "scaling_available_governors"
        try:
            text = available_path.read_text(encoding="utf-8").strip()
        except OSError:
            per_cpu_sets.append(set())
            continue
        per_cpu_sets.append(set(text.split()))
    if not per_cpu_sets:
        return set()
    result = per_cpu_sets[0]
    for s in per_cpu_sets[1:]:
        result &= s
    return result


@contextmanager
def governor_scenario(cpus: Iterable[int], governor: str, env: Any) -> Iterator[None]:
    """Context manager: snapshotea el gobernador actual, lo cambia a
    `governor`, y lo restaura al salir -- SIEMPRE, incluso si el bloque
    lanza. No usa `freqctl.snapshot_original_state`/`restore_original_state`
    (esas dos solo restauran `scaling_governor` bajo `STRATEGY_DISCRETE`,
    ver el docstring de `freqctl.set_governor` para por qué mezclarlas
    sería peligroso) -- hace su propio snapshot/restore mínimo con
    `read_governors`/`set_governor`.

    Uso típico (una corrida de fase1_telemetria/runner.py dentro del
    bloque, por escenario):

        with governor_scenario(env.delegated_cpus, "ondemand", env):
            runner.run_single(...)
    """
    cpus = tuple(cpus)
    supported = available_governors(cpus, env)
    if governor not in supported:
        raise GovernorNotAvailableError(
            f"gobernador {governor!r} no está en scaling_available_governors "
            f"de los CPUs {cpus} (disponibles: {sorted(supported) or 'ninguno detectado'})"
        )

    original = freqctl.read_governors(cpus, env)
    logger.info("governors: conmutando %s -> %s (original: %s)", cpus, governor, original)
    results = freqctl.set_governor(cpus, governor, env)
    failed = [cpu for cpu, ok in results.items() if not ok]
    if failed:
        # Restaurar lo que sí se alcanzó a cambiar antes de fallar
        # ruidosamente -- nunca dejar el nodo en un gobernador mixto sin
        # que quien llamó se entere.
        _restore(original, env)
        raise freqctl.FrequencyControlError(
            f"governors: no se pudo verificar {governor!r} en cpus {failed}"
        )

    try:
        yield
    finally:
        _restore(original, env)


def _restore(original: dict[int, str | None], env: Any) -> None:
    by_target: dict[str, list[int]] = {}
    for cpu, governor in original.items():
        if governor is None:
            continue
        by_target.setdefault(governor, []).append(cpu)
    for governor, cpus in by_target.items():
        results = freqctl.set_governor(tuple(cpus), governor, env)
        failed = [cpu for cpu, ok in results.items() if not ok]
        if failed:
            logger.error(
                "governors: restauración a %r falló para los cpus %s -- "
                "el nodo puede haber quedado en un gobernador distinto al original",
                governor, failed,
            )
