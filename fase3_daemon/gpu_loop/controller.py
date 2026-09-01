"""Puerto en Python de telemetry::GpuClockController
(common/telemetry/include/telemetry/gpu_clock_controller.hpp).

Por qué existe un puerto en vez de enlazar el C++ directamente: el loop de
GPU del daemon (§4.3 punto 2 del plan de realineación) va en Python, no en
C++ -- reutilizando `common/hpc/gpu_freqctl.py` para la actuación real.
Enlazar la clase C++ desde Python (pybind11 o similar) sería más código de
build sin ganar nada: la clase es una pura máquina de estados sin llamadas a
NVML/CUDA (ver el comentario del header original), así que portarla es un
ejercicio mecánico y verificable con el mismo tipo de test que ya usa
`test_gpu_clock_controller.cpp`.

Esta clase NO clasifica nada por sí sola -- igual que el original. Recibe
una etiqueta ya decidida (por el modelo de Fase 2 en producción, o por una
etiqueta derivada de ncu en una campaña de caracterización offline) y solo
decide si vale la pena pagar el costo de una transición de reloj ahora
mismo, con histéresis (`min_dwell_ns`, que debe fijarse desde una medición
real de T_transición_gpu, §2.4.1 -- no existe esa medición todavía en
ningún lado del proyecto).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class GpuPhaseLabel(str, Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"


@dataclass(frozen=True)
class GpuClockControllerConfig:
    """Ver el docstring de `min_dwell_ns` en gpu_clock_controller.hpp: debe
    fijarse desde T_transición_gpu MEDIDO (min_dwell_ns >= 10 * T_transición_ns
    es un punto de partida razonable), nunca asumido de la ficha técnica.
    """
    min_dwell_ns: int
    compute_bound_clock_mhz: int
    memory_bound_clock_mhz: int


@dataclass
class GpuPhaseDecision:
    """Resultado de una llamada a on_phase_begin(), para logging/export a CSV
    -- mismo propósito que GpuPhaseDecision en el header original."""
    label: GpuPhaseLabel
    target_clock_mhz: int
    applied_clock_mhz: int = 0
    clock_changed: bool = False
    clock_setter_failed: bool = False
    dwell_remaining_ns: int = 0


class GpuClockController:
    """Máquina de histéresis + `min_dwell_ns`. Sin llamadas a NVML/nvidia-smi
    propias -- recibe `set_clock` inyectado (mismo patrón que
    `orchestrator/campaign.py` inyecta `apply_frequency()` para CPU), así que
    se puede probar sin GPU real. En producción, `set_clock` envuelve
    `common.hpc.gpu_freqctl.apply_gpu_frequency`.
    """

    def __init__(self, config: GpuClockControllerConfig, set_clock: Callable[[int], bool]) -> None:
        self._config = config
        self._set_clock = set_clock
        self._current_label: GpuPhaseLabel = GpuPhaseLabel.COMPUTE_BOUND
        self._current_clock_mhz: int = 0
        self._last_change_ns: int = 0
        self._has_applied_once: bool = False

    @property
    def current_label(self) -> GpuPhaseLabel:
        return self._current_label

    @property
    def current_clock_mhz(self) -> int:
        return self._current_clock_mhz

    @property
    def has_applied_once(self) -> bool:
        return self._has_applied_once

    def on_phase_begin(self, label: GpuPhaseLabel, now_ns: int) -> GpuPhaseDecision:
        """Llamar una vez por frontera de fase, nunca una vez por muestra --
        ver el comentario de archivo del header original para por qué (el
        costo de una transición de reloj de GPU es un orden de magnitud
        mayor que escribir scaling_min_freq en CPU)."""
        desired_clock_mhz = (
            self._config.compute_bound_clock_mhz
            if label == GpuPhaseLabel.COMPUTE_BOUND
            else self._config.memory_bound_clock_mhz
        )
        decision = GpuPhaseDecision(label=label, target_clock_mhz=desired_clock_mhz)

        if not self._has_applied_once:
            self._apply(decision, desired_clock_mhz, now_ns)
            self._current_label = label
            return decision

        if desired_clock_mhz == self._current_clock_mhz:
            decision.applied_clock_mhz = self._current_clock_mhz
            decision.clock_changed = False
            decision.dwell_remaining_ns = 0
            return decision

        elapsed_ns = now_ns - self._last_change_ns
        if elapsed_ns < self._config.min_dwell_ns:
            decision.applied_clock_mhz = self._current_clock_mhz
            decision.clock_changed = False
            decision.dwell_remaining_ns = self._config.min_dwell_ns - elapsed_ns
            return decision

        self._apply(decision, desired_clock_mhz, now_ns)
        self._current_label = label
        return decision

    def _apply(self, decision: GpuPhaseDecision, desired_clock_mhz: int, now_ns: int) -> None:
        ok = self._set_clock(desired_clock_mhz)
        decision.clock_changed = True
        decision.clock_setter_failed = not ok
        if ok:
            self._current_clock_mhz = desired_clock_mhz
            self._last_change_ns = now_ns
            self._has_applied_once = True
        # Si falla, NO se actualiza current_clock_mhz_/last_change_ns_ a
        # propósito: la GPU sigue al reloj que tenía antes, y la próxima
        # llamada debe reintentar en vez de creer un cambio que no ocurrió.
        decision.applied_clock_mhz = self._current_clock_mhz
        decision.dwell_remaining_ns = 0
