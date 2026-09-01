"""Porta EXACTAMENTE los mismos casos que
common/telemetry/tests/test_gpu_clock_controller.cpp, para verificar que
fase3_daemon/gpu_loop/controller.py replica la semántica del original C++
(histéresis, min_dwell_ns, y que un setter que falla no corrompe el
estado interno) -- no es un test genérico nuevo, es la prueba de que el
puerto es fiel al original.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.controller import (
    GpuClockController,
    GpuClockControllerConfig,
    GpuPhaseLabel,
)


def _make_controller(setter):
    config = GpuClockControllerConfig(
        min_dwell_ns=1_000_000_000,  # 1s, igual que el test C++
        compute_bound_clock_mhz=1410,
        memory_bound_clock_mhz=900,
    )
    return GpuClockController(config, setter)


def test_secuencia_completa_identica_al_test_cpp():
    applied_clocks: list[int] = []

    def setter(mhz: int) -> bool:
        applied_clocks.append(mhz)
        return True

    controller = _make_controller(setter)

    # Primera llamada: siempre aplica, sin importar el dwell.
    decision = controller.on_phase_begin(GpuPhaseLabel.COMPUTE_BOUND, now_ns=0)
    assert decision.label == GpuPhaseLabel.COMPUTE_BOUND
    assert decision.clock_changed
    assert decision.applied_clock_mhz == 1410
    assert applied_clocks == [1410]

    # Segunda fase, misma etiqueta -- nada que hacer, incluso dentro del dwell.
    decision = controller.on_phase_begin(GpuPhaseLabel.COMPUTE_BOUND, now_ns=100)
    assert not decision.clock_changed
    assert applied_clocks == [1410]

    # Tercera fase, memory-bound, pero todavía dentro del piso de 1s desde
    # el último (y único) cambio en t=0 -- debe suprimirse.
    decision = controller.on_phase_begin(GpuPhaseLabel.MEMORY_BOUND, now_ns=500_000_000)
    assert not decision.clock_changed
    assert decision.dwell_remaining_ns == 500_000_000
    assert decision.applied_clock_mhz == 1410  # sigue en el reloj viejo
    assert applied_clocks == [1410]

    # Cuarta fase, misma etiqueta memory-bound, ya pasado el piso de dwell
    # (1_500_000_000 - 0 >= 1_000_000_000) -- debe aplicar esta vez.
    decision = controller.on_phase_begin(GpuPhaseLabel.MEMORY_BOUND, now_ns=1_500_000_000)
    assert decision.label == GpuPhaseLabel.MEMORY_BOUND
    assert decision.clock_changed
    assert decision.applied_clock_mhz == 900
    assert applied_clocks == [1410, 900]


def test_setter_que_falla_no_corrompe_el_estado():
    def failing_setter(_mhz: int) -> bool:
        return False

    flaky = _make_controller(failing_setter)
    decision = flaky.on_phase_begin(GpuPhaseLabel.COMPUTE_BOUND, now_ns=0)
    assert decision.clock_changed  # se intentó
    assert decision.clock_setter_failed
    assert not flaky.has_applied_once
    assert flaky.current_clock_mhz == 0
