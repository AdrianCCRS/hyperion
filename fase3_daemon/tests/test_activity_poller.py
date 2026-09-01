"""Tests de fase3_daemon/gpu_loop/activity_poller.py -- la fuente de
eventos de fase elegida (Opción C) tras confirmar que la intercepción de
cudaLaunchKernel (fase3_daemon/shim/, eliminado) no funciona."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.activity_poller import poll_phase_events
from fase3_daemon.gpu_loop.loop import GpuFeatures


def _features(util: float) -> GpuFeatures:
    return GpuFeatures(
        gpu_util_pct=util, gpu_mem_util_pct=1.0, gpu_power_mw=1.0,
        gpu_sm_clock_mhz=1.0, gpu_temperature_c=1.0,
    )


def _fake_source(readings):
    """Convierte una lista de lecturas (float de gpu_util_pct, o None) en
    query_features_fn -- cada lectura no-None se envuelve en GpuFeatures."""
    it = iter(readings)
    def query():
        value = next(it)
        return None if value is None else _features(value)
    return query


def test_emite_begin_solo_en_transicion_idle_a_activo():
    # idle, idle, activo, activo, activo, idle, activo -> 2 transiciones
    # idle->activo (posiciones 2 y 6), nunca un evento por cada muestra activa.
    readings = [1.0, 2.0, 50.0, 60.0, 55.0, 1.0, 80.0]
    query = _fake_source(readings)
    now_values = iter(range(len(readings)))

    events = list(poll_phase_events(
        query, activity_threshold_pct=5.0,
        now_fn=lambda: next(now_values), sleep_fn=lambda _s: None,
        max_events=2,
    ))

    assert len(events) == 2
    assert events[0].now_ns == 2  # primera muestra activa (indice 2)
    assert events[0].features.gpu_util_pct == 50.0
    assert events[1].now_ns == 6  # segunda transicion idle->activo (indice 6)


def test_on_end_se_llama_en_transicion_activo_a_idle():
    readings = [50.0, 1.0]  # activo, luego idle -> un solo END, sin más eventos BEGIN
    query = _fake_source(readings)
    now_values = iter(range(10))
    ends = []

    def bounded():
        try:
            return query()
        except StopIteration:
            raise RuntimeError("agotado a propósito, corta el generador")

    with pytest.raises(RuntimeError):
        list(poll_phase_events(
            bounded, activity_threshold_pct=5.0,
            now_fn=lambda: next(now_values), sleep_fn=lambda _s: None,
            on_end=ends.append, max_events=None,
        ))
    assert ends == [1]  # END en el segundo tick (indice 1), cuando pasa a idle


def test_features_none_no_cambia_el_estado():
    # None (sin señal) entre dos lecturas activas no debe generar un
    # segundo BEGIN ni confundirse con una transición. now_fn() solo se
    # consulta en muestras con señal real (índices 0, 2, 4, 5 -- las None
    # en 1 y 3 no avanzan el reloj inyectado), así que el segundo BEGIN
    # (tras el idle real del índice 4) cae en la tercera consulta a
    # now_fn(), no en el índice 5 del arreglo de lecturas.
    readings = [50.0, None, 55.0, None, 1.0, 60.0]
    query = _fake_source(readings)
    now_values = iter(range(len(readings)))

    events = list(poll_phase_events(
        query, activity_threshold_pct=5.0,
        now_fn=lambda: next(now_values), sleep_fn=lambda _s: None,
        max_events=2,
    ))

    assert len(events) == 2
    assert events[0].now_ns == 0   # primera lectura activa (índice 0)
    assert events[0].features.gpu_util_pct == 50.0
    assert events[1].now_ns == 3   # 4ta consulta con señal real (índices 0,2,4,5 -> now_fn 0,1,2,3)
    assert events[1].features.gpu_util_pct == 60.0  # la lectura del índice 5, tras el idle del índice 4


def test_sleep_fn_se_invoca_cada_iteracion():
    readings = [1.0, 1.0, 1.0]
    query = _fake_source(readings)
    sleeps = []

    def bounded():
        try:
            return query()
        except StopIteration:
            raise RuntimeError("fin")

    with pytest.raises(RuntimeError):
        list(poll_phase_events(
            bounded, activity_threshold_pct=5.0,
            now_fn=lambda: 0, sleep_fn=sleeps.append, max_events=None,
        ))
    assert sleeps == [0.05, 0.05, 0.05]  # DEFAULT_POLL_INTERVAL_S, una vez por iteración
