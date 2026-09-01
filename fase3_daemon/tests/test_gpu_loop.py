from pathlib import Path
from unittest.mock import patch
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.controller import GpuClockController, GpuClockControllerConfig, GpuPhaseLabel
from fase3_daemon.gpu_loop.loop import (
    GpuFeatures,
    PhaseBeginEvent,
    build_controller_from_policy,
    make_gpu_freqctl_setter,
    query_gpu_features,
    run,
)


def _feat(util=50.0) -> GpuFeatures:
    return GpuFeatures(
        gpu_util_pct=util, gpu_mem_util_pct=10.0, gpu_power_mw=50000.0,
        gpu_sm_clock_mhz=1200.0, gpu_temperature_c=60.0,
    )


def test_run_clasifica_y_aplica_cada_evento():
    applied: list[int] = []
    config = GpuClockControllerConfig(min_dwell_ns=0, compute_bound_clock_mhz=1410, memory_bound_clock_mhz=900)
    controller = GpuClockController(config, lambda mhz: applied.append(mhz) or True)

    events = [
        PhaseBeginEvent(now_ns=0, features=_feat(util=90)),
        PhaseBeginEvent(now_ns=10, features=_feat(util=10)),
    ]

    def classify(features: GpuFeatures) -> GpuPhaseLabel:
        return GpuPhaseLabel.COMPUTE_BOUND if features.gpu_util_pct > 50 else GpuPhaseLabel.MEMORY_BOUND

    decisions = run(events, controller, classify_fn=classify)
    assert len(decisions) == 2
    assert applied == [1410, 900]


def test_run_sin_classify_fn_lanza_notimplemented():
    config = GpuClockControllerConfig(min_dwell_ns=0, compute_bound_clock_mhz=1410, memory_bound_clock_mhz=900)
    controller = GpuClockController(config, lambda mhz: True)
    events = [PhaseBeginEvent(now_ns=0, features=_feat())]
    with pytest.raises(NotImplementedError, match="clasificador de GPU"):
        run(events, controller)  # sin classify_fn -> placeholder que lanza


def test_run_invoca_on_decision_por_cada_evento():
    calls = []
    config = GpuClockControllerConfig(min_dwell_ns=0, compute_bound_clock_mhz=1410, memory_bound_clock_mhz=900)
    controller = GpuClockController(config, lambda mhz: True)
    events = [PhaseBeginEvent(now_ns=0, features=_feat())]

    run(
        events, controller,
        classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
        on_decision=lambda event, label, decision: calls.append((event, label, decision)),
    )
    assert len(calls) == 1
    assert calls[0][1] == GpuPhaseLabel.COMPUTE_BOUND


def test_build_controller_from_policy_no_actuar_da_0mhz_sin_resolver_nivel():
    # "no_actuar" (incluyendo el caso real de hoy, sin T_transición_gpu
    # medido) nunca debe intentar resolver un nivel a MHz -- no hay nivel
    # que resolver. Si esto llamara a _level_to_mhz() por error, lanzaría
    # NotImplementedError y el test fallaría.
    policy = {
        "gpu-compute_bound": {"action": "no_actuar", "chosen_level": None},
        "gpu-memory_bound": {"action": "no_actuar", "chosen_level": None},
    }
    controller = build_controller_from_policy(policy, min_dwell_ns=1_000_000, set_clock=lambda mhz: True)
    decision = controller.on_phase_begin(GpuPhaseLabel.COMPUTE_BOUND, now_ns=0)
    assert decision.target_clock_mhz == 0


def test_build_controller_from_policy_actuar_usa_resolved_clock_mhz():
    policy = {
        "gpu-compute_bound": {"action": "actuar", "chosen_level": "F0", "resolved_clock_mhz": 1410.0},
        "gpu-memory_bound": {"action": "actuar", "chosen_level": "F4", "resolved_clock_mhz": 765.0},
    }
    controller = build_controller_from_policy(policy, min_dwell_ns=1_000_000, set_clock=lambda mhz: True)
    decision = controller.on_phase_begin(GpuPhaseLabel.COMPUTE_BOUND, now_ns=0)
    assert decision.target_clock_mhz == 1410
    decision = controller.on_phase_begin(GpuPhaseLabel.MEMORY_BOUND, now_ns=2_000_000)
    assert decision.target_clock_mhz == 765


def test_build_controller_from_policy_actuar_sin_resolved_clock_mhz_lanza():
    policy = {
        "gpu-compute_bound": {"action": "actuar", "chosen_level": "F0"},  # falta resolved_clock_mhz
        "gpu-memory_bound": {"action": "no_actuar", "chosen_level": None},
    }
    with pytest.raises(ValueError, match="resolved_clock_mhz"):
        build_controller_from_policy(policy, min_dwell_ns=1_000_000, set_clock=lambda mhz: True)


def test_make_gpu_freqctl_setter_construye_nivel_fixed_y_aplica(monkeypatch):
    from common.hpc import gpu_freqctl

    captured = {}

    def fake_apply(level, env, *, gpu_index=None):
        captured["level"] = level
        return SimpleNamespace(applied_mhz=1200, strategy=gpu_freqctl.STRATEGY_LOCKED_CLOCKS)

    monkeypatch.setattr(gpu_freqctl, "apply_gpu_frequency", fake_apply)

    env = SimpleNamespace(gpu_available_clocks_mhz=[800, 1200, 1600])
    setter = make_gpu_freqctl_setter(env)
    assert setter(1200) is True
    assert captured["level"].mode == "fixed"
    assert 0.0 <= captured["level"].fraction <= 1.0


def test_make_gpu_freqctl_setter_sin_relojes_disponibles_falla_sin_llamar_apply():
    env = SimpleNamespace(gpu_available_clocks_mhz=[])
    setter = make_gpu_freqctl_setter(env)
    assert setter(1200) is False


# --- query_gpu_features (movido desde fase3_daemon/shim/event_listener.py,
# eliminado -- la fuente de features NVML sigue siendo necesaria para el
# sondeo de activity_poller.py, ahora vive junto al resto de este módulo) ---

def test_query_gpu_features_parsea_salida_de_nvidia_smi():
    fake_stdout = "42, 10, 55.0, 1200, 65\n"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_stdout
        features = query_gpu_features()
    assert features == GpuFeatures(
        gpu_util_pct=42.0, gpu_mem_util_pct=10.0, gpu_power_mw=55000.0,
        gpu_sm_clock_mhz=1200.0, gpu_temperature_c=65.0,
    )


def test_query_gpu_features_devuelve_none_si_nvidia_smi_falla():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        assert query_gpu_features() is None


def test_query_gpu_features_devuelve_none_si_nvidia_smi_no_existe():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert query_gpu_features() is None
