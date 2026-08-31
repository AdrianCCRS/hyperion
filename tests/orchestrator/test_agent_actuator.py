from types import SimpleNamespace

import pytest

from orchestrator import agent_actuator
from orchestrator.freqctl import AppliedFrequency
from orchestrator.gpu_freqctl import AppliedGpuFrequency


def _levels():
    return {
        "REF": SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        "F2": SimpleNamespace(id="F2", mode="fixed", fraction=0.6),
    }


def _patch_controls(monkeypatch, *, gpu_apply_fails=False):
    calls = []
    original = SimpleNamespace(strategy="fake", per_cpu={})
    monkeypatch.setattr(agent_actuator.freqctl, "snapshot_original_state", lambda cpus, env: original)
    monkeypatch.setattr(agent_actuator.freqctl, "install_emergency_handlers", lambda restore: calls.append("handlers"))
    monkeypatch.setattr(
        agent_actuator.freqctl, "apply_frequency",
        lambda cpus, level, env, original: AppliedFrequency(
            level_id=level.id, strategy="fake", requested_khz=1, applied_khz=1,
            per_cpu_applied_khz={0: 1}, governor_applied=None, write_skipped_reason=None,
        ),
    )
    monkeypatch.setattr(agent_actuator.freqctl, "settle_if_configured", lambda *args, **kwargs: {0: 1})
    monkeypatch.setattr(agent_actuator.freqctl, "restore_original_state", lambda original, env: calls.append("cpu_restore") or True)

    def gpu_apply(level, env, gpu_index=None):
        if gpu_apply_fails:
            raise RuntimeError("gpu failed")
        return AppliedGpuFrequency(
            level_id=level.id, strategy="fake", requested_mhz=1, applied_mhz=1,
            write_skipped_reason=None, observed_sm_mhz=1,
        )

    monkeypatch.setattr(agent_actuator.gpu_freqctl, "apply_gpu_frequency", gpu_apply)
    monkeypatch.setattr(
        agent_actuator.gpu_freqctl, "restore_gpu_state",
        lambda env, gpu_index=None: calls.append("gpu_restore") or True,
    )
    return calls


def _actuator(monkeypatch, **kwargs):
    calls = _patch_controls(monkeypatch, **kwargs)
    actuator = agent_actuator.HardwareFrequencyActuator(
        env=SimpleNamespace(), delegated_cpus=(0,), cpu_levels=_levels(),
        gpu_levels=_levels(), install_signal_handlers=False,
    )
    return actuator, calls


def test_r3b_actuador_gpu_aplica_host_y_dispositivo(monkeypatch):
    actuator, calls = _actuator(monkeypatch)
    result = actuator.apply("gpu:REF:F2")
    assert result["cpu"]["level_id"] == "REF"
    assert result["gpu"]["level_id"] == "F2"
    assert result["cpu_settled_khz"] == {0: 1}
    assert calls == []


def test_r3b_actuador_cpu_devuelve_gpu_a_ref(monkeypatch):
    actuator, calls = _actuator(monkeypatch)
    result = actuator.apply("cpu:F2")
    assert result["cpu"]["level_id"] == "F2"
    assert result["gpu"]["level_id"] == "REF"
    assert calls == ["gpu_restore"]


def test_r3b_falla_gpu_restaura_ambos_ejes(monkeypatch):
    actuator, calls = _actuator(monkeypatch, gpu_apply_fails=True)
    with pytest.raises(RuntimeError, match="gpu failed"):
        actuator.apply("gpu:REF:F2")
    assert calls == ["cpu_restore", "gpu_restore"]


def test_r3b_restore_combina_cpu_y_gpu(monkeypatch):
    actuator, calls = _actuator(monkeypatch)
    assert actuator.restore()
    assert calls == ["cpu_restore", "gpu_restore"]


def test_r3b_rechaza_accion_mal_formada_y_restaura(monkeypatch):
    actuator, calls = _actuator(monkeypatch)
    with pytest.raises(agent_actuator.AgentActuationError):
        actuator.apply("gpu:F2")
    assert calls == ["cpu_restore", "gpu_restore"]
