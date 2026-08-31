import math
import pandas as pd
import pytest

from classifier.selector.agent import (
    AgentContractError,
    CallableDevicePolicy,
    DecisionRequest,
    FrequencyRecommendation,
    HybridAgentPolicy,
    MinimalAgentController,
    PowerLawRuntimePolicy,
)
from classifier.selector.dataset import _static_descriptors


class FakeFrequencyPolicy:
    name = "power_law"

    def __init__(self, recommendation=None, error=None):
        self.calls = []
        self.recommendation = recommendation or FrequencyRecommendation(
            "gpu:REF:F2", False, 4.0, "model_advantage_above_gate",
        )
        self.error = error

    def recommend(self, request, *, resource_state, device):
        self.calls.append((request, resource_state, device))
        if self.error:
            raise self.error
        return self.recommendation


class FakeActuator:
    def __init__(self, fail=False, restore_ok=True):
        self.actions = []
        self.restores = 0
        self.fail = fail
        self.restore_ok = restore_ok

    def apply(self, action):
        self.actions.append(action)
        if self.fail:
            raise RuntimeError("actuation failed")
        return {"applied": action}

    def restore(self):
        self.restores += 1
        return self.restore_ok


def _request(**kwargs):
    values = dict(operation="gemm", size=4096, horizon_k=10, ref_energy_j=2.0, ref_time_s=3.0)
    values.update(kwargs)
    return DecisionRequest(**values)


def _agent(device="gpu", frequency=None):
    device_policy = CallableDevicePolicy(lambda request, state: device)
    return HybridAgentPolicy(device_policy, frequency or FakeFrequencyPolicy())


def test_r3b_ml_permanece_cerrado_fuera_de_gpu_ready():
    frequency = FakeFrequencyPolicy()
    policy = _agent(frequency=frequency)
    decision = policy.decide(_request(), ready_device=None)
    assert decision.resource_state == "none_ready"
    assert decision.frequency_action == "gpu:REF:REF"
    assert decision.reason == "ml_gate_closed"
    assert frequency.calls == []


def test_r3b_ml_actua_unicamente_si_gpu_ya_esta_preparada_y_sigue_en_gpu():
    frequency = FakeFrequencyPolicy()
    decision = _agent(frequency=frequency).decide(_request(), ready_device="gpu")
    assert decision.frequency_action == "gpu:REF:F2"
    assert not decision.abstained
    assert len(frequency.calls) == 1


def test_r3b_migrar_desde_gpu_a_cpu_cierra_compuerta_ml():
    frequency = FakeFrequencyPolicy()
    decision = _agent(device="cpu", frequency=frequency).decide(_request(), ready_device="gpu")
    assert decision.frequency_action == "cpu:REF"
    assert frequency.calls == []


def test_r3b_error_del_model_cae_a_ref_y_queda_trazado():
    decision = _agent(frequency=FakeFrequencyPolicy(error=ValueError("bad model"))).decide(
        _request(), ready_device="gpu",
    )
    assert decision.frequency_action == "gpu:REF:REF"
    assert decision.abstained
    assert decision.reason == "model_error:ValueError"


def test_r3b_controlador_actualiza_estado_solo_despues_de_carga_exitosa():
    actuator = FakeActuator()
    controller = MinimalAgentController(_agent(), actuator)
    first = controller.execute(_request(), lambda decision: "ok")
    assert first.decision.resource_state == "none_ready"
    assert controller.ready_device == "gpu"
    second = controller.execute(_request(), lambda decision: "ok2")
    assert second.decision.resource_state == "gpu_ready"
    assert second.decision.frequency_action == "gpu:REF:F2"
    assert controller.close()
    assert actuator.restores == 1
    assert controller.ready_device is None


def test_r3b_falla_de_actuacion_restaura_y_no_confirma_estado():
    actuator = FakeActuator(fail=True)
    controller = MinimalAgentController(_agent(), actuator)
    with pytest.raises(RuntimeError, match="actuation failed"):
        controller.execute(_request(), lambda decision: "never")
    assert actuator.restores == 1
    assert controller.ready_device is None


def test_r3b_falla_de_restauracion_nunca_se_silencia():
    actuator = FakeActuator(fail=True, restore_ok=False)
    controller = MinimalAgentController(_agent(), actuator)
    with pytest.raises(AgentContractError, match="falló la restauración"):
        controller.execute(_request(), lambda decision: "never")
    assert controller.ready_device is None


def test_r3b_contexto_falla_ruidosamente_si_no_puede_restaurar():
    controller = MinimalAgentController(_agent(), FakeActuator(restore_ok=False))
    with pytest.raises(AgentContractError, match="falló la restauración"):
        with controller:
            pass


def test_r3b_mide_energia_de_actuacion_separada_de_la_carga():
    readings = iter((10.0, 10.25))
    controller = MinimalAgentController(
        _agent(), FakeActuator(), energy_reader=lambda: next(readings),
    )
    record = controller.execute(_request(), lambda decision: "ok")
    assert record.actuation.energy_j == pytest.approx(0.25)
    assert record.actuation.elapsed_ns >= 0
    assert record.decision.inference_time_ns >= 0


def test_r3b_rechaza_peticion_sin_horizonte_positivo():
    with pytest.raises(AgentContractError):
        _agent().decide(_request(horizon_k=0), ready_device=None)


def _power_law_training_frame():
    rows = []
    for size in (128, 256, 512, 1024):
        descriptors = _static_descriptors("gemm", size)
        ref_energy, ref_time = float(size), float(size) / 2.0
        for action, energy_ratio, time_ratio in (
            ("gpu:REF:REF", 1.0, 1.0),
            ("gpu:REF:F2", 0.8, 0.8),
        ):
            rows.append({
                "config_id": f"gemm_N{size}",
                "decision_group_id": f"gemm_N{size}:gpu_ready",
                "operation": "gemm", "size": size, **descriptors,
                "resource_state": "gpu_ready", "device": "gpu", "region": "warm",
                "frequency_action": action,
                "operation_frequency_action": f"gemm:{action}",
                "reference_action": "gpu:REF:REF",
                "energy_j": ref_energy * energy_ratio,
                "time_s": ref_time * time_ratio,
                "edp_js": ref_energy * energy_ratio * ref_time * time_ratio,
                "ref_energy_j": ref_energy, "ref_time_s": ref_time,
                "log_energy_ratio": math.log(energy_ratio),
                "log_time_ratio": math.log(time_ratio),
            })
    return pd.DataFrame(rows)


def test_r3b_power_law_runtime_usa_sondeo_ref_y_recomienda_accion():
    policy = PowerLawRuntimePolicy.fit(_power_law_training_frame())
    recommendation = policy.recommend(
        _request(size=2048), resource_state="gpu_ready", device="gpu",
    )
    assert recommendation.action == "gpu:REF:F2"
    assert not recommendation.abstained


def test_r3b_power_law_runtime_sin_sondeo_ref_se_abstiene():
    policy = PowerLawRuntimePolicy.fit(_power_law_training_frame())
    recommendation = policy.recommend(
        _request(ref_energy_j=None, ref_time_s=None),
        resource_state="gpu_ready", device="gpu",
    )
    assert recommendation.action == "gpu:REF:REF"
    assert recommendation.abstained
    assert recommendation.reason == "missing_ref_probe"
