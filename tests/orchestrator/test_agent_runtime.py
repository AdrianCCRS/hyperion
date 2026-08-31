import json
from types import SimpleNamespace

import pytest

from classifier.selector.agent import (
    CallableDevicePolicy,
    DecisionRequest,
    FrequencyRecommendation,
    HybridAgentPolicy,
    MinimalAgentController,
)
from orchestrator.agent_runtime import (
    AgentAuditLog,
    AgentRuntimeError,
    CatalogAgentRuntime,
    resolve_agent_kernel,
)


class RefFrequencyPolicy:
    name = "ref"

    def recommend(self, request, *, resource_state, device):
        action = "cpu:REF" if device == "cpu" else "gpu:REF:REF"
        return FrequencyRecommendation(action, True, None, "test")


class FakeActuator:
    def __init__(self):
        self.restores = 0

    def apply(self, action):
        return {"applied": action}

    def restore(self):
        self.restores += 1
        return True


def _entry(device="cpu", config_id="gemm_N128"):
    return SimpleNamespace(role="dataset", config_id=config_id, device=device)


def _controller(device="cpu"):
    policy = HybridAgentPolicy(
        CallableDevicePolicy(lambda request, state: device),
        RefFrequencyPolicy(),
    )
    return MinimalAgentController(policy, FakeActuator())


def test_r3b_resuelve_config_id_explicito_y_dispositivo():
    catalog = {"cpu_ref": _entry("cpu"), "gpu_ref": _entry("gpu")}
    kernel_ref, entry = resolve_agent_kernel(
        catalog, config_id="gemm_N128", device="gpu", node_id="pacca-a100",
        verifier=lambda candidate, node: candidate is catalog["gpu_ref"] and node == "pacca-a100",
    )
    assert kernel_ref == "gpu_ref"
    assert entry is catalog["gpu_ref"]


def test_r3b_rechaza_configuracion_ambigua():
    entry = _entry()
    with pytest.raises(AgentRuntimeError, match="encontrados 2"):
        resolve_agent_kernel(
            {"a": entry, "b": entry}, config_id="gemm_N128", device="cpu",
            node_id=None, verifier=lambda candidate, node: True,
        )


def test_r3b_rechaza_checksum_incorrecto():
    with pytest.raises(AgentRuntimeError, match="C02"):
        resolve_agent_kernel(
            {"cpu": _entry()}, config_id="gemm_N128", device="cpu",
            node_id="pacca-a100", verifier=lambda candidate, node: False,
        )


def test_r3b_runtime_ejecuta_catalogo_y_registra(tmp_path):
    calls = []
    audit = AgentAuditLog(tmp_path / "agent.jsonl")
    runtime = CatalogAgentRuntime(
        _controller("gpu"), {"gpu_ref": _entry("gpu")}, node_id="pacca-a100",
        executor=lambda ref, entry, decision: calls.append((ref, decision.device)) or SimpleNamespace(
            run_id="run-1", success=True, elapsed_seconds=0.1, run_dir=tmp_path / "run-1",
        ),
        audit_log=audit, verifier=lambda candidate, node: True,
    )
    record = runtime.execute(DecisionRequest("gemm", 128, 1), config_id="gemm_N128")
    assert calls == [("gpu_ref", "gpu")]
    assert record.workload_result.kernel_ref == "gpu_ref"
    event = json.loads(audit.path.read_text(encoding="utf-8"))
    assert event["status"] == "completed"
    assert event["kernel_ref"] == "gpu_ref"
    assert event["decision"]["frequency_action"] == "gpu:REF:REF"


def test_r3b_runtime_registra_fallo_y_restaura(tmp_path):
    controller = _controller("cpu")
    audit = AgentAuditLog(tmp_path / "agent.jsonl")
    runtime = CatalogAgentRuntime(
        controller, {"cpu_ref": _entry("cpu")}, node_id=None,
        executor=lambda ref, entry, decision: SimpleNamespace(success=False),
        audit_log=audit, verifier=lambda candidate, node: True,
    )
    with pytest.raises(AgentRuntimeError, match="falló"):
        runtime.execute(DecisionRequest("gemm", 128, 1), config_id="gemm_N128")
    event = json.loads(audit.path.read_text(encoding="utf-8"))
    assert event["status"] == "failed"
    assert event["ready_device_after"] is None
    assert controller.actuator.restores == 1
