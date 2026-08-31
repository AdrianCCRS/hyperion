"""R3-B: máquina de estados mínima y política híbrida del agente.

Este módulo no escribe frecuencias por sí mismo. Separa deliberadamente la
decisión (probable y comprobable en cualquier host) de la actuación física,
que se inyecta mediante :class:`FrequencyActuator` y reutilizará los
controladores verificados de ``orchestrator.freqctl``/``gpu_freqctl``.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable, Protocol

import pandas as pd

from .compact import RESOURCE_STATES
from .dataset import _static_descriptors
from .dvfs import (
    DVFS_FEATURES,
    CostModels,
    REGION_NOISE_PCT,
    _net_edp,
    _ref_action,
    _size_regime,
    fit_cost_models,
    predict_costs,
)


class AgentContractError(RuntimeError):
    """Una decisión o transición viola el contrato mínimo de R3-B."""


@dataclass(frozen=True)
class DecisionRequest:
    operation: str
    size: int
    horizon_k: int
    # El ancla REF procede de un despacho real. Sin ella, la capa DVFS debe
    # abstenerse: el target relativo de R3-A no puede reconstruir costos.
    ref_energy_j: float | None = None
    ref_time_s: float | None = None


@dataclass(frozen=True)
class FrequencyRecommendation:
    action: str
    abstained: bool
    uncertainty_pct: float | None
    reason: str


@dataclass(frozen=True)
class AgentDecision:
    operation: str
    size: int
    horizon_k: int
    resource_state: str
    device: str
    frequency_action: str
    device_policy: str
    frequency_policy: str
    abstained: bool
    reason: str
    inference_time_ns: int


@dataclass(frozen=True)
class ActuationMeasurement:
    action: str
    elapsed_ns: int
    energy_j: float | None
    metadata: Any = None


@dataclass(frozen=True)
class ExecutionRecord:
    decision: AgentDecision
    actuation: ActuationMeasurement
    workload_result: Any
    ready_device_after: str


class DevicePolicy(Protocol):
    name: str

    def choose_device(self, request: DecisionRequest, resource_state: str) -> str: ...


class FrequencyPolicy(Protocol):
    name: str

    def recommend(
        self, request: DecisionRequest, *, resource_state: str, device: str,
    ) -> FrequencyRecommendation: ...


class FrequencyActuator(Protocol):
    def apply(self, action: str) -> Any: ...
    def restore(self) -> bool: ...


class CallableDevicePolicy:
    """Adaptador para la tabla/umbral de dispositivo congelada fuera de R3."""

    def __init__(self, choose: Callable[[DecisionRequest, str], str], *, name: str = "simple_device_policy"):
        self._choose = choose
        self.name = name

    def choose_device(self, request: DecisionRequest, resource_state: str) -> str:
        return str(self._choose(request, resource_state))


class PowerLawRuntimePolicy:
    """Adaptador desplegable de la familia ``power_law`` de R3-A.

    Se ajusta sobre el dataset DVFS de desarrollo y solo reconstruye costos
    para una petición cuando existe un sondeo REF real de esa petición.
    """

    name = "power_law"

    def __init__(self, models: CostModels, actions: dict[tuple[str, str], tuple[str, ...]]):
        self.models = models
        self.actions = actions

    @classmethod
    def fit(cls, dvfs: pd.DataFrame) -> "PowerLawRuntimePolicy":
        models = fit_cost_models(dvfs, "power_law")
        actions = {
            (str(device), str(region)): tuple(sorted(group["frequency_action"].astype(str).unique()))
            for (device, region), group in dvfs.groupby(["device", "region"], observed=True)
        }
        return cls(models, actions)

    def recommend(
        self, request: DecisionRequest, *, resource_state: str, device: str,
    ) -> FrequencyRecommendation:
        reference = _ref_action(device)
        if request.ref_energy_j is None or request.ref_time_s is None:
            return FrequencyRecommendation(reference, True, None, "missing_ref_probe")
        if request.ref_energy_j <= 0 or request.ref_time_s <= 0:
            raise AgentContractError("el sondeo REF debe tener energía y tiempo positivos")
        if resource_state not in RESOURCE_STATES:
            raise AgentContractError(f"estado desconocido: {resource_state!r}")
        region = dict(zip(("cpu", "gpu"), RESOURCE_STATES[resource_state]))[device]
        actions = self.actions.get((device, region), ())
        if reference not in actions:
            return FrequencyRecommendation(reference, True, None, "no_action_catalog")

        descriptors = _static_descriptors(request.operation, request.size)
        rows = []
        for action in actions:
            rows.append({
                "config_id": f"runtime:{request.operation}_N{request.size}",
                "decision_group_id": f"runtime:{request.operation}_N{request.size}:{resource_state}",
                "operation": request.operation,
                "size": request.size,
                **descriptors,
                "resource_state": resource_state,
                "device": device,
                "region": region,
                "frequency_action": action,
                "operation_frequency_action": f"{request.operation}:{action}",
                "reference_action": reference,
                "ref_energy_j": request.ref_energy_j,
                "ref_time_s": request.ref_time_s,
            })
        predicted = predict_costs(self.models, pd.DataFrame(rows))
        switched = predicted["frequency_action"].astype(str).to_numpy() != reference
        predicted["pred_net_edp_js"] = _net_edp(
            predicted["pred_energy_j"].to_numpy(float),
            predicted["pred_time_s"].to_numpy(float), switched, 0.0, 0.0,
        )
        best = float(predicted["pred_net_edp_js"].min())
        regime = _size_regime(request.operation, request.size, self.models.size_thresholds)
        context = (resource_state, device, regime)
        model_error = self.models.uncertainty_pct_by_context.get(
            context, self.models.uncertainty_pct,
        )
        uncertainty = max(float(REGION_NOISE_PCT[region]), float(model_error))
        equivalent = predicted[
            predicted["pred_net_edp_js"] <= best * (1.0 + uncertainty / 100.0)
        ]
        if reference in set(equivalent["frequency_action"].astype(str)):
            return FrequencyRecommendation(reference, True, uncertainty, "ref_inside_equivalent_set")
        action = str(predicted.loc[predicted["pred_net_edp_js"].idxmin(), "frequency_action"])
        return FrequencyRecommendation(action, False, uncertainty, "model_advantage_above_gate")


class HybridAgentPolicy:
    """Política acordada: dispositivo simple; ML solo en ``gpu_ready``/GPU."""

    def __init__(self, device_policy: DevicePolicy, frequency_policy: FrequencyPolicy):
        self.device_policy = device_policy
        self.frequency_policy = frequency_policy

    def decide(self, request: DecisionRequest, *, ready_device: str | None) -> AgentDecision:
        if request.size <= 0 or request.horizon_k <= 0:
            raise AgentContractError("size y horizon_k deben ser positivos")
        state = "none_ready" if ready_device is None else f"{ready_device}_ready"
        if state not in RESOURCE_STATES:
            raise AgentContractError(f"ready_device inválido: {ready_device!r}")
        started = perf_counter_ns()
        device = self.device_policy.choose_device(request, state)
        if device not in ("cpu", "gpu"):
            raise AgentContractError(f"dispositivo inválido: {device!r}")

        if state == "gpu_ready" and device == "gpu":
            try:
                recommendation = self.frequency_policy.recommend(
                    request, resource_state=state, device=device,
                )
            except Exception as error:  # fallback seguro y trazable
                recommendation = FrequencyRecommendation(
                    _ref_action(device), True, None,
                    f"model_error:{type(error).__name__}",
                )
        else:
            recommendation = FrequencyRecommendation(
                _ref_action(device), True, None, "ml_gate_closed",
            )
        expected_prefix = f"{device}:"
        if not recommendation.action.startswith(expected_prefix):
            raise AgentContractError(
                f"acción {recommendation.action!r} incompatible con {device!r}",
            )
        elapsed = perf_counter_ns() - started
        return AgentDecision(
            operation=request.operation, size=request.size, horizon_k=request.horizon_k,
            resource_state=state, device=device,
            frequency_action=recommendation.action,
            device_policy=self.device_policy.name,
            frequency_policy=self.frequency_policy.name,
            abstained=recommendation.abstained, reason=recommendation.reason,
            inference_time_ns=elapsed,
        )


class MinimalAgentController:
    """Ejecuta decisiones, mide actuación y conserva el estado confirmado."""

    def __init__(
        self, policy: HybridAgentPolicy, actuator: FrequencyActuator, *,
        energy_reader: Callable[[], float] | None = None,
    ) -> None:
        self.policy = policy
        self.actuator = actuator
        self.energy_reader = energy_reader
        self.ready_device: str | None = None
        self._closed = False

    def execute(self, request: DecisionRequest, workload: Callable[[AgentDecision], Any]) -> ExecutionRecord:
        if self._closed:
            raise AgentContractError("el controlador ya fue cerrado")
        decision = self.policy.decide(request, ready_device=self.ready_device)
        energy_before = self.energy_reader() if self.energy_reader else None
        started = perf_counter_ns()
        try:
            metadata = self.actuator.apply(decision.frequency_action)
            elapsed = perf_counter_ns() - started
            energy_after = self.energy_reader() if self.energy_reader else None
            actuation = ActuationMeasurement(
                action=decision.frequency_action, elapsed_ns=elapsed,
                energy_j=(energy_after - energy_before)
                if energy_before is not None and energy_after is not None else None,
                metadata=metadata,
            )
            result = workload(decision)
        except BaseException as error:
            # Una actuación o carga fallida no confirma la transición; se
            # restaura antes de propagar para no dejar el nodo modificado.
            restored = self.actuator.restore()
            self.ready_device = None
            if not restored:
                raise AgentContractError(
                    "falló la restauración después de una ejecución incompleta",
                ) from error
            raise
        self.ready_device = decision.device
        return ExecutionRecord(decision, actuation, result, self.ready_device)

    def close(self) -> bool:
        if self._closed:
            return True
        restored = bool(self.actuator.restore())
        if restored:
            self.ready_device = None
            self._closed = True
        return restored

    def __enter__(self) -> "MinimalAgentController":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.close():
            raise AgentContractError("falló la restauración al cerrar el controlador")
