"""Prueba física del actuador de R3-B en paccaA100 -- no forma parte del
dataset final, ni del pipeline de campañas.

Objetivo (protocolo §18, "pendiente de validación física"):

1. Confirmar que ``HardwareFrequencyActuator`` (el actuador real, no un doble
   de prueba) aplica una frecuencia CPU y que el reloj observado bajo carga
   real se sostiene en el valor pedido.
2. Medir el costo REAL de actuación (tiempo + energía de aplicar y
   restaurar), en vez de asumirlo en cero como hace la evaluación offline de
   R3-A.
3. Confirmar que la restauración deja el nodo en su estado original, incluso
   tras una excepción simulada durante la carga.

No escribe nada en ``hyperion-results``. Imprime un resumen JSON a stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector.agent import (
    CallableDevicePolicy,
    DecisionRequest,
    FrequencyRecommendation,
    HybridAgentPolicy,
    MinimalAgentController,
)
from orchestrator import environment, freqctl
from orchestrator.agent_actuator import HardwareFrequencyActuator
from orchestrator.manifest import FrequencyLevel

DELEGATED_CPUS = (0, 1, 2, 3, 4, 5)
RAPL_ENERGY_PATHS = (
    Path("/sys/class/powercap/intel-rapl:0/energy_uj"),
    Path("/sys/class/powercap/intel-rapl:1/energy_uj"),
)


def read_rapl_joules() -> float | None:
    total_uj = 0
    for path in RAPL_ENERGY_PATHS:
        try:
            total_uj += int(path.read_text().strip())
        except (OSError, ValueError):
            return None
    return total_uj / 1e6


class _FixedFrequencyPolicy:
    """Devuelve exactamente la acción pedida -- este smoke test valida
    actuación física, no la calidad de la decisión (ya evaluada offline)."""

    name = "smoke_fixed"

    def __init__(self, action: str) -> None:
        self._action = action

    def recommend(self, request, *, resource_state, device):
        return FrequencyRecommendation(self._action, False, None, "smoke_test_forced")


def busy_load(cpus: tuple[int, ...], seconds: float) -> list[subprocess.Popen]:
    """Carga real y sostenida por CPU delegado (mismo patrón de warm-up de
    ARC-165: ``taskset -c <cpu> yes`` -- sin ella, un núcleo inactivo nunca
    refleja el candado bajo intel_pstate)."""
    procs = []
    for cpu in cpus:
        procs.append(subprocess.Popen(
            ["taskset", "-c", str(cpu), "timeout", str(seconds), "yes"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    return procs


def run_case(controller: MinimalAgentController, action: str, *, device: str) -> dict:
    request = DecisionRequest(operation="smoke", size=1, horizon_k=1)

    def workload(decision):
        procs = busy_load(DELEGATED_CPUS, seconds=2.0)
        time.sleep(0.6)  # deja pasar la ventana de asentamiento de P-state
        observed_khz = {
            cpu: freqctl.read_observed_frequency_khz(controller.actuator.env, cpu)
            for cpu in DELEGATED_CPUS
        }
        for proc in procs:
            proc.wait(timeout=5)
        return observed_khz

    energy_before = read_rapl_joules()
    started = time.perf_counter_ns()
    record = controller.execute(request, workload)
    elapsed_total_ns = time.perf_counter_ns() - started
    energy_after = read_rapl_joules()

    return {
        "action": action,
        "device": device,
        "actuation_only_elapsed_ns": record.actuation.elapsed_ns,
        "total_elapsed_ns_incl_workload": elapsed_total_ns,
        "actuation_metadata": record.actuation.metadata,
        "observed_freq_khz_during_load": record.workload_result,
        "rapl_energy_j_full_window": (
            energy_after - energy_before
            if energy_before is not None and energy_after is not None else None
        ),
        "ready_device_after": record.ready_device_after,
    }


def main() -> None:
    env = environment.detect_environment("0-5")
    cpu_levels = {
        "REF": FrequencyLevel(id="REF", mode="native_governor", fraction=None),
        "F3": FrequencyLevel(id="F3", mode="fixed", fraction=0.5),
    }
    gpu_levels = {
        "REF": FrequencyLevel(id="REF", mode="native_governor", fraction=None),
        "F3": FrequencyLevel(id="F3", mode="fixed", fraction=0.5),
    }
    actuator = HardwareFrequencyActuator(
        env=env, delegated_cpus=DELEGATED_CPUS,
        cpu_levels=cpu_levels, gpu_levels=gpu_levels, gpu_index=0,
    )
    policy = HybridAgentPolicy(
        CallableDevicePolicy(lambda request, state: "cpu"),
        _FixedFrequencyPolicy("cpu:F3"),
    )
    results: dict = {}
    with MinimalAgentController(policy, actuator) as controller:
        # Caso 1: aplicar F3 (50%) bajo carga real, medir overhead real.
        results["cpu_f3_under_load"] = run_case(controller, "cpu:F3", device="cpu")

        # Caso 2: restaurar explícitamente a REF y confirmar que el reloj
        # observado vuelve a subir bajo la misma carga real.
        controller.policy.frequency_policy._action = "cpu:REF"
        results["cpu_ref_under_load"] = run_case(controller, "cpu:REF", device="cpu")

        # Caso 3: una carga que falla -- confirma que restore() se invoca y
        # que el estado queda limpio (ready_device vuelve a None).
        def failing_workload(decision):
            raise RuntimeError("smoke_test_forced_failure")

        controller.policy.frequency_policy._action = "cpu:F3"
        try:
            controller.execute(DecisionRequest("smoke", 1, 1), failing_workload)
            results["failure_case"] = {"raised": False}
        except Exception as error:  # noqa: BLE001 -- registrar cualquier excepción
            results["failure_case"] = {
                "raised": True, "type": type(error).__name__,
                "ready_device_after_failure": controller.ready_device,
            }
    results["controller_closed_cleanly"] = True
    print(json.dumps(results, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
