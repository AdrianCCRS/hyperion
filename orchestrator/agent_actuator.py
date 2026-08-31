"""Adaptador seguro de actuación CPU/GPU para el agente mínimo R3-B."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from . import freqctl, gpu_freqctl


class AgentActuationError(RuntimeError):
    """Una acción conjunta es inválida o no pudo aplicarse/restaurarse."""


class HardwareFrequencyActuator:
    """Aplica acciones ``cpu:NIVEL`` o ``gpu:HOST:GPU`` y restaura ambas.

    El snapshot CPU se toma una sola vez y solo sobre los CPU delegados (más
    sus hermanos SMT declarados por el entorno, contrato de ``freqctl``).
    Nunca descubre ni escribe rutas por cuenta propia.
    """

    def __init__(
        self, *, env: Any, delegated_cpus: tuple[int, ...],
        cpu_levels: Mapping[str, Any], gpu_levels: Mapping[str, Any],
        frequency_settle: Mapping[str, Any] | None = None,
        gpu_index: int | str | None = None, install_signal_handlers: bool = True,
    ) -> None:
        if not delegated_cpus:
            raise AgentActuationError("se requiere al menos un CPU delegado")
        self.env = env
        self.delegated_cpus = tuple(delegated_cpus)
        self.cpu_levels = dict(cpu_levels)
        self.gpu_levels = dict(gpu_levels)
        self.frequency_settle = frequency_settle or {}
        self.gpu_index = gpu_index
        self.original_cpu = freqctl.snapshot_original_state(self.delegated_cpus, env)
        if install_signal_handlers:
            freqctl.install_emergency_handlers(self.restore)

    def _cpu(self, level_id: str):
        try:
            level = self.cpu_levels[level_id]
        except KeyError as error:
            raise AgentActuationError(f"nivel CPU desconocido: {level_id!r}") from error
        applied = freqctl.apply_frequency(
            self.delegated_cpus, level, self.env, original=self.original_cpu,
        )
        settled = freqctl.settle_if_configured(
            self.delegated_cpus, applied, self.env,
            settle_config=self.frequency_settle,
        )
        return applied, settled

    def _gpu(self, level_id: str):
        try:
            level = self.gpu_levels[level_id]
        except KeyError as error:
            raise AgentActuationError(f"nivel GPU desconocido: {level_id!r}") from error
        return gpu_freqctl.apply_gpu_frequency(
            level, self.env, gpu_index=self.gpu_index,
        )

    def apply(self, action: str) -> dict[str, Any]:
        parts = str(action).split(":")
        try:
            if len(parts) == 2 and parts[0] == "cpu":
                cpu_applied, settled = self._cpu(parts[1])
                if not gpu_freqctl.restore_gpu_state(self.env, gpu_index=self.gpu_index):
                    raise AgentActuationError("no se pudo devolver la GPU a REF")
                return {
                    "cpu": asdict(cpu_applied), "gpu": {"level_id": "REF"},
                    "cpu_settled_khz": settled,
                }
            if len(parts) == 3 and parts[0] == "gpu":
                cpu_applied, settled = self._cpu(parts[1])
                gpu_applied = self._gpu(parts[2])
                return {
                    "cpu": asdict(cpu_applied), "gpu": asdict(gpu_applied),
                    "cpu_settled_khz": settled,
                }
            raise AgentActuationError(f"acción de frecuencia inválida: {action!r}")
        except BaseException:
            # Puede haber una aplicación parcial (CPU aplicada y GPU fallida).
            # Restaurar ambos ejes antes de propagar es obligatorio.
            self.restore()
            raise

    def restore(self) -> bool:
        cpu_ok = freqctl.restore_original_state(self.original_cpu, self.env)
        gpu_ok = gpu_freqctl.restore_gpu_state(self.env, gpu_index=self.gpu_index)
        return bool(cpu_ok and gpu_ok)
