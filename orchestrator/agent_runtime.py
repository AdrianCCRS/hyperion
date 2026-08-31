"""Ejecución catalogada y registro durable del agente mínimo R3-B.

Este módulo no construye comandos de carga. Resuelve exclusivamente entradas
ya declaradas en el catálogo y delega la ejecución al runner inyectado.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from classifier.selector.agent import (
    AgentDecision,
    DecisionRequest,
    ExecutionRecord,
    MinimalAgentController,
)

from .catalog import KernelEntry, verify_binary


class AgentRuntimeError(RuntimeError):
    """La petición no puede resolverse o ejecutarse bajo el contrato R3-B."""


@dataclass(frozen=True)
class CatalogWorkloadResult:
    kernel_ref: str
    config_id: str
    run_result: Any


def resolve_agent_kernel(
    catalog: Mapping[str, KernelEntry], *, config_id: str, device: str,
    node_id: str | None, verifier: Callable[[KernelEntry, str | None], bool] = verify_binary,
) -> tuple[str, KernelEntry]:
    """Resuelve exactamente un kernel dataset y verifica su binario.

    ``config_id`` es la clave explícita que empareja CPU/GPU en el catálogo;
    no se fabrica un id de kernel a partir del nombre de la operación.
    """
    if device not in ("cpu", "gpu"):
        raise AgentRuntimeError(f"dispositivo inválido: {device!r}")
    matches = [
        (kernel_ref, entry)
        for kernel_ref, entry in catalog.items()
        if entry.role == "dataset"
        and entry.config_id == config_id
        and entry.device == device
    ]
    if len(matches) != 1:
        raise AgentRuntimeError(
            f"se esperaba un kernel dataset para config_id={config_id!r}, "
            f"device={device!r}; encontrados {len(matches)}",
        )
    kernel_ref, entry = matches[0]
    if not verifier(entry, node_id):
        raise AgentRuntimeError(
            f"C02: checksum de {kernel_ref!r} no coincide antes de ejecutar el agente",
        )
    return kernel_ref, entry


class AgentAuditLog:
    """Registro JSONL append-only; cada evento se vacía y sincroniza a disco."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: Mapping[str, Any]) -> None:
        event = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload),
        }
        encoded = (
            json.dumps(event, allow_nan=False, default=str, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise AgentRuntimeError("registro de auditoría escrito parcialmente")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class CatalogAgentRuntime:
    """Une política, actuación y un runner sin aceptar comandos arbitrarios."""

    def __init__(
        self, controller: MinimalAgentController, catalog: Mapping[str, KernelEntry], *,
        node_id: str | None,
        executor: Callable[[str, KernelEntry, AgentDecision], Any],
        audit_log: AgentAuditLog,
        verifier: Callable[[KernelEntry, str | None], bool] = verify_binary,
    ) -> None:
        self.controller = controller
        self.catalog = catalog
        self.node_id = node_id
        self.executor = executor
        self.audit_log = audit_log
        self.verifier = verifier

    @staticmethod
    def _run_summary(result: Any) -> dict[str, Any]:
        return {
            name: getattr(result, name, None)
            for name in ("run_id", "success", "elapsed_seconds", "run_dir")
        }

    def execute(self, request: DecisionRequest, *, config_id: str) -> ExecutionRecord:
        ready_before = self.controller.ready_device

        def workload(decision: AgentDecision) -> CatalogWorkloadResult:
            kernel_ref, entry = resolve_agent_kernel(
                self.catalog,
                config_id=config_id,
                device=decision.device,
                node_id=self.node_id,
                verifier=self.verifier,
            )
            result = self.executor(kernel_ref, entry, decision)
            if getattr(result, "success", True) is not True:
                raise AgentRuntimeError(f"la carga catalogada {kernel_ref!r} falló")
            return CatalogWorkloadResult(kernel_ref, config_id, result)

        try:
            record = self.controller.execute(request, workload)
        except BaseException as error:
            self.audit_log.append({
                "status": "failed",
                "config_id": config_id,
                "request": asdict(request),
                "ready_device_before": ready_before,
                "ready_device_after": self.controller.ready_device,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            raise

        workload_result = record.workload_result
        self.audit_log.append({
            "status": "completed",
            "config_id": config_id,
            "kernel_ref": workload_result.kernel_ref,
            "request": asdict(request),
            "decision": asdict(record.decision),
            "actuation": asdict(record.actuation),
            "ready_device_before": ready_before,
            "ready_device_after": record.ready_device_after,
            "run": self._run_summary(workload_result.run_result),
        })
        return record

