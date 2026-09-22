"""Auditoría fail-closed de cobertura analítica para campañas GPU con CUPTI.

No sustituye ni consulta los valores históricos de ncu. Su único propósito es
impedir que una campaña nueva produzca un dataset parcialmente etiquetado.
"""
from __future__ import annotations

from typing import Any, Mapping

from fase1_telemetria import gpu_oi_models


def audit(manifest: Any, catalog: Mapping[str, Any]) -> list[str]:
    trace = dict(getattr(manifest, "gpu", {}).get("activity_trace", {}) or {})
    if not trace.get("enabled", False):
        return []
    parameters_by_kernel = trace.get("parameters_by_kernel")
    errors: list[str] = []
    if not isinstance(parameters_by_kernel, Mapping):
        return ["gpu.activity_trace.parameters_by_kernel debe ser un mapa"]
    registered = set(gpu_oi_models.registered_kernel_refs())
    for kernel_ref in getattr(manifest, "kernels", ()):
        entry = catalog[kernel_ref]
        if getattr(entry, "device", "cpu") != "gpu":
            continue
        if kernel_ref not in registered:
            errors.append(f"{kernel_ref}: sin modelo analítico registrado")
        elif not isinstance(parameters_by_kernel.get(kernel_ref), Mapping):
            errors.append(f"{kernel_ref}: sin parámetros analíticos explícitos")
        if not getattr(entry, "cupti_activity_exec_path", None):
            errors.append(f"{kernel_ref}: sin ejecutable CUDA directo para CUPTI Activity")
        if not getattr(entry, "cupti_activity_binary_checksum", None):
            errors.append(f"{kernel_ref}: sin checksum del ejecutable CUDA directo")
    return errors


def require_complete(manifest: Any, catalog: Mapping[str, Any]) -> None:
    errors = audit(manifest, catalog)
    if errors:
        raise ValueError(
            "Cobertura OI analítica incompleta; no se inicia la campaña CUPTI:\n- "
            + "\n- ".join(errors)
        )
