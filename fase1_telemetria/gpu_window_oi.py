"""Verdad Roofline GPU por ventana a partir de actividad CUDA analítica.

CUPTI Activity aporta únicamente ``start_ns``/``end_ns`` y la identidad del
lanzamiento. Un registro analítico separado aporta FLOPs y bytes del
lanzamiento completo. Este módulo une ambas piezas sin depender de CUDA y es
por tanto comprobable localmente.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable

from fase1_telemetria import gpu_oi_models


@dataclass(frozen=True)
class LaunchWork:
    """Trabajo analítico de un evento CUDA observado por CUPTI."""

    kernel_name: str
    start_ns: int
    end_ns: int
    flops: float
    bytes_moved: float
    launch_index: int | None = None
    model_id: str | None = None
    activity_kind: str = "kernel"

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("un lanzamiento requiere 0 <= start_ns < end_ns")
        if not math.isfinite(self.flops) or self.flops < 0:
            raise ValueError("flops debe ser finito y no negativo")
        if not math.isfinite(self.bytes_moved) or self.bytes_moved <= 0:
            raise ValueError("bytes_moved debe ser finito y positivo")


@dataclass(frozen=True)
class WindowTruth:
    """Trabajo agregado y etiqueta física candidata de una ventana."""

    start_ns: int
    end_ns: int
    flops: float
    bytes_moved: float
    operational_intensity: float | None
    active_ns: int
    active_fraction: float
    overlapping_launches: int


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def aggregate_launches_in_window(
    launches: Iterable[LaunchWork],
    *,
    start_ns: int,
    end_ns: int,
) -> WindowTruth:
    """Prorratea lanzamientos por solape temporal y suma FLOPs/bytes.

    El prorrateo supone una tasa uniforme de trabajo dentro de cada
    lanzamiento. Es una aproximación explícita y auditable; evita asignar el
    trabajo completo a dos ventanas cuando un lanzamiento cruza el límite.
    ``active_ns`` usa la unión de intervalos y por eso no supera la duración
    de la ventana aunque haya kernels concurrentes.
    """
    if start_ns < 0 or end_ns <= start_ns:
        raise ValueError("la ventana requiere 0 <= start_ns < end_ns")

    flops = 0.0
    bytes_moved = 0.0
    overlaps: list[tuple[int, int]] = []
    count = 0
    for launch in launches:
        overlap_start = max(start_ns, launch.start_ns)
        overlap_end = min(end_ns, launch.end_ns)
        if overlap_end <= overlap_start:
            continue
        overlap_ns = overlap_end - overlap_start
        fraction = overlap_ns / (launch.end_ns - launch.start_ns)
        flops += launch.flops * fraction
        bytes_moved += launch.bytes_moved * fraction
        overlaps.append((overlap_start, overlap_end))
        count += 1

    active_ns = _union_duration(overlaps)
    duration_ns = end_ns - start_ns
    oi = flops / bytes_moved if bytes_moved > 0 else None
    return WindowTruth(
        start_ns=start_ns,
        end_ns=end_ns,
        flops=flops,
        bytes_moved=bytes_moved,
        operational_intensity=oi,
        active_ns=active_ns,
        active_fraction=active_ns / duration_ns,
        overlapping_launches=count,
    )


def build_window_truth(
    launches: Iterable[LaunchWork],
    *,
    start_ns: int,
    end_ns: int,
    window_ns: int = 120_000_000,
) -> list[WindowTruth]:
    """Divide un intervalo en ventanas contiguas y agrega su trabajo CUDA."""
    if window_ns <= 0:
        raise ValueError("window_ns debe ser positivo")
    if start_ns < 0 or end_ns <= start_ns:
        raise ValueError("el intervalo requiere 0 <= start_ns < end_ns")
    launch_list = list(launches)
    out: list[WindowTruth] = []
    cursor = start_ns
    while cursor < end_ns:
        boundary = min(cursor + window_ns, end_ns)
        out.append(
            aggregate_launches_in_window(
                launch_list,
                start_ns=cursor,
                end_ns=boundary,
            )
        )
        cursor = boundary
    return out


def load_cupti_launch_work(
    trace_path: str | Path,
    *,
    kernel_ref: str,
    parameters: dict[str, int | float | str],
    measured_start_ns: int | None = None,
    measured_end_ns: int | None = None,
) -> list[LaunchWork]:
    """Carga un CSV del preload CUPTI y le adjunta trabajo analítico.

    El orden lógico es el ``launch_index`` emitido después de ordenar por
    timestamp. Si el registro no cubre algún nombre o índice, el modelo
    analítico falla cerrado y el archivo no produce etiquetas parciales.
    """
    if (measured_start_ns is None) != (measured_end_ns is None):
        raise ValueError("la región medida requiere inicio y fin juntos")
    if measured_start_ns is not None and measured_end_ns <= measured_start_ns:
        raise ValueError("la región medida requiere start < end")

    path = Path(trace_path)
    raw_events: list[tuple[int, str, str, int, int, int]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"launch_index", "kernel_name", "start_ns", "end_ns"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"traza CUPTI sin columnas requeridas: {sorted(missing)}")
        for row in reader:
            raw_events.append((
                int(row["launch_index"]),
                row.get("activity_kind") or "kernel",
                row["kernel_name"],
                int(row["start_ns"]),
                int(row["end_ns"]),
                int(row.get("transfer_bytes") or 0),
            ))

    if measured_start_ns is not None:
        raw_events = [
            event for event in raw_events
            if event[3] >= measured_start_ns and event[4] <= measured_end_ns
        ]

    spec = gpu_oi_models.operation_spec_for_kernel(kernel_ref)
    launches: list[LaunchWork] = []
    if spec is not None:
        if spec.allow_unmatched_prefix:
            candidates: list[list[tuple[int, str, str, int, int, int]]] = []
            for offset in range(spec.events_per_operation):
                candidate = raw_events[offset:]
                if len(candidate) < spec.events_per_operation:
                    continue
                if len(candidate) % spec.events_per_operation:
                    continue
                first = candidate[:spec.events_per_operation]
                try:
                    gpu_oi_models.work_for_operation(
                        kernel_ref,
                        [(kind, name, transfer_bytes) for _, kind, name, _, _, transfer_bytes in first],
                        0, parameters,
                    )
                except ValueError:
                    continue
                candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(
                    f"no se pudo sincronizar de forma única el bucle de operaciones de {kernel_ref}"
                )
            raw_events = candidates[0]
        if len(raw_events) % spec.events_per_operation != 0:
            raise ValueError(
                f"traza CUPTI incompleta para {kernel_ref}: {len(raw_events)} eventos no es "
                f"múltiplo de {spec.events_per_operation}"
            )
        for operation_index, offset in enumerate(range(0, len(raw_events), spec.events_per_operation)):
            group = raw_events[offset:offset + spec.events_per_operation]
            work = gpu_oi_models.work_for_operation(
                kernel_ref,
                [(kind, name, transfer_bytes) for _, kind, name, _, _, transfer_bytes in group],
                operation_index,
                parameters,
            )
            total_duration = sum(end_ns - start_ns for _, _, _, start_ns, end_ns, _ in group)
            if total_duration <= 0:
                raise ValueError(f"operación CUPTI sin duración positiva para {kernel_ref}")
            for launch_index, kind, name, start_ns, end_ns, _ in group:
                fraction = (end_ns - start_ns) / total_duration
                launches.append(LaunchWork(
                    kernel_name=name, start_ns=start_ns, end_ns=end_ns,
                    flops=work.flops * fraction, bytes_moved=work.bytes_moved * fraction,
                    launch_index=launch_index, model_id=work.model_id, activity_kind=kind,
                ))
        return launches

    logical_kernel_index = 0
    for launch_index, kind, kernel_name, start_ns, end_ns, _ in raw_events:
        # Los modelos de lanzamiento único inicializan los datos fuera del
        # bucle medido. Las copias sin un modelo de operación explícito no se
        # etiquetan por inferencia: se omiten para no fabricarles FLOPs/bytes.
        if kind != "kernel":
            continue
        work = gpu_oi_models.work_for_launch(
            kernel_ref,
            kernel_name,
            logical_kernel_index,
            parameters,
        )
        launches.append(
            LaunchWork(
                kernel_name=kernel_name,
                start_ns=start_ns,
                end_ns=end_ns,
                flops=work.flops,
                bytes_moved=work.bytes_moved,
                launch_index=launch_index,
                model_id=work.model_id,
                activity_kind=kind,
            )
        )
        logical_kernel_index += 1
    return launches
