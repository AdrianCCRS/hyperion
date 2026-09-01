"""Modulo compartido de parseo de CSV de ncu para caracterizacion de
precision GPU (ARC-110).

Reemplaza la logica duplicada de sweep_ncu_launch_count.py,
reparse_ncu_launch_count.py y extend_ncu_lud_convergence.py, que
seleccionaban un unico juego de contadores (FP32 o FP64) segun
entry.gpu_precision -- si un kernel declarado FP32 ejecutaba operaciones
FP64 (o viceversa), esas operaciones quedaban invisibles y los FLOPs
subestimados sin ninguna senal de que la mezcla existia.

Este modulo siempre recolecta AMBAS precisiones simultaneamente y deja la
decision de que hacer con gpu_precision para el llamador -- nunca decide
por si mismo que precision "importa".
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

# ARC-110: nombres confirmados contra `ncu --query-metrics` real en pacca
# (ncu 2026.1.1.0, paccaA100), no adivinados -- ver
# docs/justifications/data/ncu_all_metrics_2026.1.1.0.txt para el listado
# completo verificado.
METRICS_FP32 = (
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
)
METRICS_FP64 = (
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
)
METRIC_DRAM_BYTES = "dram__bytes.sum"

# ARC-110 (paso 3 del pedido del usuario): una FMA cuenta como 2 FLOPs,
# add/mul como 1 -- NO se amplia a division/raiz/trascendentales sin que
# el usuario lo pida explicitamente, aunque el kernel las use y ncu las
# pueda contar (eso se reporta como hallazgo separado, no se agrega aqui
# en silencio).
ALL_METRICS = (METRIC_DRAM_BYTES, *METRICS_FP32, *METRICS_FP64)


@dataclass(frozen=True)
class GpuPrecisionResult:
    """Cantidades y proporciones medidas -- nunca un ridge ni una
    decision de "que precision importa", eso lo decide el llamador con
    el criterio del paso 6 (no asignar ridge unico a un kernel mixto sin
    decision explicita)."""

    flops_fp32: float
    flops_fp64: float
    flops_total: float
    dram_bytes: float
    fraction_fp32: float | None  # None si flops_total == 0 (division indefinida)
    fraction_fp64: float | None
    n_launches: int
    operational_intensity: float | None  # None si dram_bytes <= 0


def _to_num(value: str) -> float:
    """N/A o vacio -> 0.0, nunca una excepcion no controlada (mismo
    criterio que 'bytes_moved_window == 0 -> NaN controlado' en
    postprocess.py, AGENTS.md seccion 4)."""
    text = value.replace(",", "").strip()
    return float(text) if text not in ("", "N/A") else 0.0


def parse_ncu_csv_totals(csv_text: str) -> tuple[dict[str, float], int]:
    """Suma cada metrica sobre TODAS las filas (todos los lanzamientos
    perfilados) del CSV crudo de `ncu --csv`. Devuelve (totales, numero
    de lanzamientos distintos por columna ID)."""
    lines = csv_text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if header_idx is None:
        return {}, 0
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)

    totals: dict[str, float] = {}
    for row in rows:
        name = row["Metric Name"]
        totals[name] = totals.get(name, 0.0) + _to_num(row["Metric Value"])

    n_launches = len({row["ID"] for row in rows})
    return totals, n_launches


def compute_gpu_precision_result(totals: dict[str, float], n_launches: int) -> GpuPrecisionResult:
    """A partir de los totales ya agregados (ver parse_ncu_csv_totals),
    calcula FLOPs por precision por separado. Nunca usa gpu_precision --
    eso se compara DESPUES, por el llamador, contra lo aqui medido."""
    fma32, add32, mul32 = (totals.get(m, 0.0) for m in METRICS_FP32)
    fma64, add64, mul64 = (totals.get(m, 0.0) for m in METRICS_FP64)

    flops_fp32 = 2.0 * fma32 + add32 + mul32
    flops_fp64 = 2.0 * fma64 + add64 + mul64
    flops_total = flops_fp32 + flops_fp64
    dram_bytes = totals.get(METRIC_DRAM_BYTES, 0.0)

    fraction_fp32 = (flops_fp32 / flops_total) if flops_total > 0 else None
    fraction_fp64 = (flops_fp64 / flops_total) if flops_total > 0 else None
    operational_intensity = (flops_total / dram_bytes) if dram_bytes > 0 else None

    return GpuPrecisionResult(
        flops_fp32=flops_fp32,
        flops_fp64=flops_fp64,
        flops_total=flops_total,
        dram_bytes=dram_bytes,
        fraction_fp32=fraction_fp32,
        fraction_fp64=fraction_fp64,
        n_launches=n_launches,
        operational_intensity=operational_intensity,
    )


# ARC-110 (paso 6): un kernel es "mixto" si ambas precisiones representan
# una FRACCION no trivial del total -- el umbral aqui NO decide que es
# "despreciable" metodologicamente, solo evita que ruido de arranque de
# una sola cuenta (ver el caso ya documentado en main.tex para precision
# simple en CPU, ARC-101: "una o dos cuentas triviales... consistentes con
# ruido de arranque") dispare una alerta de mezcla espuria. Cualquier
# kernel que cruce este umbral se reporta al usuario, nunca se resuelve
# solo.
#
# Un piso ABSOLUTO de FLOPs (version anterior de este modulo) genera
# falsos positivos reales: rodinia_heartwall mostro fp64=1.1e6 frente a
# fp32=9.2e10 (fraccion ~0.00001, ruido de arranque genuino) y aun asi
# cruzaba un piso absoluto de 1000 FLOPs. El piso relativo evita esto.
_MIXED_PRECISION_MIN_FRACTION = 0.001  # 0.1% del total -- por debajo de
# esto se trata como ruido de arranque, no como mezcla real; los 3 casos
# reales encontrados en la auditoria (30.4%, 71.3%, 82.3%) estan muy por
# encima de este umbral.


def is_mixed_precision(result: GpuPrecisionResult) -> bool:
    if result.fraction_fp32 is None or result.fraction_fp64 is None:
        return False
    return (
        result.fraction_fp32 >= _MIXED_PRECISION_MIN_FRACTION
        and result.fraction_fp64 >= _MIXED_PRECISION_MIN_FRACTION
    )
