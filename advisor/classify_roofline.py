"""Clasificacion Roofline de validacion (compute_bound / memory_bound /
ambiguous) a partir de reportes REALES de Advisor -- nunca un umbral
arbitrario tipo "AI < 10". Cada constante de este modulo esta declarada,
nombrada, y justificada en su comentario; ver
docs/advisor/pipeline_advisor_diseno_y_arquitectura.md seccion 2 para la
justificacion completa de cada una.

No se mezcla con phase_label_train de Hyperion ni con
vtune_validation_class de la campana de VTune -- columna propia
(advisor_roofline_class), i_ridge propio (i_ridge_advisor), nunca
promediados ni sustituidos entre si.
"""
from __future__ import annotations

from dataclasses import dataclass

import advisor_report_parser as parser

VALID_CLASSES = ("compute_bound", "memory_bound", "ambiguous_loop", "invalid")
VALID_KERNEL_CLASSES = ("compute_bound", "memory_bound", "ambiguous", "invalid")

# --- Constantes declaradas (seccion 2 del documento de diseno) -----------

# Techo de memoria: DRAM, variante "single node" -- coincide con el dominio
# real de la corrida (6 hilos anclados a cores 0-5, todos en el mismo nodo
# NUMA, confirmado con lscpu en paccaA100). La variante "DRAM Bandwidth"
# (sin "single node") mide un escenario distinto (posiblemente cruzando
# NUMA) que no corresponde a como se corre este pipeline.
MEMORY_ROOF_NAME = "DRAM Bandwidth (single node)"

# Techos de computo candidatos, por precision -- el pico FMA (no el de Add
# ni el escalar) porque la clasificacion compara contra la capacidad
# absoluta del hardware, no contra lo que ese loop especifico logro
# vectorizar (ver justificacion "por que FMA" en el documento de diseno
# 2.2 -- evitar el sesgo circular de "mal vectorizado = parece compute
# bound").
COMPUTE_ROOF_NAME_BY_PRECISION = {
    "dp": "DP Vector FMA Peak",
    "sp": "SP Vector FMA Peak",
}

# Cobertura minima de tiempo propio acumulado para considerar "suficientes"
# los loops calientes de un kernel (regla 80/20 de hot-path, criterio
# estandar de profiling, declarado explicitamente como parametro de diseno
# -- no derivado matematicamente de nada mas fundamental).
HOT_LOOP_COVERAGE_FRACTION = 0.80

# Margen multiplicativo alrededor de i_ridge_advisor para marcar un loop
# individual como ambiguo en vez de forzar compute/memory. Multiplicativo
# (no de puntos porcentuales) porque AI e i_ridge son ambos valores de
# intensidad (FLOP/byte) comparados en escala logaritmica en cualquier
# grafico Roofline -- un margen aditivo no tendria sentido dimensional.
# Justificado por la advertencia ya documentada de que la simulacion de
# cache de Advisor extrapola desde un subconjunto de accesos (no es exacta
# al 100%) -- un margen de tolerancia es la forma honesta de no
# sobre-afirmar precision que el mecanismo de medicion no tiene.
AMBIGUOUS_AI_LOG_MARGIN = 0.25  # ±25%: ambiguo si 0.8*i_ridge <= AI <= 1.25*i_ridge

# Diferencia minima de fraccion de tiempo propio (entre la clase lider y la
# segunda, sobre los loops calientes) para que el kernel completo adopte
# esa clase en vez de quedar ambiguo. 15pp declarado explicitamente --
# mismo espiritu que MEMORY_VS_CORE_MARGIN_PP del clasificador de VTune
# (raperezp/validation_classifier.py), adaptado a este contexto.
KERNEL_DOMINANCE_MARGIN_PP = 15.0


@dataclass
class RidgePoint:
    precision: str  # "dp" | "sp"
    p_peak_flops_per_s: float
    bw_peak_bytes_per_s: float
    i_ridge_flops_per_byte: float
    compute_roof_name: str
    memory_roof_name: str


@dataclass
class LoopVerdict:
    loop_name: str | None
    source_location: str | None
    self_time_s: float | None
    self_time_pct: float | None
    precision: str | None
    arithmetic_intensity: float | None
    self_gflop: float | None
    self_dram_gb: float | None
    verdict_class: str
    reason: str
    i_ridge_used: float | None


@dataclass
class KernelVerdict:
    kernel: str
    klass: str
    advisor_roofline_class: str
    confidence: str
    reason: str
    hot_loops_considered: int
    hot_loops_total_self_time_s: float
    coverage_fraction_achieved: float | None


def compute_ridge_point(roofs: dict[str, dict[str, object]], precision: str) -> RidgePoint | None:
    """i_ridge_advisor = P_peak / BW_peak, ambos leidos de
    `advisor --report=roofs` de ESTA corrida (nunca de una ficha tecnica ni
    de un valor de otra corrida) -- independiente del i_ridge de Hyperion."""
    compute_roof_name = COMPUTE_ROOF_NAME_BY_PRECISION.get(precision)
    if compute_roof_name is None:
        return None
    p_peak_entry = roofs.get(compute_roof_name)
    bw_peak_entry = roofs.get(MEMORY_ROOF_NAME)
    if p_peak_entry is None or bw_peak_entry is None:
        return None
    p_peak = p_peak_entry["value"]
    bw_peak = bw_peak_entry["value"]
    if not bw_peak:
        return None
    return RidgePoint(
        precision=precision, p_peak_flops_per_s=p_peak, bw_peak_bytes_per_s=bw_peak,
        i_ridge_flops_per_byte=p_peak / bw_peak,
        compute_roof_name=compute_roof_name, memory_roof_name=MEMORY_ROOF_NAME,
    )


def classify_loop(row: dict[str, str], roofs: dict[str, dict[str, object]]) -> LoopVerdict:
    loop_name = parser.loop_name(row)
    source = parser.loop_source_location(row)
    self_time_s = parser.loop_self_time_seconds(row)
    self_time_pct = parser.loop_self_time_percent(row)
    precision = parser.determine_loop_precision(row)
    self_gflop = parser.loop_self_gflop(row)
    self_dram_gb = parser.loop_self_dram_gb(row)

    base = dict(
        loop_name=loop_name, source_location=source, self_time_s=self_time_s,
        self_time_pct=self_time_pct, precision=precision, self_gflop=self_gflop,
        self_dram_gb=self_dram_gb,
    )

    if precision is None:
        return LoopVerdict(
            **base, arithmetic_intensity=None, verdict_class="invalid", i_ridge_used=None,
            reason="No se pudo determinar precision (Data Types/conteo dinamico dp_compute/"
                   "sp_compute ausentes o inconsistentes) -- no se puede elegir compute roof.",
        )
    if precision == "mixed":
        return LoopVerdict(
            **base, arithmetic_intensity=None, verdict_class="invalid", i_ridge_used=None,
            reason="Loop mezcla FP32 y FP64 realmente ejecutado (evidencia dinamica) -- "
                   "un solo compute roof no representa este loop, se marca invalido en vez "
                   "de forzar una precision.",
        )

    ridge = compute_ridge_point(roofs, precision)
    if ridge is None:
        return LoopVerdict(
            **base, arithmetic_intensity=None, verdict_class="invalid", i_ridge_used=None,
            reason=f"No se encontraron los roofs necesarios ({COMPUTE_ROOF_NAME_BY_PRECISION.get(precision)!r} "
                   f"y/o {MEMORY_ROOF_NAME!r}) en roofs.csv de esta corrida.",
        )

    if self_dram_gb is None or self_gflop is None:
        return LoopVerdict(
            **base, arithmetic_intensity=None, verdict_class="invalid", i_ridge_used=ridge.i_ridge_flops_per_byte,
            reason="Self GFLOP o Self DRAM GB ausentes -- probablemente la coleccion no uso "
                   "--enable-cache-simulation (Self DRAM GB solo existe con ella) o el loop "
                   "no ejecuto trafico de memoria medible.",
        )
    if self_dram_gb == 0:
        return LoopVerdict(
            **base, arithmetic_intensity=float("inf"), verdict_class="compute_bound",
            i_ridge_used=ridge.i_ridge_flops_per_byte,
            reason="Self DRAM GB=0 (sin trafico simulado a DRAM, todo servido por cache) -- "
                   "intensidad aritmetica no acotada, compute_bound por ausencia total de "
                   "presion de memoria en este nivel.",
        )

    ai = (self_gflop * 1e9) / (self_dram_gb * 1e9)  # FLOP / byte, unidades consistentes
    low = ridge.i_ridge_flops_per_byte * (1 - AMBIGUOUS_AI_LOG_MARGIN)
    high = ridge.i_ridge_flops_per_byte * (1 + AMBIGUOUS_AI_LOG_MARGIN)

    if low <= ai <= high:
        verdict_class = "ambiguous_loop"
        reason = (f"AI={ai:.3f} FLOP/byte dentro del margen declarado (±{AMBIGUOUS_AI_LOG_MARGIN:.0%}) "
                  f"de i_ridge_advisor={ridge.i_ridge_flops_per_byte:.3f} ({precision.upper()}, "
                  f"{ridge.compute_roof_name}/{ridge.memory_roof_name}) -- no hay margen suficiente "
                  "para descartar el error de simulacion de cache.")
    elif ai < ridge.i_ridge_flops_per_byte:
        verdict_class = "memory_bound"
        reason = (f"AI={ai:.3f} FLOP/byte < i_ridge_advisor={ridge.i_ridge_flops_per_byte:.3f} "
                  f"({precision.upper()}, {ridge.compute_roof_name}/{ridge.memory_roof_name}).")
    else:
        verdict_class = "compute_bound"
        reason = (f"AI={ai:.3f} FLOP/byte > i_ridge_advisor={ridge.i_ridge_flops_per_byte:.3f} "
                  f"({precision.upper()}, {ridge.compute_roof_name}/{ridge.memory_roof_name}).")

    return LoopVerdict(**base, arithmetic_intensity=ai, verdict_class=verdict_class,
                        i_ridge_used=ridge.i_ridge_flops_per_byte, reason=reason)


def select_hot_loops(rows: list[dict[str, str]],
                      coverage_fraction: float = HOT_LOOP_COVERAGE_FRACTION
                      ) -> tuple[list[dict[str, str]], float, float]:
    """Ordena por Self Time descendente y toma el minimo prefijo que cubre
    `coverage_fraction` del tiempo propio total. Devuelve (loops_calientes,
    cobertura_lograda, tiempo_propio_total_del_kernel)."""
    timed = [(r, parser.loop_self_time_seconds(r) or 0.0) for r in rows]
    total = sum(t for _, t in timed)
    if total <= 0:
        return [], 0.0, 0.0
    timed.sort(key=lambda rt: rt[1], reverse=True)
    hot: list[dict[str, str]] = []
    acc = 0.0
    for row, t in timed:
        if acc / total >= coverage_fraction and hot:
            break
        hot.append(row)
        acc += t
    return hot, acc / total, total


def aggregate_kernel_verdict(kernel: str, klass: str, hot_loop_verdicts: list[LoopVerdict],
                              coverage_fraction_achieved: float) -> KernelVerdict:
    """Pondera por Self Time entre los loops calientes ya clasificados
    individualmente -- ver justificacion en el documento de diseno 2.4."""
    usable = [v for v in hot_loop_verdicts if v.verdict_class in ("compute_bound", "memory_bound")
              and v.self_time_s]
    total_hot_time = sum((v.self_time_s or 0.0) for v in hot_loop_verdicts)

    if not usable:
        return KernelVerdict(
            kernel=kernel, klass=klass, advisor_roofline_class="invalid", confidence="NA",
            reason="Ningun loop caliente produjo un veredicto compute/memory_bound valido "
                   "(todos invalid o ambiguous_loop) -- no hay evidencia suficiente.",
            hot_loops_considered=len(hot_loop_verdicts), hot_loops_total_self_time_s=total_hot_time,
            coverage_fraction_achieved=coverage_fraction_achieved,
        )

    time_by_class: dict[str, float] = {}
    for v in usable:
        time_by_class[v.verdict_class] = time_by_class.get(v.verdict_class, 0.0) + (v.self_time_s or 0.0)

    ranked = sorted(time_by_class.items(), key=lambda kv: kv[1], reverse=True)
    leader_class, leader_time = ranked[0]
    runner_up_time = ranked[1][1] if len(ranked) > 1 else 0.0
    usable_total = sum(time_by_class.values())
    leader_pct = 100.0 * leader_time / usable_total if usable_total else 0.0
    runner_up_pct = 100.0 * runner_up_time / usable_total if usable_total else 0.0
    margin_pp = leader_pct - runner_up_pct

    detail = "; ".join(f"{cls}={100.0 * t / usable_total:.1f}% del tiempo caliente clasificable"
                        for cls, t in ranked)

    if margin_pp >= KERNEL_DOMINANCE_MARGIN_PP:
        return KernelVerdict(
            kernel=kernel, klass=klass, advisor_roofline_class=leader_class,
            confidence="alta" if margin_pp >= 2 * KERNEL_DOMINANCE_MARGIN_PP else "media",
            reason=f"{leader_class} domina por {margin_pp:.1f}pp de tiempo propio caliente "
                   f"(margen declarado={KERNEL_DOMINANCE_MARGIN_PP:.0f}pp). {detail}.",
            hot_loops_considered=len(hot_loop_verdicts), hot_loops_total_self_time_s=total_hot_time,
            coverage_fraction_achieved=coverage_fraction_achieved,
        )
    return KernelVerdict(
        kernel=kernel, klass=klass, advisor_roofline_class="ambiguous", confidence="baja",
        reason=f"Ninguna clase domina por el margen declarado ({KERNEL_DOMINANCE_MARGIN_PP:.0f}pp) "
               f"entre los loops calientes clasificables -- kernel mixto o evidencia insuficiente. {detail}.",
        hot_loops_considered=len(hot_loop_verdicts), hot_loops_total_self_time_s=total_hot_time,
        coverage_fraction_achieved=coverage_fraction_achieved,
    )
