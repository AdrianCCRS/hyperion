"""Clasificacion de kernels NPB segun las metricas de VTune. Ver PLAN.md
Fase 4.2/4.3 y Fase 6, y context/02_decisiones.md D3 (recalibrada,
2026-08-07).

`clasificar_nativo()` ya NO es "sin calibracion" -- decision tomada con el
usuario tras encontrar que STREAM (memory-bound por construccion) caia en
'ambiguous' comparando Memory Bound contra su propio complemento (formula
anterior, simetrica en 50%). Los datos reales mostraron una brecha enorme
entre las anclas (Memory Bound: DGEMM=8.7%/EP=6.1% vs STREAM=51.9%; DRAM
Bound: 2.2%/0.0% vs 67.7%) que una frontera fija en 50% no refleja. Se
decidio calibrar la posicion de cada kernel relativa a los anclas DGEMM
(computo) y STREAM (memoria), combinando Memory Bound + Cache Bound + DRAM
Bound -- las tres metricas que VTune expone del mismo reporte hpc-
performance. El nombre `classification_vtune_native`/`clasificar_nativo` se
mantiene por continuidad con el resto del plan y el esquema del CSV, aunque
ya no sea "nativo" en el sentido original de "sin tocar STREAM/DGEMM".

`eje_techos()` sigue igual (D3 revisada, Fase 4.3) -- columna informativa
aparte, nunca fusionada con `classification_vtune_native` (D8).
"""
from __future__ import annotations

CLASES_VALIDAS = ("compute_bound", "memory_bound", "ambiguous", "invalid")

# Metricas usadas para la posicion relativa compute<->memoria. Las tres
# vienen del mismo reporte de hpc-performance (kernel y anclas), ninguna
# depende de uncore mas alla de lo que ya se confirmo disponible en Fase 0.
METRICAS_CALIBRACION = ("memory_bound_pct", "cache_bound_pct", "dram_bound_pct_or_na")


def _posicion_relativa(valor: float | None, ancla_compute: float | None,
                        ancla_memoria: float | None) -> float | None:
    """0.0 = tan 'computo' como el ancla DGEMM, 1.0 = tan 'memoria' como el
    ancla STREAM. Puede salir fuera de [0,1] si el kernel es mas extremo que
    cualquiera de las dos anclas -- valido, no se recorta a proposito."""
    if valor is None or ancla_compute is None or ancla_memoria is None:
        return None
    span = ancla_memoria - ancla_compute
    if abs(span) < 1e-9:
        return None
    return (valor - ancla_compute) / span


def clasificar_nativo(reporte_hpc: dict, ancla_compute: dict, ancla_memoria: dict,
                       margen: float = 0.15) -> tuple[str, str, str]:
    """PLAN.md Fase 4.2 (D3 recalibrada, 2026-08-07).

    reporte_hpc: dict de vtune_parser.parse_summary_text() del kernel a
        clasificar (UNA corrida de hpc-performance de ese kernel).
    ancla_compute: mismo formato, de la corrida hpc-performance de DGEMM
        (calibration/dgemm_hpc en la estructura de Fase 3).
    ancla_memoria: idem, de la corrida hpc-performance de STREAM
        (calibration/stream_hpc).
    margen: ancho de la zona ambigua alrededor del punto medio calibrado
        (0.5 en la escala relativa 0..1). Sigue siendo un criterio
        declarado, no derivado -- igual que el margen anterior, pero ahora
        aplicado a una escala anclada empiricamente en vez de un 50% fijo
        sin referencia.

    Requiere que ambas anclas ya se hayan corrido (Fase 3, calibration/) --
    si no estan disponibles, no se puede calcular la posicion relativa y el
    resultado es 'invalid'. Esta es una diferencia real respecto a la
    version anterior de esta funcion: ya no funciona con una sola corrida
    aislada del kernel.

    Devuelve (clase, confianza, justificacion). clase en CLASES_VALIDAS.
    """
    mem = reporte_hpc.get("memory_bound_pct")
    if mem is None:
        return "invalid", "NA", "Memory Bound no disponible en el reporte del kernel"

    if not ancla_compute or not ancla_memoria:
        return "invalid", "NA", "Anclas DGEMM/STREAM no disponibles -- no se puede calibrar (ver calibration/calibration_summary.json)"

    posiciones = []
    detalle = []
    for metrica in METRICAS_CALIBRACION:
        pos = _posicion_relativa(
            reporte_hpc.get(metrica), ancla_compute.get(metrica), ancla_memoria.get(metrica)
        )
        if pos is not None:
            posiciones.append(pos)
            detalle.append(f"{metrica}={reporte_hpc.get(metrica):.1f}% (pos={pos:.2f})")

    if not posiciones:
        return "invalid", "NA", (
            "Ninguna de Memory/Cache/DRAM Bound se pudo calibrar contra las anclas "
            "(faltan valores en el kernel o en las anclas)"
        )

    posicion = sum(posiciones) / len(posiciones)
    detalle_txt = "; ".join(detalle)

    if posicion > 0.5 + margen:
        return (
            "memory_bound",
            "alta_confianza",
            f"Posicion relativa promedio={posicion:.2f} (0=DGEMM, 1=STREAM), "
            f"por encima de 0.5+{margen:.2f} -> cerca del ancla de memoria. {detalle_txt}",
        )
    if posicion < 0.5 - margen:
        return (
            "compute_bound",
            "alta_confianza",
            f"Posicion relativa promedio={posicion:.2f} (0=DGEMM, 1=STREAM), "
            f"por debajo de 0.5-{margen:.2f} -> cerca del ancla de computo. {detalle_txt}",
        )
    return (
        "ambiguous",
        "zona_intermedia",
        f"Posicion relativa promedio={posicion:.2f}, dentro de la zona "
        f"ambigua [{0.5 - margen:.2f}, {0.5 + margen:.2f}]. {detalle_txt}",
    )


def eje_techos(dp_gflops_kernel: float | None, dgemm_gflops_ref: float | None) -> str:
    """PLAN.md Fase 4.3 (D3 revisada). Columna informativa e independiente
    de clasificar_nativo() -- nunca compite con classification_vtune_native
    (D8). Solo compara el eje de computo; no hay eje de memoria para
    kernels NPB en este nodo (sin uncore ni LIKWID, ver context/04)."""
    if dp_gflops_kernel is None or not dgemm_gflops_ref:
        return "NA - DP GFLOPS no disponible"
    pct_del_techo = 100.0 * dp_gflops_kernel / dgemm_gflops_ref
    return f"{pct_del_techo:.1f}% del techo de computo (DGEMM={dgemm_gflops_ref:.1f} GFLOP/s)"


def nota_revision_manual(kernel: str, clase: str) -> str | None:
    """Decision D4 (context/02_decisiones.md): EP no se descarta, pero si
    sale memory_bound o ambiguous se marca para revision manual en vez de
    aceptarse sin mas -- riesgo de subconteo de FLOPs (sqrt/log/rand no
    siempre entran en el contador de DP GFLOPS). Se trata igual para IS si
    se retoma en este nodo. Devuelve None si no aplica ninguna bandera."""
    if kernel.lower() in ("ep", "is") and clase in ("memory_bound", "ambiguous"):
        return (
            f"Kernel '{kernel}' salio '{clase}' -- requiere revision manual "
            "antes de aceptarse (D4: riesgo de subconteo de FLOPs / kernel "
            "sin FLOPs, ver context/03_kernels_notas.md)."
        )
    return None
