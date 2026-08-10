"""Parser de `vtune -report summary -r <dir>` para resultados de
Microarchitecture Exploration (VTune 2023, paccaA100).

Advertencia deliberada, en la misma linea que pipelinevtune/vtune_parser.py:
los nombres de etiqueta de abajo (`Retiring`, `Back-End Bound`, `Memory
Bound`, `Core Bound`, ...) son los que documenta Intel para el viewpoint
"Microarchitecture Exploration" del Top-Down Microarchitecture Analysis
Method (TMAM) y los que este proyecto observo en la practica para
hpc-performance/hotspots en este mismo VTune 2023.0.0 -- pero uarch-
exploration en si NUNCA se corrio con exito en este nodo antes de este
permiso nuevo (ver docs/vtune/Informe_VTune_Profiler.md §4.1 y
pipelinevtune/context/04). Es decir: este parser esta escrito a partir de
documentacion de Intel + el patron ya confirmado de otros analisis de este
mismo VTune, NO de una captura real de uarch-exploration en este nodo.

Regla del proyecto que aplica aqui igual que en todos lados: la PRIMERA
corrida real de la campana debe reconciliar estos nombres contra la salida
real (guardada en <kernel>/summary_raw.txt por run_validation.py) antes de
confiar en las columnas de consolidated_validation.csv. Si algo no
coincide, se corrige este archivo y se documenta el cambio -- no se fuerza
el parser a "encontrar" un numero donde no lo hay.

Tolerante a metricas ausentes: una etiqueta no encontrada o 'N/A' queda en
None, nunca lanza excepcion.
"""
from __future__ import annotations

import re

_NUM_RE = r"(N/A|[\d.]+)"


def _num(texto: str, etiqueta: str) -> float | None:
    """Busca '<etiqueta>: <numero>' al inicio de linea (indentacion
    arbitraria), tolerando texto despues del numero (ej. '34.5% of
    Pipeline Slots', '1.234 GHz'). None si la etiqueta no aparece o es N/A.
    """
    patron = re.compile(rf"^[ \t]*{re.escape(etiqueta)}:\s*{_NUM_RE}", re.MULTILINE)
    m = patron.search(texto)
    if not m or m.group(1) == "N/A":
        return None
    return float(m.group(1))


def _entero_con_comas(texto: str, etiqueta: str) -> int | None:
    m = re.search(rf"^[ \t]*{re.escape(etiqueta)}:\s*(N/A|[\d,]+)", texto, re.MULTILINE)
    if not m or m.group(1) == "N/A":
        return None
    return int(m.group(1).replace(",", ""))


def _texto(texto: str, etiqueta: str) -> str | None:
    m = re.search(rf"^[ \t]*{re.escape(etiqueta)}:\s*(.+)$", texto, re.MULTILINE)
    return m.group(1).strip() if m else None


# Las 4 categorias de Nivel 1 del TMAM (Yasin 2014; ver docs/vtune/
# vtune_cross_validation.md seccion E para la metodologia completa). Suman
# ~100% de "Pipeline Slots" por construccion del modelo.
TOP_LEVEL_LABELS = {
    "retiring_pct": "Retiring",
    "frontend_bound_pct": "Front-End Bound",
    "bad_speculation_pct": "Bad Speculation",
    "backend_bound_pct": "Back-End Bound",
}

# Subcategorias de Nivel 2 relevantes para este proyecto (no se extraen
# todas las que VTune pueda imprimir -- solo las que la clasificacion de
# validation_classifier.py realmente usa, siguiendo el mismo principio de
# "pocas metricas, pero defendibles" que pide el CSV consolidado).
LEVEL2_LABELS = {
    "memory_bound_pct": "Memory Bound",
    "core_bound_pct": "Core Bound",
    "light_operations_pct": "Light Operations",
    "heavy_operations_pct": "Heavy Operations",
    "branch_mispredict_pct": "Branch Mispredict",
    "machine_clears_pct": "Machine Clears",
}

# Nivel 3, bajo Memory Bound -- el desglose que decide si el cuello de
# botella esta cerca del core (L1/L2) o de DRAM.
LEVEL3_MEMORY_LABELS = {
    "l1_bound_pct": "L1 Bound",
    "l2_bound_pct": "L2 Bound",
    "l3_bound_pct": "L3 Bound",
    "dram_bound_pct": "DRAM Bound",
    "store_bound_pct": "Store Bound",
}

# Nivel 3, bajo Core Bound.
LEVEL3_CORE_LABELS = {
    "divider_pct": "Divider",
    "ports_utilization_pct": "Ports Utilization",
}


def parse_uarch_summary_text(texto: str) -> dict:
    """Parsea `vtune -report summary -r <uarch-exploration dir>`.

    Devuelve un dict plano (sin anidar, para que quede trivial volcarlo a
    fila de CSV) con: metadata de la corrida (elapsed_time_s, cpi_rate,
    average_cpu_frequency_ghz, instructions_retired, collector_type) + las
    4 categorias de Nivel 1 + las subcategorias de Nivel 2/3 listadas
    arriba. Todo lo demas que VTune pueda imprimir (hay mas subcategorias
    de las que este proyecto necesita) se ignora a proposito -- ver D-CSV
    en docs/vtune/vtune_cross_validation.md sobre por que el CSV final no
    exporta todo lo que VTune reporta.
    """
    out: dict[str, float | int | str | None] = {}

    out["elapsed_time_s"] = _num(texto, "Elapsed Time")
    out["cpi_rate"] = _num(texto, "CPI Rate")
    out["average_cpu_frequency_ghz"] = _num(texto, "Average CPU Frequency")
    out["instructions_retired"] = _entero_con_comas(texto, "Instructions Retired")
    out["clockticks"] = _entero_con_comas(texto, "Clockticks")
    out["collector_type"] = _texto(texto, "Collector Type")

    for key, label in TOP_LEVEL_LABELS.items():
        out[key] = _num(texto, label)
    for key, label in LEVEL2_LABELS.items():
        out[key] = _num(texto, label)
    for key, label in LEVEL3_MEMORY_LABELS.items():
        out[key] = _num(texto, label)
    for key, label in LEVEL3_CORE_LABELS.items():
        out[key] = _num(texto, label)

    out["ipc"] = (1.0 / out["cpi_rate"]) if out.get("cpi_rate") else None

    return out


def top_level_sum(parsed: dict) -> float | None:
    """Suma de las 4 categorias de Nivel 1 -- deberia rondar 100% por
    construccion del modelo TMAM. Se expone para que run_validation.py
    pueda marcar quality_status='topdown_no_suma_100' si se aleja demasiado
    (señal de metricas degradadas/multiplexacion excesiva, no un bug del
    parser) en vez de clasificar a ciegas sobre numeros que no cuadran."""
    vals = [parsed.get(k) for k in TOP_LEVEL_LABELS]
    if any(v is None for v in vals):
        return None
    return sum(vals)  # type: ignore[arg-type]


def parse_hw_events_csv(csv_text: str) -> list[dict[str, str]]:
    """Parsea `vtune -report hw-events -r <dir> -format=csv` como una lista
    de filas crudas (nombre de evento/metrica -> valor), sin interpretar
    nada. Uso: archivar los eventos de PMU realmente configurados por VTune
    para uarch-exploration en este nodo (ver seccion 5 de
    docs/vtune/vtune_cross_validation.md) -- es la fuente empirica real,
    mas confiable que cualquier lista generica de eventos de documentacion.
    Tolerante a formato vacio/inesperado: devuelve [] en vez de lanzar.
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows
