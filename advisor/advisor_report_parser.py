"""Parser de los reportes OFICIALES y estructurados de Advisor (CSV real,
via `advisor --report=survey/roofs --format=csv`) -- nunca scraping de la
GUI. Nombres de columna verificados contra una corrida real en paccaA100
(2023.0.0, EP.A, 2026-08-11): 145 columnas confirmadas en el header real de
`--report=survey --format=csv --show-all-columns --mix --dynamic`. Ver
docs/advisor/pipeline_advisor_diseno_y_arquitectura.md seccion 1 para la
evidencia completa.

Formato real del archivo (confirmado, no documentado por Intel a este
nivel de detalle en ningun lado publico que encontramos):

    sep=,
    <linea en blanco>
    "Intel(R) Advisor Command Line Tool
    Copyright (C) 2009-2023 Intel Corporation. All rights reserved."
    "Survey Data version=1.1.0","delimiter=,"
    <linea en blanco>
    "ID","Function Call Sites and Loops",...  <- header real, recien aqui
    "3","[loop in ...]",...                    <- filas de datos

El banner de copyright ocupa una sola celda CSV con un salto de linea
adentro (comillas balanceadas) -- por eso se usa el modulo csv estandar
(maneja campos multilinea correctamente) en vez de partir por lineas a
mano, que rompería justo en esa celda.
"""
from __future__ import annotations

import csv
import io
import re

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float(raw: str | None) -> float | None:
    """Tolerante a formatos reales observados: '5.103', '< 0.001s', '57.447',
    '4.050s', '', celdas vacias. Nunca lanza -- None si no hay numero."""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text == "N/A":
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_survey_csv(csv_text: str) -> list[dict[str, str]]:
    """Devuelve una fila (dict columna->valor crudo, tal cual el CSV) por
    cada loop/funcion reportado. No convierte tipos aqui -- eso lo hace el
    llamador via _to_float() sobre las columnas especificas que necesite,
    para no perder silenciosamente columnas que no se esperaban."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0] == "ID" and len(row) > 1 and "Function Call Sites" in row[1]:
            header_idx = i
            break
    if header_idx is None:
        return []

    header = rows[header_idx]
    out: list[dict[str, str]] = []
    for row in rows[header_idx + 1:]:
        if not row or not row[0]:
            continue
        record = dict(zip(header, row))
        out.append(record)
    return out


def parse_roofs_csv(csv_text: str) -> dict[str, dict[str, object]]:
    """Devuelve {nombre_del_roof: {"value": float (bytes/s o FLOP/s, SIN
    convertir de prefijo), "type": "memory"|"compute", "device": str}}.
    Nombres de roof tal cual los reporta Advisor -- ver la lista real
    capturada en docs/advisor/pipeline_advisor_diseno_y_arquitectura.md
    seccion 1 (DRAM Bandwidth (single node), DP Vector FMA Peak, etc.)."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    out: dict[str, dict[str, object]] = {}
    header_idx = None
    for i, row in enumerate(rows):
        if row[:1] == ["Name"]:
            header_idx = i
            break
    if header_idx is None:
        return out
    for row in rows[header_idx + 1:]:
        if len(row) < 4:
            continue
        name, bandwidth_raw, roof_type, device = row[0], row[1], row[2], row[3]
        value = _to_float(bandwidth_raw)
        if value is None:
            continue
        out[name] = {"value": value, "type": roof_type, "device": device}
    return out


# --------------------------------------------------------------------------
# Accesores tipados para las columnas que classify_roofline.py realmente usa
# (documentado explicitamente cuales son -- el resto de las 145 columnas
# quedan disponibles en el dict crudo por si se necesitan despues, pero no
# se hardcodea una lista completa de "las que importan" mas alla de estas).
# --------------------------------------------------------------------------


def loop_self_time_seconds(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Self Time"))


def loop_self_time_percent(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Self Time Percent"))


def loop_self_gflop(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Self GFLOP"))


def loop_self_gintop(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Self GINTOP"))


def loop_self_dram_gb(row: dict[str, str]) -> float | None:
    """Bytes (en GB) que el simulador de cache de Advisor estima que
    llegaron a DRAM para este loop -- requiere que la coleccion haya usado
    --enable-cache-simulation; en una coleccion sin simular, esta columna
    sale vacia/None (nunca se rellena con un valor inventado)."""
    return _to_float(row.get("Self DRAM GB"))

def loop_source_location(row: dict[str, str]) -> str | None:
    return row.get("Source Location") or None


def loop_name(row: dict[str, str]) -> str | None:
    return row.get("Function Call Sites and Loops") or None


def loop_data_types(row: dict[str, str]) -> str | None:
    """P.ej. 'Float64; UInt64', 'Float64', 'Float32' -- evidencia REAL de
    precision, medida por instrumentacion, nunca asumida. Ver
    kernel_registry.py para el hint (no autoritativo) de codigo fuente."""
    return row.get("Data Types") or None


def loop_vector_isa(row: dict[str, str]) -> str | None:
    return row.get("Vector ISA") or None


def loop_traits(row: dict[str, str]) -> str | None:
    return row.get("Traits") or None


def loop_dynamic_dp_compute(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Dynamic dp_compute"))


def loop_dynamic_sp_compute(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Dynamic sp_compute"))


def loop_dynamic_int_compute(row: dict[str, str]) -> float | None:
    return _to_float(row.get("Dynamic int_compute"))


def loop_dynamic_fma(row: dict[str, str]) -> float | None:
    """Suma de FMA vectorial + escalar realmente ejecutado -- evidencia
    dinamica real de uso de FMA, no inferido de que el compilador lo haya
    emitido (emitido != ejecutado, un branch puede evitarlo en runtime)."""
    vec = _to_float(row.get("Dynamic fma_vector_compute")) or 0.0
    scal = _to_float(row.get("Dynamic fma_scalar_compute")) or 0.0
    if row.get("Dynamic fma_vector_compute") is None and row.get("Dynamic fma_scalar_compute") is None:
        return None
    return vec + scal


def determine_loop_precision(row: dict[str, str]) -> str | None:
    """'dp' | 'sp' | 'mixed' | None, a partir de evidencia REAL medida
    (columna Data Types + conteo dinamico dp/sp_compute), nunca asumida.
    Si Data Types no distingue con claridad, se usa el conteo dinamico como
    desempate -- documentado, no silencioso."""
    data_types = (loop_data_types(row) or "").lower()
    has_f64 = "float64" in data_types
    has_f32 = "float32" in data_types
    if has_f64 and not has_f32:
        return "dp"
    if has_f32 and not has_f64:
        return "sp"
    dp_count = loop_dynamic_dp_compute(row)
    sp_count = loop_dynamic_sp_compute(row)
    if dp_count is not None and sp_count is not None:
        if dp_count > 0 and sp_count == 0:
            return "dp"
        if sp_count > 0 and dp_count == 0:
            return "sp"
        if dp_count > 0 and sp_count > 0:
            return "mixed"
    if has_f64 and has_f32:
        return "mixed"
    return None
