"""Parser de reportes de texto de VTune 2023 (`vtune -report summary` /
`-report hotspots`). Ver PLAN.md Fase 4.1.

Nombres de campo y formato confirmados contra capturas reales del nodo
Cartagena (paccaA100) en la Fase 0 -- ver
tests/unit/fixtures/real_summary_ep_C.txt y real_summary_stream.txt, y
context/04_vtune_selfchecker_resultados.md (addendum).

Diferencia deliberada respecto a las fixtures ILUSTRATIVAS de tests/unit/
(summary_*_template.txt): el reporte real de `hpc-performance` en este nodo
NO trae una "Core Bound" ni el desglose Top-Down de 4 categorias -- por eso
este parser no intenta extraer `core_bound_pct`. Ese campo no existe en la
salida real; inventarlo seria peor que no tenerlo. La "contraparte de
computo" para clasificar_nativo() se calcula en classifier.py como
complemento de memory_bound_pct, no se lee de aqui (ver PLAN.md Fase 4.2,
D3-native revisada).

Tolerante a metricas ausentes: un campo no encontrado o en 'N/A' queda en
None, nunca lanza excepcion. Si el formato cambia de forma irreconocible
(ninguno de los campos esperados aparece), parse_summary_text sigue
devolviendo un dict con todo en None -- quien llama debe decidir si eso
cuenta como "invalid" (ver clasificar_nativo, que ya trata memory_bound_pct
None como invalid).
"""
from __future__ import annotations

import re

_NUM_RE = r"(N/A|[\d.]+)"


def _extraer_numero(texto: str, etiqueta: str) -> float | None:
    """Busca '<etiqueta>: <numero>' al inicio de linea (indentacion
    arbitraria), tolerando texto despues del numero (ej. '6.1% of Pipeline
    Slots', '3.492 GHz', '48.7% (7.794 out of 16)'). Devuelve None si la
    etiqueta no aparece o si el valor es N/A."""
    patron = re.compile(rf"^[ \t]*{re.escape(etiqueta)}:\s*{_NUM_RE}", re.MULTILINE)
    m = patron.search(texto)
    if not m or m.group(1) == "N/A":
        return None
    return float(m.group(1))


def _extraer_numa(texto: str) -> float | None:
    """'NUMA: % of Remote Accesses: 0.0%' tiene dos ':' -- caso especial,
    no encaja con el patron generico de _extraer_numero."""
    m = re.search(rf"^[ \t]*NUMA:\s*%\s*of\s*Remote\s*Accesses:\s*{_NUM_RE}", texto, re.MULTILINE)
    if not m or m.group(1) == "N/A":
        return None
    return float(m.group(1))


def _extraer_bloque(texto: str, inicio_re: str, fin_re: str) -> str:
    """Devuelve la subcadena entre la primera linea que matchea `inicio_re`
    y la siguiente que matchea `fin_re` (excluida). Usado para desambiguar
    etiquetas repetidas en distintos niveles de anidacion, ej. '128-bit'
    aparece tanto bajo 'SP FLOPs' como bajo 'DP FLOPs'."""
    m_inicio = re.search(inicio_re, texto, re.MULTILINE)
    if not m_inicio:
        return ""
    resto = texto[m_inicio.end():]
    m_fin = re.search(fin_re, resto, re.MULTILINE)
    return resto[: m_fin.start()] if m_fin else resto


def parse_summary_text(texto: str) -> dict:
    """Parsea la salida de `vtune -report summary -r <hpc-performance dir>`.

    Devuelve un dict con, como minimo, las claves usadas por
    clasificar_nativo() y eje_techos() (Fase 4.2/4.3): 'memory_bound_pct' y
    'dp_gflops'. El resto son columnas adicionales del esquema de
    consolidated_results.csv / vectorization_detail.csv (Fase 5) que ya se
    pueden sacar del mismo reporte sin costo extra.

    Los porcentajes 'packed_*_pct' se leen del bloque 'DP FLOPs' (no del de
    'SP FLOPs', que tiene las mismas etiquetas '128-bit'/'256-bit'/'512-bit'
    mas arriba en el mismo reporte) -- los kernels NPB de este proyecto son
    de doble precision.
    """
    bloque_dp = _extraer_bloque(texto, r"^[ \t]*DP FLOPs:", r"^[ \t]*x87 FLOPs:")
    non_fp = _extraer_numero(texto, "Non-FP")

    return {
        "dp_gflops": _extraer_numero(texto, "DP GFLOPS"),
        "sp_gflops": _extraer_numero(texto, "SP GFLOPS"),
        "cpi": _extraer_numero(texto, "CPI Rate"),
        "average_frequency_ghz": _extraer_numero(texto, "Average CPU Frequency"),
        "physical_core_utilization_pct": _extraer_numero(
            texto, "Effective Physical Core Utilization"
        ),
        "memory_bound_pct": _extraer_numero(texto, "Memory Bound"),
        "cache_bound_pct": _extraer_numero(texto, "Cache Bound"),
        "dram_bound_pct_or_na": _extraer_numero(texto, "DRAM Bound"),
        "numa_remote_access_pct": _extraer_numa(texto),
        "vectorization_pct": _extraer_numero(texto, "Vectorization"),
        "packed_128_pct": _extraer_numero(bloque_dp, "128-bit"),
        "packed_256_pct": _extraer_numero(bloque_dp, "256-bit"),
        "packed_512_pct": _extraer_numero(bloque_dp, "512-bit"),
        "non_fp_uops_pct": non_fp,
        "fp_uops_pct": (100.0 - non_fp) if non_fp is not None else None,
        "fp_arith_mem_read_ratio": _extraer_numero(texto, "FP Arith/Mem Rd Instr. Ratio"),
        "fp_arith_mem_write_ratio": _extraer_numero(texto, "FP Arith/Mem Wr Instr. Ratio"),
    }


_HOTSPOT_ROW_RE = re.compile(
    r"^(?P<function>\S.*?)\s{2,}(?P<module>\S+)\s+(?P<cpu_time>[\d.]+)s\s+(?P<pct>[\d.]+)%",
    re.MULTILINE,
)


def _extraer_entero_con_comas(texto: str, etiqueta: str) -> int | None:
    """'Instructions Retired: 579,328,000,000' -- entero con separador de
    miles por coma, no encaja con _NUM_RE (que no acepta comas)."""
    m = re.search(rf"^[ \t]*{re.escape(etiqueta)}:\s*(N/A|[\d,]+)", texto, re.MULTILINE)
    if not m or m.group(1) == "N/A":
        return None
    return int(m.group(1).replace(",", ""))


def parse_hotspots_text(texto: str) -> dict:
    """Parsea la tabla 'Top Hotspots' de `vtune -report hotspots`. Devuelve
    la funcion dominante (primera fila, VTune ya la ordena por CPU Time
    descendente) o todo en None si la tabla no aparece o esta vacia."""
    resultado = {"instructions_retired": _extraer_entero_con_comas(texto, "Instructions Retired")}
    m = _HOTSPOT_ROW_RE.search(texto)
    if not m:
        resultado.update({
            "dominant_function": None,
            "dominant_function_module": None,
            "dominant_function_cpu_time": None,
            "dominant_function_percentage": None,
        })
        return resultado
    resultado.update({
        "dominant_function": m.group("function").strip(),
        "dominant_function_module": m.group("module"),
        "dominant_function_cpu_time": float(m.group("cpu_time")),
        "dominant_function_percentage": float(m.group("pct")),
    })
    return resultado
