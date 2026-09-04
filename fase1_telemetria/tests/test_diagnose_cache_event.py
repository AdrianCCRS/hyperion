"""F1-CPU-003: pruebas herméticas del análisis de `perf list` del diagnóstico
de traducción del evento de caché. No ejecutan `perf` ni tocan hardware."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria.diagnose_cache_event import _analyse_perf_list


_PERF_LIST_ICELAKE = """\
  cache-misses                                       [Hardware event]
  cache-references                                   [Hardware event]
  LLC-load-misses                                    [Hardware cache event]
  LLC-loads                                          [Hardware cache event]
  mem_load_retired.l3_miss                           [Kernel PMU event]
  longest_lat_cache.miss                             [Kernel PMU event]
  longest_lat_cache.reference                        [Kernel PMU event]
"""


def test_extrae_lineas_de_los_alias_genericos():
    a = _analyse_perf_list(_PERF_LIST_ICELAKE)
    assert a["cache_misses_line"] and "cache-misses" in a["cache_misses_line"]
    assert a["cache_references_line"] and "cache-references" in a["cache_references_line"]


def test_recolecta_lineas_relacionadas_con_llc():
    a = _analyse_perf_list(_PERF_LIST_ICELAKE)
    joined = "\n".join(a["llc_lines"]).lower()
    assert "longest_lat_cache" in joined
    assert "llc-load-misses" in joined
    # no incluye los alias genéricos (no llevan LLC/L3 en el nombre)
    assert "cache-misses  " not in joined


def test_sin_evidencia_llc_devuelve_listas_vacias():
    a = _analyse_perf_list("  branches   [Hardware event]\n  branch-misses   [Hardware event]\n")
    assert a["cache_misses_line"] is None
    assert a["cache_references_line"] is None
    assert a["llc_lines"] == []
