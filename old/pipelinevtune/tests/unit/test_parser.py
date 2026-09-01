"""
Tests de vtune_parser.py (definitivo, PLAN.md Fase 4.1) contra capturas
REALES del nodo Cartagena (paccaA100) y contra la plantilla ilustrativa que
simula un reporte casi vacio (esa sí sigue siendo un caso sintetico valido:
no depende del formato exacto, solo prueba que N/A no rompe el parser).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vtune_parser import parse_hotspots_text, parse_summary_text

FIXTURES = Path(__file__).parent / "fixtures"


def _leer(nombre: str) -> str:
    return (FIXTURES / nombre).read_text()


def test_parser_extrae_metricas_reales_ep():
    r = parse_summary_text(_leer("real_summary_ep_C.txt"))
    assert r["dp_gflops"] == 9.841
    assert r["memory_bound_pct"] == 6.1
    assert r["cache_bound_pct"] == 11.5
    assert r["dram_bound_pct_or_na"] == 0.0
    assert r["cpi"] == 0.641


def test_parser_extrae_metricas_reales_stream():
    r = parse_summary_text(_leer("real_summary_stream.txt"))
    assert r["dp_gflops"] == 2.746
    assert r["memory_bound_pct"] == 51.9
    assert r["dram_bound_pct_or_na"] == 67.7


def test_parser_no_inventa_core_bound():
    # El reporte real no trae "Core Bound" (confirmado en Fase 0 -- el
    # desglose Top-Down de 4 categorias no esta disponible en este nodo sin
    # Microarchitecture Exploration). El parser no debe fabricar esa clave.
    r = parse_summary_text(_leer("real_summary_ep_C.txt"))
    assert "core_bound_pct" not in r


def test_parser_tolera_reporte_casi_vacio():
    r = parse_summary_text(_leer("summary_missing_metrics_template.txt"))
    assert r["memory_bound_pct"] is None
    assert r["dp_gflops"] is None
    # no debe lanzar excepcion por llegar hasta aqui


def test_parser_hotspots_extrae_funcion_dominante():
    r = parse_hotspots_text(_leer("real_hotspots_ep_C.txt"))
    assert r["dominant_function"] == "MAIN__._omp_fn.1"
    assert r["dominant_function_module"] == "ep.C.x"
    assert r["dominant_function_percentage"] == 52.3
    assert r["dominant_function_cpu_time"] == 55.505


def test_parser_hotspots_tolera_tabla_ausente():
    r = parse_hotspots_text("Elapsed Time: 1.0s\n(sin tabla Top Hotspots)")
    assert r["dominant_function"] is None
    assert r["dominant_function_percentage"] is None
