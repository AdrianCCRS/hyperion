"""
Tests de clasificar_nativo() y nota_revision_manual() (classifier.py
definitivo, PLAN.md Fase 4.2 / D4) contra capturas reales.

clasificar_nativo() ya NO es "sin calibracion" -- ver el docstring del
modulo en classifier.py y context/02_decisiones.md (D3 recalibrada,
2026-08-07) para el porque: comparar Memory Bound contra su propio
complemento (formula anterior) ponia la frontera de decision fija en 50%,
y STREAM (memory-bound por construccion, Memory Bound=51.9%) caia del lado
equivocado por pura cercania a ese punto medio simetrico, aunque estuviera
clarisimamente separado de las anclas de computo (DGEMM=8.7%, EP=6.1%).
Ahora la posicion de cada kernel se calibra contra DGEMM (computo) y
STREAM (memoria) usando Memory Bound + Cache Bound + DRAM Bound juntos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from classifier import clasificar_nativo, nota_revision_manual
from vtune_parser import parse_summary_text

FIXTURES = Path(__file__).parent / "fixtures"

# Anclas reales medidas en Fase 0/3 (calibration/dgemm_hpc_summary.txt y
# calibration/stream_hpc_summary.txt de la corrida real en paccaA100).
ANCLA_COMPUTE_DGEMM = {"memory_bound_pct": 8.7, "cache_bound_pct": 6.7, "dram_bound_pct_or_na": 2.2}
ANCLA_MEMORIA_STREAM = {"memory_bound_pct": 51.9, "cache_bound_pct": 19.2, "dram_bound_pct_or_na": 67.7}


def _clasificar_desde_fixture(nombre: str, margen: float = 0.15):
    reporte = parse_summary_text((FIXTURES / nombre).read_text())
    return clasificar_nativo(reporte, ANCLA_COMPUTE_DGEMM, ANCLA_MEMORIA_STREAM, margen=margen)


def test_ep_real_clasifica_compute_bound():
    clase, confianza, justificacion = _clasificar_desde_fixture("real_summary_ep_C.txt")
    assert clase == "compute_bound"
    assert confianza == "alta_confianza"
    assert "memory_bound_pct=6.1%" in justificacion


def test_stream_real_clasifica_memory_bound():
    # Antes de la recalibracion (formula del complemento simetrico), este
    # caso salia 'ambiguous' -- ver el hallazgo documentado en
    # context/02_decisiones.md. Con la posicion calibrada contra las
    # anclas, STREAM cae exactamente en el ancla de memoria (posicion=1.0).
    clase, confianza, justificacion = _clasificar_desde_fixture("real_summary_stream.txt")
    assert clase == "memory_bound"
    assert confianza == "alta_confianza"
    assert "dram_bound_pct_or_na=67.7%" in justificacion


def test_dgemm_clasifica_compute_bound_contra_si_mismo():
    # Caso trivial: el ancla de computo evaluada contra si misma debe caer
    # exactamente en el extremo compute_bound (posicion=0.0). Confirma que
    # la formula esta bien orientada, no una garantia sobre kernels NPB.
    clase, _, _ = clasificar_nativo(ANCLA_COMPUTE_DGEMM, ANCLA_COMPUTE_DGEMM, ANCLA_MEMORIA_STREAM)
    assert clase == "compute_bound"


def test_margen_configurable_cambia_el_resultado():
    # Con un margen absurdamente amplio, hasta STREAM (posicion=1.0) deja
    # de bastar -- confirma que el margen es un parametro real, no
    # hardcodeado dentro de la funcion.
    clase, _, _ = _clasificar_desde_fixture("real_summary_stream.txt", margen=0.6)
    assert clase == "ambiguous"


def test_invalid_cuando_falta_memory_bound_del_kernel():
    reporte = parse_summary_text((FIXTURES / "summary_missing_metrics_template.txt").read_text())
    clase, confianza, justificacion = clasificar_nativo(reporte, ANCLA_COMPUTE_DGEMM, ANCLA_MEMORIA_STREAM)
    assert clase == "invalid"
    assert confianza == "NA"
    assert "no disponible" in justificacion


def test_invalid_cuando_faltan_anclas():
    reporte = parse_summary_text((FIXTURES / "real_summary_ep_C.txt").read_text())
    clase, confianza, justificacion = clasificar_nativo(reporte, {}, {})
    assert clase == "invalid"
    assert "calibrar" in justificacion


def test_nota_revision_manual_marca_ep_cuando_no_es_compute_bound():
    assert nota_revision_manual("ep", "memory_bound") is not None
    assert nota_revision_manual("ep", "ambiguous") is not None
    assert nota_revision_manual("ep", "compute_bound") is None


def test_nota_revision_manual_no_aplica_a_otros_kernels():
    assert nota_revision_manual("cg", "memory_bound") is None
    assert nota_revision_manual("mg", "ambiguous") is None
