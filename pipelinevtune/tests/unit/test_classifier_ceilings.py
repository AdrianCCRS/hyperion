"""
Tests de eje_techos() (classifier.py definitivo, PLAN.md Fase 4.3). Confirma
que es una columna informativa independiente de clasificar_nativo(), y que
tolera falta de datos sin excepcion. Un caso usa numeros reales capturados
en Fase 0/3 (EP dp_gflops=9.841, DGEMM=728.719 GFLOP/s medido en Fase 3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from classifier import eje_techos


def test_ep_real_lejos_del_techo_de_computo():
    resultado = eje_techos(dp_gflops_kernel=9.841, dgemm_gflops_ref=728.719)
    assert "1.4%" in resultado
    assert "DGEMM" in resultado


def test_kernel_cerca_del_techo_de_computo():
    resultado = eje_techos(dp_gflops_kernel=180.0, dgemm_gflops_ref=200.0)
    assert "90.0%" in resultado


def test_na_cuando_falta_dp_gflops_del_kernel():
    resultado = eje_techos(dp_gflops_kernel=None, dgemm_gflops_ref=187.4)
    assert resultado.startswith("NA")


def test_na_cuando_falta_techo_de_referencia():
    # ej. el ancla DGEMM fallo o no se corrio todavia
    resultado = eje_techos(dp_gflops_kernel=50.0, dgemm_gflops_ref=None)
    assert resultado.startswith("NA")


def test_na_cuando_techo_es_cero():
    resultado = eje_techos(dp_gflops_kernel=50.0, dgemm_gflops_ref=0.0)
    assert resultado.startswith("NA")
