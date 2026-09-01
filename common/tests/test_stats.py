from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.stats import paired_significance_test


def test_diferencia_grande_y_consistente_es_significativa():
    rng = np.random.default_rng(0)
    baseline = rng.normal(100, 2, size=20)
    candidate = baseline - 20  # mejora grande y consistente
    result = paired_significance_test(baseline, candidate)
    assert result.significant
    assert bool(result) is True
    assert result.p_value < 0.05


def test_sin_diferencia_real_no_es_significativa():
    rng = np.random.default_rng(1)
    baseline = rng.normal(100, 5, size=20)
    candidate = baseline + rng.normal(0, 0.01, size=20)  # ruido mínimo, sin señal real
    result = paired_significance_test(baseline, candidate)
    assert not result.significant


def test_diferencias_todas_cero():
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    result = paired_significance_test(baseline, baseline.copy())
    assert not result.significant
    assert result.test_name == "sin_diferencia"


def test_menos_de_8_pares_usa_wilcoxon_automaticamente():
    baseline = np.array([10.0, 11.0, 9.0, 12.0])
    candidate = np.array([8.0, 9.0, 7.0, 10.0])
    result = paired_significance_test(baseline, candidate)
    assert result.test_name == "wilcoxon"


def test_force_test_respeta_la_prueba_pedida():
    rng = np.random.default_rng(2)
    baseline = rng.normal(50, 3, size=10)
    candidate = baseline - 5
    result = paired_significance_test(baseline, candidate, force_test="ttest")
    assert result.test_name == "ttest"


def test_forma_distinta_lanza_error():
    with pytest.raises(ValueError, match="misma forma"):
        paired_significance_test(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_menos_de_2_pares_lanza_error():
    with pytest.raises(ValueError, match="al menos 2 pares"):
        paired_significance_test(np.array([1.0]), np.array([2.0]))


def test_nan_se_descartan_antes_de_contar_pares():
    baseline = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    candidate = np.array([2.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
    result = paired_significance_test(baseline, candidate)
    assert result.n_pairs == 8  # una fila descartada por NaN
