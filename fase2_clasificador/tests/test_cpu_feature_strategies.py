import pandas as pd

from fase2_clasificador.analysis.cpu_feature_strategies import (
    FORBIDDEN_FEATURES,
    add_online_physical_features,
    production_variants,
)
from fase2_clasificador.analysis.evaluate_cpu_feature_strategies import _prepare, _prototype
from fase2_clasificador.analysis.evaluate_cpu_selective_nested import _score


def _frame():
    return pd.DataFrame([{
        "delta_instructions": 1_000, "delta_cycles": 500,
        "delta_cache_references": 80, "delta_cache_misses": 20,
        "delta_stalled_cycles_mem_any": 100, "mpki": 20.0,
    }])


def test_deriva_interacciones_pmu_sin_columnas_de_verdad():
    result = add_online_physical_features(_frame())
    assert result.loc[0, "cache_references_per_ki"] == 80.0
    assert result.loc[0, "cache_misses_per_cycle"] == 0.04
    assert result.loc[0, "stalls_mem_per_ki"] == 100.0


def test_variantes_de_produccion_no_contienen_fuga_ni_identidad():
    variants = production_variants()
    assert "baseline_pmu" in variants
    assert all(not (set(features) & FORBIDDEN_FEATURES) for features in variants.values())
    assert all("running_ratio" not in features for features in variants.values())


def test_muestreo_por_familia_y_clase_conserva_las_columnas_de_protocolo():
    rows = []
    for family in ("a", "b"):
        for label in ("compute_bound", "memory_bound"):
            for _ in range(4):
                row = _frame().iloc[0].to_dict()
                row.update({"kernel_ref": family, "phase_label_train": label})
                rows.append(row)
    result = _prepare(pd.DataFrame(rows), max_per_family_class=2, seed=1)
    assert {"kernel_family", "_class"}.issubset(result.columns)
    assert len(result) == 8


def test_abstencion_mide_calidad_solo_sobre_la_cobertura_seleccionada():
    y = pd.Series([False, True, True]).to_numpy()
    probability = pd.Series([0.1, 0.51, 0.9]).to_numpy()
    f1, coverage = _score(y, probability, threshold=0.8)
    assert coverage == 2 / 3
    assert f1 == 1.0


def test_xgboost_no_duplica_el_balance_cuando_recibe_pesos_explicitos():
    model = _prototype("xgboost", seed=1, train_y=pd.Series([False, False, True]).to_numpy(),
                       n_jobs=1, scale_pos_weight=1.0)
    assert model.get_params()["scale_pos_weight"] == 1.0
