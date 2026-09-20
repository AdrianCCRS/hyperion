import pandas as pd

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import (
    add_physical_features,
    frequency_sensitivity,
    nested_selection,
)


def _dataset():
    rows = []
    for family, label, base in [("compute_a", "compute_bound", 90), ("compute_b", "compute_bound", 85),
                                ("memory_a", "memory_bound", 20), ("memory_b", "memory_bound", 25)]:
        for level in ("F0", "F1"):
            for rep in range(2):
                rows.append({"kernel_family": family, "phase_label_train": label, "gpu_freq_level_id": level,
                             "gpu_util_pct_median": base + rep, "gpu_mem_util_pct_median": 10,
                             "gpu_power_mw_median": 100000 + base, "gpu_sm_clock_mhz_median": 1000})
    return pd.DataFrame(rows)


def test_seleccion_anidada_no_usa_el_fold_externo_para_elegir():
    result = nested_selection(_dataset(), {"median": ["gpu_util_pct_median", "gpu_mem_util_pct_median",
                                                        "gpu_power_mw_median", "gpu_sm_clock_mhz_median"]},
                              "lofo", seed=1, model_names={"mayoritaria", "arbol_prof1"})
    assert result["n_families"] == 4
    assert len(result["per_outer_fold"]) == 4
    assert set(result["selection_by_outer_fold"]) == set(result["per_outer_fold"])


def test_sensibilidad_por_frecuencia_reporta_cada_nivel():
    rows = frequency_sensitivity(_dataset(), {"median": ["gpu_util_pct_median", "gpu_mem_util_pct_median",
                                                           "gpu_power_mw_median", "gpu_sm_clock_mhz_median"]},
                                 seed=1, model_names={"arbol_prof1"})
    assert len(rows) == 1
    assert set(rows[0]["per_level"]) == {"F0", "F1"}


def test_features_fisicas_derivan_potencia_y_ratios_sin_duracion_cruda():
    frame = pd.DataFrame([{
        "gpu_util_pct_median": 50, "gpu_mem_util_pct_median": 25, "gpu_power_mw_median": 100000,
        "gpu_util_pct_min": 20, "gpu_util_pct_max": 80, "gpu_power_mw_min": 90000, "gpu_power_mw_max": 110000,
        "gpu_sm_clock_mhz_min": 900, "gpu_sm_clock_mhz_max": 1000,
        "gpu_energy_delta_mj_sum": 1000, "covered_duration_ns": 10_000_000,
    }])
    result = add_physical_features(frame)
    assert result.loc[0, "gpu_power_mw_per_util_pct"] == 2000
    assert result.loc[0, "gpu_mem_util_to_gpu_util"] == 0.5
    assert result.loc[0, "gpu_util_pct_range"] == 60
    assert result.loc[0, "gpu_energy_mean_power_w"] == 100
