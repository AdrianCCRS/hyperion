"""Contrato de la capa DVFS offline R3-A."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import dvfs

pytest.importorskip("sklearn")


def _row(config_id, operation, size, device, action, region, energy, time):
    cpu_level = action.split(":")[1]
    gpu_level = action.split(":")[2] if device == "gpu" else np.nan
    return {
        "config_id": config_id, "operation": operation, "size": size,
        "device": device, "action_id": action, "cpu_level": cpu_level,
        "gpu_level": gpu_level, "region": region,
        "energy_mean": energy, "time_mean": time, "edp_mean": energy * time,
        "eligible_repetitions": True,
    }


def _candidates():
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 96, 128, 192, 256, 384):
            config_id = f"{operation}_N{size}"
            scale = (size / 64.0) ** (3 if operation == "gemm" else 2)
            # CPU gana a REF en cold/cold; GPU gana cuando ya esta caliente
            # para gemm. Las acciones no-REF tienen respuestas distintas por
            # operacion, para que no exista una clase de frecuencia universal.
            for region, startup in (("cold", 4.0), ("warm", 1.0)):
                cpu_ref = scale * startup
                gpu_ref = (0.45 * scale + (3.0 if region == "cold" else 0.0))
                for action, factor in (("cpu:REF", 1.0), ("cpu:F0", 0.90 if operation == "gemm" else 1.05)):
                    rows.append(_row(config_id, operation, size, "cpu", action, region,
                                     cpu_ref * factor, cpu_ref))
                for action, factor in (("gpu:REF:REF", 1.0), ("gpu:REF:F0", 0.85 if operation == "gemm" else 1.08)):
                    rows.append(_row(config_id, operation, size, "gpu", action, region,
                                     gpu_ref * factor, gpu_ref))
    return pd.DataFrame(rows)


def test_dataset_elige_dispositivo_a_ref_antes_de_expandir_frecuencias():
    frame = dvfs.build_dvfs_dataset(_candidates())
    assert frame["config_id"].nunique() == 12
    assert frame["decision_group_id"].nunique() == 36
    # Cada grupo contiene acciones de un solo dispositivo y una sola region.
    assert frame.groupby("decision_group_id")["device"].nunique().max() == 1
    assert frame.groupby("decision_group_id")["region"].nunique().max() == 1
    assert frame.groupby("decision_group_id")["reference_action"].nunique().max() == 1


def test_dataset_no_convierte_filas_de_accion_en_unidades_independientes():
    frame = dvfs.build_dvfs_dataset(_candidates())
    configs = dvfs.configuration_frame(frame)
    from classifier.selector.sizes import interpolation_folds
    for _, train, test in interpolation_folds(configs, n_folds=2):
        assert set(train["config_id"]).isdisjoint(set(test["config_id"]))
        assert set(frame[frame["config_id"].isin(test["config_id"])]["config_id"]) == set(test["config_id"])


def test_abstencion_prefiere_ref_si_esta_en_la_banda_equivalente():
    base = dvfs.build_dvfs_dataset(_candidates())
    group = base[base["decision_group_id"] == base["decision_group_id"].iloc[0]].copy()
    group["pred_energy_j"] = group["energy_j"]
    group["pred_time_s"] = group["time_s"]
    # Fuerza diferencia sub-ruido entre REF y alternativa.
    ref = group["reference_action"].iloc[0]
    group.loc[group["frequency_action"] != ref, "pred_energy_j"] = (
        group.loc[group["frequency_action"] == ref, "pred_energy_j"].iloc[0] * 0.99
    )
    group.loc[group["frequency_action"] != ref, "pred_time_s"] = (
        group.loc[group["frequency_action"] == ref, "pred_time_s"].iloc[0]
    )
    decision = dvfs.choose_actions(group, model_uncertainty_pct=0.0).iloc[0]
    assert decision["selected_action"] == ref
    assert bool(decision["abstained"])


def test_overhead_puede_hacer_que_ref_sea_la_decision_neta():
    base = dvfs.build_dvfs_dataset(_candidates())
    group = base[base["decision_group_id"] == base["decision_group_id"].iloc[0]].copy()
    group["pred_energy_j"] = group["energy_j"]
    group["pred_time_s"] = group["time_s"]
    without = dvfs.choose_actions(group, model_uncertainty_pct=0.0).iloc[0]
    with_cost = dvfs.choose_actions(
        group, model_uncertainty_pct=0.0, overhead_energy_j=1e6, overhead_time_s=1e6,
    ).iloc[0]
    assert with_cost["selected_action"] == with_cost["reference_action"]
    assert with_cost["selected_edp_js"] <= without["selected_edp_js"] + 1e20


def test_evaluacion_parea_modelo_baselines_y_oraculo_en_mismos_pliegues():
    frame = dvfs.build_dvfs_dataset(_candidates())
    results, decisions = dvfs.evaluate_dvfs(frame, families=("ridge",))
    assert set(results["method"]) == {"baseline", "model"}
    assert {"always_ref", "best_constant_train", "oracle", "ridge"} <= set(results["name"])
    assert not decisions.empty
    assert (results["edp_sum_ratio_vs_oracle"] >= 1.0 - 1e-9).all()
    for (fold, state), group in results.groupby(["fold", "resource_state"], observed=True):
        assert group["n"].nunique() == 1, (fold, state)


def test_power_law_calibra_incertidumbre_por_contexto():
    frame = dvfs.build_dvfs_dataset(_candidates())
    models = dvfs.fit_cost_models(frame, "power_law")
    assert models.uncertainty_pct_by_context
    # La clave es (resource_state, device, size_regime): el regimen de
    # tamano evita que una minoria de configs dificiles contamine el umbral
    # de confianza de las demas dentro del mismo (resource_state, device).
    assert all(len(key) == 3 for key in models.uncertainty_pct_by_context)
    assert any(key[:2] == ("gpu_ready", "gpu") for key in models.uncertainty_pct_by_context)
    assert all(value >= 0 for value in models.uncertainty_pct_by_context.values())
    assert models.size_thresholds


def test_size_regime_es_relativo_a_cada_operacion():
    # axpy solo tiene tamanos grandes (31623/100000); no hay umbral absoluto
    # valido entre operaciones -- ver _size_regimes.
    thresholds = {"axpy": 65811.5, "gemm": 160.0}
    assert dvfs._size_regime("axpy", 31623, thresholds) == "small"
    assert dvfs._size_regime("axpy", 100000, thresholds) == "large"
    assert dvfs._size_regime("gemm", 128, thresholds) == "small"
    assert dvfs._size_regime("gemm", 192, thresholds) == "large"
    # operacion sin umbral conocido -> por defecto "large" (conservador).
    assert dvfs._size_regime("spmv", 10, thresholds) == "large"


def test_seleccion_es_una_familia_y_una_baseline_globales():
    frame = dvfs.build_dvfs_dataset(_candidates())
    results, _ = dvfs.evaluate_dvfs(frame, families=("ridge",))
    selected = dvfs.select_dvfs_policy(results)
    assert selected["family"] == "ridge"
    assert selected["baseline"] in {"always_ref", "best_constant_train"}
    assert isinstance(selected["adopt_model"], bool)
