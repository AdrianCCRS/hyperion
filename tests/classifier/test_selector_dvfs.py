"""Contrato de la capa DVFS offline R3-A."""
from pathlib import Path
import math
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


# --------------------------------------------------------------------------
# curve_physical -- hallazgo del plan de reformulacion, seccion 6.5-bis:
# predecir 7 parametros de la curva fisica t(f)=ta+tb/f_dev+tc/f_host en vez
# de 40 costos categoricos independientes.
# --------------------------------------------------------------------------

def test_relative_frequency_ref_siempre_es_uno():
    for level in ("REF", "F0"):
        assert dvfs._relative_frequency(level, "cpu") == pytest.approx(1.0)
        assert dvfs._relative_frequency(level, "gpu") == pytest.approx(1.0)
    # F6 real (freq_khz_observed / gpu_sm_clock_mhz) no llega al minimo
    # nominal declarado en el manifiesto -- CPU y GPU difieren entre si.
    assert dvfs._relative_frequency("F6", "cpu") == pytest.approx(0.267)
    assert dvfs._relative_frequency("F6", "gpu") == pytest.approx(0.149)
    with pytest.raises(dvfs.DVFSContractError):
        dvfs._relative_frequency("F9", "cpu")


def test_device_host_frequency_cpu_usa_el_mismo_nivel_dos_veces():
    fd, fh = dvfs._device_host_frequency("cpu:F3", "cpu")
    assert fd == fh == pytest.approx(dvfs._relative_frequency("F3", "cpu"))
    fd, fh = dvfs._device_host_frequency("gpu:F0:F6", "gpu")
    assert fd == pytest.approx(dvfs._relative_frequency("F6", "gpu"))  # nivel del dispositivo
    assert fh == pytest.approx(dvfs._relative_frequency("F0", "cpu"))  # nivel del anfitrion


def _candidates_con_barrido_frecuencia():
    """Sigue la ley fisica t=ta+tb/f_dev+tc/f_host, E=ea+eb/f_dev+eg*f_dev+eh/f_host
    con parametros distintos por operacion, para que curve_physical tenga
    una forma real que recuperar (no solo 2 puntos por accion)."""
    rows = []
    cpu_levels = ("REF", "F0", "F2", "F4", "F6")
    gpu_levels = ("REF", "F0", "F3", "F6")
    for operation, (ta, tb, tc, ea, eb, eg, eh) in (
        ("gemm", (0.02, 0.30, 0.02, 0.01, 0.05, 0.02, 0.01)),
        ("stencil", (0.05, 0.05, 0.05, 0.03, 0.01, 0.04, 0.03)),
    ):
        for size in (128, 256, 512):
            config_id = f"{operation}_N{size}"
            scale = (size / 128.0)
            for region, cold_mult in (("cold", 3.0), ("warm", 1.0)):
                for cpu_level in cpu_levels:
                    f = dvfs._relative_frequency(cpu_level, "cpu")
                    t = (ta + tb / f + tc / f) * scale * cold_mult
                    e = (ea + eb / f + eg * f + eh / f) * scale * cold_mult
                    rows.append(_row(config_id, operation, size, "cpu", f"cpu:{cpu_level}", region, e, t))
                for host_level in gpu_levels:
                    for dev_level in gpu_levels:
                        fd = dvfs._relative_frequency(dev_level, "gpu")
                        fh = dvfs._relative_frequency(host_level, "cpu")
                        t = (ta + tb / fd + tc / fh) * scale * cold_mult * 0.4
                        e = (ea + eb / fd + eg * fd + eh / fh) * scale * cold_mult * 0.4
                        rows.append(_row(
                            config_id, operation, size, "gpu",
                            f"gpu:{host_level}:{dev_level}", region, e, t,
                        ))
    return pd.DataFrame(rows)


def test_curve_physical_esta_registrada_como_familia():
    assert "curve_physical" in dvfs.DVFS_FAMILIES


def test_curve_physical_ajusta_y_reconstruye_ref_de_forma_autoconsistente():
    frame = dvfs.build_dvfs_dataset(_candidates_con_barrido_frecuencia())
    models = dvfs.fit_cost_models(frame, "curve_physical", calibration_splits=0)
    predicted = dvfs.predict_costs(models, frame)
    ref_rows = predicted[predicted["frequency_action"] == predicted["reference_action"]]
    # En REF, costo(accion)/costo(REF) == 1 por construccion (autoconsistente,
    # no una segunda cantidad predicha) -- ver PhysicalCurveCostModel.
    np.testing.assert_allclose(ref_rows["pred_energy_j"], ref_rows["ref_energy_j"], rtol=1e-6)
    np.testing.assert_allclose(ref_rows["pred_time_s"], ref_rows["ref_time_s"], rtol=1e-6)


def test_curve_physical_predice_razonablemente_bien_la_forma_conocida():
    frame = dvfs.build_dvfs_dataset(_candidates_con_barrido_frecuencia())
    models = dvfs.fit_cost_models(frame, "curve_physical", calibration_splits=0)
    predicted = dvfs.predict_costs(models, frame)
    error_pct = 100.0 * np.abs(predicted["pred_edp_js"] / predicted["edp_js"] - 1.0)
    # Los datos siguen la forma exacta que el modelo asume; el error debe ser
    # chico (no cero: hay Ridge con regularizacion sobre pocos config_id).
    assert error_pct.median() < 15.0


def test_curve_physical_participa_en_evaluate_dvfs_sin_romper_el_contrato():
    frame = dvfs.build_dvfs_dataset(_candidates_con_barrido_frecuencia())
    results, decisions = dvfs.evaluate_dvfs(frame, families=("curve_physical",))
    assert (results["method"] == "model").any()
    assert set(decisions["family"]) == {"curve_physical"}
    assert not decisions.empty


def test_curve_physical_recorta_log_ratio_ante_extrapolacion_absurda():
    # Reproduce el caso real (cholesky_N256, gpu_ready, host F6) donde el
    # regresor de parametros extrapola un valor absurdo para un grupo fuera
    # de muestra y, dividido entre una fraccion de frecuencia chica, produce
    # un log-ratio disparatado. Fuerza el mismo mecanismo con parametros
    # de curva inventados, sin pasar por el ajuste real.
    curve = dvfs.PhysicalCurveCostModel()
    huge = 1e12

    class _FixedModel:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.full(len(x), self.value)

    curve._models = {
        "ta": _FixedModel(0.0), "tb": _FixedModel(huge), "tc": _FixedModel(0.0),
        "ea": _FixedModel(0.0), "eb": _FixedModel(huge), "eg": _FixedModel(0.0), "eh": _FixedModel(0.0),
    }
    x = pd.DataFrame([{
        "frequency_action": "gpu:F6:F6", "device": "gpu",
        "operation": "cholesky", "resource_state": "gpu_ready",
        "log10_n": 2.4, "flops_per_dispatch_analytic": 1.0,
        "log10_flops_per_dispatch": 0.0, "logical_bytes_per_dispatch": 1.0,
        "log10_logical_bytes": 0.0, "arithmetic_intensity_analytic": 1.0,
    }])
    ratio = curve.predict_log_ratio(x, "time")[0]
    assert abs(ratio) <= math.log(8.0) + 1e-9
    ratio_e = curve.predict_log_ratio(x, "energy")[0]
    assert abs(ratio_e) <= math.log(8.0) + 1e-9
