"""Target de horizonte K y baselines de la enmienda 2026-08-30-A."""
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import compact, horizon, sizes
from classifier.selector.compact import CompactDatasetError


def _candidate_row(
    config_id: str, operation: str, size: int, device: str, action_id: str,
    region: str, energy: float, time: float, *, spread: float = 0.0,
) -> dict:
    return {
        "config_id": config_id, "operation": operation, "size": size,
        "family": "matrix", "device": device, "action_id": action_id,
        "cpu_level": "REF", "gpu_level": "REF" if device == "gpu" else None,
        "region": region, "n_repetitions": 3,
        "energy_mean": energy, "energy_min": energy * (1 - spread), "energy_max": energy * (1 + spread),
        "time_mean": time, "time_min": time * (1 - spread), "time_max": time * (1 + spread),
        "edp_mean": energy * time, "edp_std": 0.0, "edp_cv_pct": 0.0,
        "time_cv_pct": 0.0, "energy_cv_pct": 0.0,
        "eligible_repetitions": True,
    }


def _ref_only_candidates(config_id="gemm_N8192", operation="gemm", size=8192, **overrides) -> pd.DataFrame:
    """Una configuracion con GPU cara en frio y barata en caliente -> cruce finito."""
    costs = {
        ("cpu", "cold"): (10.0, 10.0), ("cpu", "warm"): (4.0, 4.0),
        ("gpu", "cold"): (50.0, 50.0), ("gpu", "warm"): (1.0, 1.0),
    }
    costs.update(overrides)
    rows = [
        _candidate_row(
            config_id, operation, size, device,
            compact.CPU_REF_ACTION if device == "cpu" else compact.GPU_REF_ACTION,
            region, energy, time, spread=0.1,
        )
        for (device, region), (energy, time) in costs.items()
    ]
    return pd.DataFrame(rows)


def _two_operation_candidates() -> pd.DataFrame:
    """gemm cruza a GPU en tamanos grandes; stencil nunca cruza (GPU cara siempre)."""
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 128, 256, 512):
            config_id = f"{operation}_N{size}"
            cpu_warm = (size / 64.0) ** 3
            gpu_warm = 0.5 if operation == "gemm" else 4000.0
            for device, warm, cold in (
                ("cpu", cpu_warm, cpu_warm * 2),
                ("gpu", gpu_warm, gpu_warm * 5),
            ):
                action = compact.CPU_REF_ACTION if device == "cpu" else compact.GPU_REF_ACTION
                rows.append(_candidate_row(config_id, operation, size, device, action, "cold", cold, cold))
                rows.append(_candidate_row(config_id, operation, size, device, action, "warm", warm, warm))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# edp_total_state / consistencia con K = 1
# --------------------------------------------------------------------------


def test_dispositivo_inicializado_paga_solo_costo_caliente():
    costs = {"e_cold": 10.0, "t_cold": 10.0, "e_warm": 4.0, "t_warm": 4.0}
    e, t, edp = horizon.edp_total_state(costs, device="cpu", state="cpu_ready", k=3)
    assert e == pytest.approx(12.0) and t == pytest.approx(12.0) and edp == pytest.approx(144.0)


def test_dispositivo_no_inicializado_paga_arranque_una_vez():
    costs = {"e_cold": 10.0, "t_cold": 10.0, "e_warm": 4.0, "t_warm": 4.0}
    e, t, edp = horizon.edp_total_state(costs, device="gpu", state="cpu_ready", k=3)
    # E = 10 + 2*4 = 18; T = 10 + 2*4 = 18
    assert e == pytest.approx(18.0) and t == pytest.approx(18.0) and edp == pytest.approx(324.0)


def test_k_menor_que_uno_es_error():
    with pytest.raises(CompactDatasetError, match="K debe ser"):
        horizon.edp_total_state({"e_cold": 1, "t_cold": 1, "e_warm": 1, "t_warm": 1},
                                 device="cpu", state="none_ready", k=0)


def test_k_uno_reproduce_exactamente_el_target_compacto():
    candidates = _two_operation_candidates()
    compacto = compact.build_compact_dataset(candidates).set_index(["config_id", "resource_state"])
    horizonte = horizon.build_horizon_dataset(candidates, k_grid=(1,)).set_index(["config_id", "resource_state"])
    common = compacto.index.intersection(horizonte.index)
    assert len(common) == len(compacto)
    for key in common:
        assert horizonte.loc[key, "y_log_edp_ratio_k"] == pytest.approx(
            compacto.loc[key, "y_log_edp_ratio"]
        )
        assert horizonte.loc[key, "device_label"] == compacto.loc[key, "device_label"]


# --------------------------------------------------------------------------
# build_horizon_dataset
# --------------------------------------------------------------------------


def test_una_fila_por_config_estado_y_k():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates)
    assert len(frame) == 8 * 3 * len(horizon.K_GRID)
    assert frame.groupby(["config_id", "resource_state", "k"]).size().max() == 1


def test_costos_crudos_estan_prohibidos_como_entrada():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates)
    assert any(column.startswith("cost_") for column in frame.columns)
    # Ninguna columna cost_* -- los insumos crudos de `oracle_k` y de la tabla
    # empirica -- puede colarse en la lista auditada de caracteristicas.
    columns = horizon.horizon_feature_columns(frame)
    assert not any(column.startswith("cost_") for column in columns)
    # Guarda defensiva: si alguna columna cost_* entrara a HORIZON_FEATURES en
    # el futuro, el chequeo bloqueante debe seguir rechazandola.
    import classifier.selector.horizon as horizon_module
    original = horizon_module.HORIZON_FEATURES
    try:
        horizon_module.HORIZON_FEATURES = (*original, "cost_cpu_e_cold")
        with pytest.raises(CompactDatasetError, match="costos medidos"):
            horizon.horizon_feature_columns(frame)
    finally:
        horizon_module.HORIZON_FEATURES = original


def test_features_de_horizonte_incluyen_k_y_no_tienen_fuga():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates)
    columns = horizon.horizon_feature_columns(frame)
    assert "k" in columns and "log10_k" in columns and "resource_state" in columns
    assert compact.leaking_columns(columns) == []


def test_grid_de_k_invalida_o_vacia_es_error():
    candidates = _ref_only_candidates()
    with pytest.raises(CompactDatasetError, match="rejilla de K"):
        horizon.build_horizon_dataset(candidates, k_grid=())
    with pytest.raises(CompactDatasetError, match="rejilla de K"):
        horizon.build_horizon_dataset(candidates, k_grid=(0, 1))


# --------------------------------------------------------------------------
# switch_k_for_state / mapa de cambio por estado
# --------------------------------------------------------------------------


def test_switch_k_es_finito_solo_si_gpu_gana_en_caliente():
    cpu = {"e_cold": 1.0, "t_cold": 1.0, "e_warm": 2.0, "t_warm": 2.0}
    gpu_gana_caliente = {"e_cold": 100.0, "t_cold": 100.0, "e_warm": 1.0, "t_warm": 1.0}
    assert math.isfinite(horizon.switch_k_for_state(cpu, gpu_gana_caliente, "gpu_ready"))
    gpu_pierde_siempre = {"e_cold": 200.0, "t_cold": 200.0, "e_warm": 3.0, "t_warm": 3.0}
    assert horizon.switch_k_for_state(cpu, gpu_pierde_siempre, "gpu_ready") == float("inf")


def test_los_tres_estados_convergen_al_mismo_conjunto_asintotico():
    candidates = _two_operation_candidates()
    switch_map = horizon.state_switch_map(candidates)
    migration = horizon.horizon_migration_summary(switch_map)
    asintoticos = set(migration["asymptotic_gpu_wins"])
    assert len(asintoticos) == 1
    # gemm (4 tamanos) gana en caliente en este catalogo sintetico; stencil no.
    assert asintoticos == {4}


def test_gemm_migra_a_gpu_y_stencil_nunca_migra():
    candidates = _two_operation_candidates()
    switch_map = horizon.state_switch_map(candidates)
    gemm = switch_map[(switch_map["operation"] == "gemm") & (switch_map["resource_state"] == "cpu_ready")]
    assert (gemm["switches_within_horizon"] == 1).any()
    stencil = switch_map[(switch_map["operation"] == "stencil") & (switch_map["resource_state"] == "gpu_ready")]
    assert (stencil["switches_within_horizon"] == 0).all()


def test_banda_de_switch_k_encierra_el_valor_central():
    candidates = _two_operation_candidates()
    switch_map = horizon.state_switch_map(candidates)
    finite = switch_map[np.isfinite(switch_map["switch_k"])]
    assert not finite.empty
    assert (finite["switch_k_low"] <= finite["switch_k"]).all()
    assert (finite["switch_k"] <= finite["switch_k_high"]).all()


# --------------------------------------------------------------------------
# Baselines de horizonte (seccion 12.4)
# --------------------------------------------------------------------------


def test_stay_on_ready_device_k_ignora_el_horizonte():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates, k_grid=(1, 1000))
    devices = sizes.BASELINES["stay_on_ready_device_k"](frame)(frame)
    elegido = pd.Series(devices, index=frame.index)
    assert (elegido[frame["resource_state"] == "gpu_ready"] == "gpu").all()
    assert (elegido[frame["resource_state"] == "none_ready"] == "cpu").all()


def test_oracle_k_alcanza_el_edp_minimo_del_horizonte():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates, k_grid=(1, 30, 1000))
    for _, group in frame.groupby(["resource_state", "k"], observed=True):
        devices = sizes.BASELINES["oracle_k"](group)(group)
        metrics = sizes.evaluate_devices(group, devices)
        assert metrics["edp_sum_ratio_vs_oracle"] == pytest.approx(1.0)


def test_k_break_even_table_train_se_degrada_sin_columnas_de_costo():
    # Sobre el dataset compacto (K=1 implicito) no hay cost_cpu_e_cold/etc.
    candidates = _two_operation_candidates()
    compacto = compact.build_compact_dataset(candidates)
    gpu_ready = compacto[compacto["resource_state"] == "gpu_ready"]
    predict = sizes.BASELINES["k_break_even_table_train"](gpu_ready)
    devices = predict(gpu_ready)
    esperado = sizes.BASELINES["stay_on_ready_device_k"](gpu_ready)(gpu_ready)
    assert list(devices) == list(esperado)


def test_k_break_even_table_train_predice_migracion_a_gpu_en_horizontes_grandes():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates, k_grid=horizon.K_GRID)
    cpu_ready = frame[frame["resource_state"] == "cpu_ready"]
    # Entrenar solo con tamanos pequenos de gemm (donde K=1 aun no cruza) y
    # aplicar por interpolacion al tamano de prueba mas grande.
    train = cpu_ready[cpu_ready["size"] <= 256]
    test = cpu_ready[cpu_ready["size"] == 512]
    predict = sizes.BASELINES["k_break_even_table_train"](train)
    for k in (1, 1000):
        subset = test[test["k"] == k]
        devices = predict(subset)
        # Con K=1000 el arranque de GPU esta amortizado de sobra en gemm.
        if k == 1000:
            assert "gpu" in set(devices[subset["operation"] == "gemm"])


def test_k_break_even_table_train_es_seguro_cuando_train_esta_vacio():
    candidates = _two_operation_candidates()
    frame = horizon.build_horizon_dataset(candidates)
    vacio = frame.iloc[0:0]
    predict = sizes.BASELINES["k_break_even_table_train"](vacio)
    devices = predict(frame.head(3))
    assert len(devices) == 3
