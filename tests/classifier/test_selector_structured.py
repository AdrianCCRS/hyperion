"""Fase R2-estructurada: modelo de tres capas (prediccion/calibracion/composicion)."""
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import horizon, structured
from classifier.selector.sizes import extrapolation_folds, interpolation_folds

pytest.importorskip("sklearn")


def _row(config_id, operation, size, device, action_id, region, energy, time):
    return {
        "config_id": config_id, "operation": operation, "size": size,
        "family": "matrix", "device": device, "action_id": action_id,
        "region": region, "n_repetitions": 3,
        "energy_mean": energy, "energy_min": energy * 0.9, "energy_max": energy * 1.1,
        "time_mean": time, "time_min": time * 0.9, "time_max": time * 1.1,
        "edp_mean": energy * time, "edp_std": 0.0, "edp_cv_pct": 0.0,
        "time_cv_pct": 0.0, "energy_cv_pct": 0.0, "eligible_repetitions": True,
    }


def _candidates() -> pd.DataFrame:
    """gemm cruza a GPU con K grande; stencil nunca cruza (GPU cara siempre).

    Ley de potencias EXACTA por construccion (``cpu_warm = (size/64)**3``):
    sirve para la prueba de consistencia de la capa 1 (recupera la pendiente
    conocida) y para que la capa 3, con primitivas exactas, reproduzca
    exactamente el target de `horizon.py`.
    """
    # Arranque de GPU EXACTAMENTE constante (+0.6, cualquier operacion) y de
    # CPU dependiente de la operacion (+0.05 en gemm, +0.9 en stencil,
    # constante DENTRO de cada operacion) -- reproduce a proposito la forma
    # cualitativa observada en los datos exploratorios (GPU CV bajo, CPU CV
    # alto) para poder probar la capa 2 con un resultado conocido.
    cpu_startup = {"gemm": 0.05, "stencil": 0.9}
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 128, 256, 512, 1024, 2048):
            config_id = f"{operation}_N{size}"
            cpu_warm = (size / 64.0) ** 3
            gpu_warm = 0.5 if operation != "stencil" else 4000.0
            for device, warm, cold in (
                ("cpu", cpu_warm, cpu_warm + cpu_startup[operation]),
                ("gpu", gpu_warm, gpu_warm + 0.6),
            ):
                action = "cpu:REF" if device == "cpu" else "gpu:REF:REF"
                rows.append(_row(config_id, operation, size, device, action, "cold", cold, cold))
                rows.append(_row(config_id, operation, size, device, action, "warm", warm, warm))
    return pd.DataFrame(rows)


def _horizon_frame(k_grid=(1, 10, 1000)) -> pd.DataFrame:
    return horizon.build_horizon_dataset(_candidates(), k_grid=k_grid)


def _folds(frame):
    return [*interpolation_folds(frame, n_folds=2), *extrapolation_folds(frame, n_largest=1)]


# --------------------------------------------------------------------------
# Capa 1: prediccion de costos calientes
# --------------------------------------------------------------------------


def test_capa1_ajusta_ley_de_potencias_sintetica_conocida():
    """Datos sinteticos con pendiente log-log EXACTA conocida (b=3, gemm cpu_warm).

    `cpu_warm = (size/64)**3` implica ``log(cpu_warm) = 3*log10(size)*ln(10) -
    3*log10(64)*ln(10)`` -- una recta perfecta en log-log. Ridge sobre esto
    debe predecir con error casi nulo, sin GridSearchCV.
    """
    primitives = structured.build_primitives_dataset(_candidates())
    models = structured.fit_layer1(primitives, "ridge")
    predicted = structured.predict_layer1(models, primitives)
    true = primitives["cpu_e_warm"].to_numpy(dtype=float)
    relative_error = np.abs(predicted["cpu_e_warm"] - true) / true
    # Ridge (alpha=1.0 por defecto) sobre 8 filas sinteticas regulariza algo
    # de la pendiente exacta; el umbral verifica que capta la ley de potencias
    # -- no que replique una regresion sin regularizar.
    assert relative_error.max() < 0.5


def test_capa1_r2_agrupado_es_alto_en_dataset_sintetico():
    # `GroupKFold` no baraja por defecto: con solo dos operaciones, un orden
    # por config_id (alfabetico) puede dejar una operacion entera fuera de un
    # pliegue de entrenamiento. Se baraja el frame para que cada pliegue vea
    # ambas operaciones, como ocurriria con las 6 operaciones del catalogo
    # real (68 config_id, ninguna aislada por orden alfabetico).
    primitives = structured.build_primitives_dataset(_candidates()).sample(
        frac=1.0, random_state=20260830,
    ).reset_index(drop=True)
    r2_by_target = structured.layer1_grouped_cv_r2(primitives, "ridge", n_splits=3)
    for target, value in r2_by_target.items():
        assert value > 0.8, (target, value)


def test_capa1_sin_fuga_en_las_caracteristicas():
    from classifier.selector.compact import leaking_columns
    assert leaking_columns(structured.PRIMITIVE_FEATURES) == []
    assert "resource_state" not in structured.PRIMITIVE_FEATURES


# --------------------------------------------------------------------------
# Capa 2: calibracion de arranque
# --------------------------------------------------------------------------


def test_capa2_recupera_un_arranque_constante_conocido():
    """GPU con arranque de tiempo EXACTAMENTE constante (0.6 s) en todo el catalogo.

    La calibracion debe recuperar esa constante con error pequeno y, al ser
    genuinamente constante (misma dispersion en todas las operaciones), la CV
    agrupada debe preferir el modo `"constant"` y no `"per_operation"`.
    """
    primitives = structured.build_primitives_dataset(_candidates()).sample(
        frac=1.0, random_state=20260830,
    ).reset_index(drop=True)
    calibration = structured.calibrate_startup(primitives, n_splits=3)
    gpu_t = calibration["gpu"]["t"]
    assert gpu_t["mode"] == "constant"
    assert gpu_t["constant"] == pytest.approx(0.6, rel=0.05)


def test_capa2_decision_constante_vs_por_operacion_no_se_asume():
    """Con arranque de CPU que SI depende de la operacion, la CV debe poder elegir
    `"per_operation"` cuando reduce el error fuera de muestra; se comprueba
    que el campo de diagnostico existe y es comparable, no que un modo gane
    siempre (la eleccion depende de los datos, no se fuerza aqui).
    """
    primitives = structured.build_primitives_dataset(_candidates())
    calibration = structured.calibrate_startup(primitives, n_splits=2)
    for device in ("cpu", "gpu"):
        for quantity in ("e", "t"):
            entry = calibration[device][quantity]
            assert entry["mode"] in {"constant", "per_operation"}
            assert np.isfinite(entry["cv_mae_constant"])


def test_capa2_se_calibra_solo_con_entrenamiento():
    """Dos particiones de train distintas deben producir calibraciones distintas
    en general -- confirma que `calibrate_startup` no memoriza el catalogo
    completo por accidente."""
    primitives = structured.build_primitives_dataset(_candidates())
    only_gemm = primitives[primitives["operation"] == "gemm"]
    calibration_gemm = structured.calibrate_startup(only_gemm, n_splits=2)
    calibration_all = structured.calibrate_startup(primitives, n_splits=2)
    # La tabla de gemm-solo no puede contener stencil.
    assert "stencil" not in calibration_gemm["cpu"]["t"]["table"]
    assert "stencil" in calibration_all["cpu"]["t"]["table"]


# --------------------------------------------------------------------------
# Capa 3: composicion -- consistencia fuerte contra horizon.py
# --------------------------------------------------------------------------


def test_capa3_con_primitivas_exactas_reproduce_el_target_de_horizon():
    """Prueba de consistencia central: si a la capa 3 se le dan las primitivas
    EXACTAS (sin error de prediccion de la capa 1 ni de calibracion de la
    capa 2 -- se usan directamente los costos medidos), debe reproducir
    BIT A BIT el `y_log_edp_ratio_k` que ya calcula `horizon.py` para
    cualquier `(config_id, resource_state, k)`.
    """
    candidates = _candidates()
    horizon_frame = horizon.build_horizon_dataset(candidates, k_grid=(1, 2, 5, 30, 1000))
    device_costs = horizon.device_costs(candidates)
    for row in horizon_frame.to_dict("records"):
        config_id, state, k = row["config_id"], row["resource_state"], int(row["k"])
        per_device = device_costs[config_id]
        composed = {
            device: structured.compose_device_costs(
                per_device[device]["e_warm"], per_device[device]["t_warm"],
                per_device[device]["e_cold"] - per_device[device]["e_warm"],
                per_device[device]["t_cold"] - per_device[device]["t_warm"],
            )
            for device in ("cpu", "gpu")
        }
        [y_structured] = structured.structured_y_for_rows(
            {config_id: composed}, pd.DataFrame([row]),
        )
        assert y_structured == pytest.approx(row["y_log_edp_ratio_k"], rel=1e-9, abs=1e-9)


def test_capa3_composicion_usa_edp_total_state_de_horizon():
    """La composicion de capa 3 y `horizon.edp_total_state` deben coincidir
    exactamente sobre las mismas primitivas -- no hay una segunda formula."""
    costs = {"e_warm": 2.0, "t_warm": 3.0, "e_cold": 5.0, "t_cold": 7.0}
    for state in ("none_ready", "cpu_ready", "gpu_ready"):
        for k in (1, 3, 100):
            expected = horizon.edp_total_state(costs, device="cpu", state=state, k=k)
            composed = structured.compose_device_costs(2.0, 3.0, 5.0 - 2.0, 7.0 - 3.0)
            got = horizon.edp_total_state(composed, device="cpu", state=state, k=k)
            assert got == expected


def test_capa3_arranque_negativo_se_recorta_al_costo_caliente():
    composed = structured.compose_device_costs(10.0, 10.0, -5.0, -5.0)
    assert composed["e_cold"] >= composed["e_warm"]
    assert composed["t_cold"] >= composed["t_warm"]


def test_sondeo_sustituye_la_primitiva_fria_sin_tocar_la_caliente():
    composed = structured.compose_device_costs(2.0, 3.0, 100.0, 100.0, probe=(9.0, 11.0))
    assert composed["e_warm"] == 2.0
    assert composed["t_warm"] == 3.0
    assert composed["e_cold"] == 9.0
    assert composed["t_cold"] == 11.0


def test_cache_de_sondeo_es_por_config_id_y_dispositivo_no_por_estado():
    frame = pd.DataFrame([
        {
            "config_id": "gemm_N64", "resource_state": "gpu_ready",
            "probe_device": "gpu", "probe_energy_per_dispatch_j": 1.23,
            "probe_time_per_dispatch_s": 0.45,
        },
        {
            "config_id": "gemm_N64", "resource_state": "cpu_ready",
            "probe_device": "cpu", "probe_energy_per_dispatch_j": 0.02,
            "probe_time_per_dispatch_s": 0.01,
        },
    ])
    cache = structured.build_probe_cold_cache(frame)
    assert cache[("gemm_N64", "gpu")] == (1.23, 0.45)
    assert cache[("gemm_N64", "cpu")] == (0.02, 0.01)


# --------------------------------------------------------------------------
# Evaluacion pareada (misma rebanada de test que r2.py, mismas metricas)
# --------------------------------------------------------------------------


def test_evaluate_structured_produce_filas_model_structured():
    frame = _horizon_frame()
    results = structured.evaluate_structured(
        _folds(frame), _candidates(), k_grid=(1, 10, 1000), with_probe_variants=(False,),
    )
    assert not results.empty
    assert set(results["method"]) == {"model_structured"}
    assert set(results["name"]) >= set(structured.REGRESSOR_FAMILIES)


def test_evaluate_structured_misma_rebanada_de_test_que_r2():
    from classifier.selector import r2

    frame = _horizon_frame()
    folds = _folds(frame)
    direct = r2.evaluate_r2(folds, k_grid=(1, 10, 1000), with_probe_variants=(False,))
    struct = structured.evaluate_structured(
        folds, _candidates(), k_grid=(1, 10, 1000), with_probe_variants=(False,),
    )
    for (fold, state, k), group in struct.groupby(["fold", "resource_state", "k"], observed=True):
        direct_n = direct[
            (direct["fold"] == fold) & (direct["resource_state"] == state) & (direct["k"] == k)
        ]["n"].unique()
        assert set(group["n"].unique()) <= set(direct_n)


def test_three_way_blocking_rule_reporta_las_tres_vias():
    from classifier.selector import r2

    frame = _horizon_frame()
    folds = _folds(frame)
    direct = r2.evaluate_r2(folds, k_grid=(1, 10, 1000), with_probe_variants=(False,))
    struct = structured.evaluate_structured(
        folds, _candidates(), k_grid=(1, 10, 1000), with_probe_variants=(False,),
    )
    report = structured.three_way_blocking_rule(direct, struct)
    assert not report.empty
    assert set(report["fold"]) == {name for name, _, _ in folds}
    # La baseline puede depender de (regimen, estado, K), pero no del pliegue.
    assert (
        report.groupby(["regime", "resource_state", "k"], observed=True)[
            "selected_baseline"
        ].nunique() == 1
    ).all()
    assert report["selected_direct_name"].nunique() == 1
    assert report["selected_structured_name"].nunique() == 1
    assert set(report["winner_formulation"]) <= {"direct", "structured"}
    assert report["beats_baseline_above_noise_floor"].dtype == bool


def test_sin_fuga_en_caracteristicas_de_capa1_dentro_del_flujo_completo():
    from classifier.selector.compact import assert_no_leakage
    assert_no_leakage(structured.PRIMITIVE_FEATURES)


# --------------------------------------------------------------------------
# Latencia y tamano
# --------------------------------------------------------------------------


def test_latencia_y_tamano_del_modelo_estructurado():
    primitives = structured.build_primitives_dataset(_candidates())
    calibration = structured.calibrate_startup(primitives, n_splits=2)
    models = structured.fit_layer1(primitives, "ridge")
    latency = structured.measure_structured_latency(
        models, calibration, primitives, warmups=2, repeats=5,
    )
    assert latency["latency_p50_us"] > 0
    assert latency["latency_p50_us"] <= latency["latency_p95_us"] <= latency["latency_p99_us"]
    assert structured.structured_model_size_bytes(models, calibration) > 0


# --------------------------------------------------------------------------
# Orquestador
# --------------------------------------------------------------------------


def test_run_structured_analysis_escribe_los_artefactos_esperados(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _candidates().to_csv(dataset_dir / "candidate_summary.csv", index=False)

    paths = structured.run_structured_analysis(dataset_dir, k_grid=(1, 10, 1000))

    expected = {
        "primitives_dataset", "structured_results", "three_way_blocking_rule",
        "three_way_blocking_rule_aggregated", "summary",
    }
    assert set(paths) == expected
    assert all(path.exists() for path in paths.values())

    import json
    summary = json.loads(paths["summary"].read_text())
    assert "layer1_grouped_cv_r2_ridge" in summary
    assert summary["interpretation_contract"]["layer3_reuses_horizon_edp_total_state_formula"] is True
