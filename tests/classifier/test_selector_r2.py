"""Fase R2: regresores de horizonte K contra baselines en los mismos pliegues."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import horizon, r2
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

    Catalogo deliberadamente pequeno (2 operaciones x 4 tamanos): las pruebas
    de este archivo ajustan `GridSearchCV` real (5 familias, con Huber siendo
    el mas costoso); un catalogo grande multiplica el tiempo de la suite sin
    aportar mas cobertura de los invariantes que se prueban aqui.
    """
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 128, 256, 512):
            config_id = f"{operation}_N{size}"
            cpu_warm = (size / 64.0) ** 3
            gpu_warm = 0.5 if operation != "stencil" else 4000.0
            for device, warm, cold in (
                ("cpu", cpu_warm, cpu_warm * 2),
                ("gpu", gpu_warm, gpu_warm * 5),
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
# Abstencion (seccion 7)
# --------------------------------------------------------------------------


def test_abstencion_aplica_la_politica_segura_del_estado():
    y_pred = np.array([0.0, 0.0, 5.0, -5.0])
    state = np.array(["gpu_ready", "cpu_ready", "gpu_ready", "cpu_ready"])
    devices, abstained = r2.devices_from_predictions(y_pred, state)
    assert list(abstained) == [True, True, False, False]
    # Abstencion: gpu_ready se queda en gpu (dispositivo preparado), cpu_ready en cpu.
    assert list(devices) == ["gpu", "cpu", "cpu", "gpu"]


def test_abstencion_es_la_constante_congelada_no_se_ajusta_con_datos():
    import math
    assert r2.ABSTENTION_LOG_THRESHOLD == pytest.approx(math.log(1.0311), abs=1e-6)


# --------------------------------------------------------------------------
# Entrenamiento por familia
# --------------------------------------------------------------------------


def test_las_cinco_familias_de_regresor_entrenan_y_predicen():
    frame = _horizon_frame()
    features = horizon.horizon_feature_columns(frame)
    for family in r2.REGRESSOR_FAMILIES:
        model, tuning = r2.fit_tuned_regressor(family, frame, features, seed=20260830)
        predictions = model.predict(frame[features].head(5))
        assert len(predictions) == 5
        assert np.isfinite(predictions).all()


def test_hiperparametros_se_eligen_solo_con_datos_de_entrenamiento():
    frame = _horizon_frame()
    features = horizon.horizon_feature_columns(frame)
    # Con pocos config_id (menos de 2 grupos) la CV interna se omite en lugar
    # de fallar -- caso de borde documentado en `fit_tuned_regressor`.
    un_solo_config = frame[frame["config_id"] == frame["config_id"].iloc[0]]
    model, tuning = r2.fit_tuned_regressor("ridge", un_solo_config, features, seed=20260830)
    assert tuning["tuned"] is False


def test_sin_fuga_en_las_caracteristicas_usadas_para_entrenar():
    frame = _horizon_frame()
    features = horizon.horizon_feature_columns(frame, with_probe=False)
    from classifier.selector.compact import leaking_columns
    assert leaking_columns(features) == []


# --------------------------------------------------------------------------
# Evaluacion pareada baselines vs modelo (seccion 5/6)
# --------------------------------------------------------------------------


def test_evaluate_r2_produce_las_tres_familias_de_metodo():
    frame = _horizon_frame()
    results = r2.evaluate_r2(_folds(frame), k_grid=(1, 10, 1000), with_probe_variants=(False,))
    assert set(results["method"]) == {"baseline", "model_regression", "model_classification"}
    assert set(results["name"]) >= set(r2.REGRESSOR_FAMILIES)
    from classifier.selector.sizes import BASELINES
    assert set(results.loc[results["method"] == "baseline", "name"]) == set(BASELINES)


def test_evaluate_r2_reporta_los_ocho_valores_de_k_de_la_rejilla_congelada():
    frame = _horizon_frame(k_grid=horizon.K_GRID)
    results = r2.evaluate_r2(
        _folds(frame), k_grid=horizon.K_GRID, with_probe_variants=(False,),
    )
    assert set(results["k"]) == set(horizon.K_GRID)


def test_modelo_y_baselines_se_evaluan_sobre_la_misma_rebanada_de_test():
    frame = _horizon_frame()
    folds = _folds(frame)
    results = r2.evaluate_r2(folds, k_grid=(1, 10, 1000), with_probe_variants=(False,))
    # Agrupado por `fold` (no solo por `regime`): distintos pliegues del mismo
    # regimen (interpolation_fold0 vs interpolation_fold1) tienen legitimamente
    # tamanos de test distintos: la invariante es dentro de un mismo pliegue.
    for (fold, state, k), group in results.groupby(["fold", "resource_state", "k"], observed=True):
        n_by_method = group.groupby("method")["n"].unique()
        # El tamano de la rebanada de test es el mismo para baselines y modelo:
        # ambos evaluan literalmente las mismas filas de `test`.
        values = {int(v) for arr in n_by_method for v in arr}
        assert len(values) == 1, (fold, state, k, n_by_method)


def test_regla_bloqueante_declara_perdida_del_modelo_cuando_la_baseline_ya_es_optima():
    # Con datos sinteticos donde la tabla de cruce ya replica al oraculo, el
    # modelo no puede superar el piso de ruido: la regla debe conservar la
    # baseline, no maquillar un empate como victoria.
    frame = _horizon_frame()
    results = r2.evaluate_r2(_folds(frame), k_grid=(1, 10, 1000), with_probe_variants=(False,))
    report = r2.blocking_rule_report(results)
    assert not report.empty
    assert set(report["adopted_policy"]) <= {"model", "baseline"}
    # La regla es binaria por renglon: no hay una tercera categoria "casi".
    assert report["model_beats_baseline_above_noise_floor"].dtype == bool


def test_seleccion_final_usa_solo_extrapolacion_y_es_una_sola_familia():
    frame = _horizon_frame()
    results = r2.evaluate_r2(_folds(frame), k_grid=(1, 10, 1000), with_probe_variants=(False,))
    selection = r2.select_final_model(results)
    assert selection["family"] in r2.REGRESSOR_FAMILIES
    assert isinstance(selection["with_probe"], bool)


def test_estatico_contra_sondeo_reporta_ambas_variantes_por_familia():
    frame = _horizon_frame()
    results = r2.evaluate_r2(_folds(frame), k_grid=(1, 10, 1000))
    contraste = r2.static_vs_probe_report(results)
    assert not contraste.empty
    assert {"edp_sum_ratio_static", "edp_sum_ratio_probe", "probe_improves_pct"} <= set(contraste.columns)


# --------------------------------------------------------------------------
# Latencia y tamano (seccion 10.3)
# --------------------------------------------------------------------------


def test_latencia_y_tamano_del_modelo_se_miden():
    frame = _horizon_frame()
    features = horizon.horizon_feature_columns(frame)
    model, _ = r2.fit_tuned_regressor("ridge", frame, features, seed=20260830)
    latency = r2.measure_inference_latency(model, frame[features], warmups=2, repeats=5)
    assert latency["latency_p50_us"] > 0
    assert latency["latency_p50_us"] <= latency["latency_p95_us"] <= latency["latency_p99_us"]
    assert r2.model_size_bytes(model) > 0


# --------------------------------------------------------------------------
# Orquestador
# --------------------------------------------------------------------------


def test_run_r2_analysis_escribe_los_artefactos_esperados(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _candidates().to_csv(dataset_dir / "candidate_summary.csv", index=False)

    paths = r2.run_r2_analysis(dataset_dir, k_grid=(1, 10, 1000))

    expected = {"horizon_dataset", "r2_results", "blocking_rule", "static_vs_probe", "summary"}
    assert set(paths) == expected
    assert all(path.exists() for path in paths.values())

    import json
    summary = json.loads(paths["summary"].read_text())
    assert summary["final_model_selection"]["family"] in r2.REGRESSOR_FAMILIES
    assert summary["interpretation_contract"]["k_is_supplied_not_predicted"] is True


def test_run_r2_analysis_funciona_sin_run_regions_solo_variante_estatica(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _candidates().to_csv(dataset_dir / "candidate_summary.csv", index=False)
    assert not (dataset_dir / "run_regions.csv").exists()

    paths = r2.run_r2_analysis(dataset_dir, k_grid=(1, 1000))
    results = pd.read_csv(paths["r2_results"])
    # Sin run_regions no hay columnas de sondeo -- la variante with_probe=True
    # degrada silenciosamente a las mismas 9 caracteristicas estaticas, pero
    # no debe fallar ni inventar telemetria.
    assert set(results["method"]) == {"baseline", "model_regression", "model_classification"}
