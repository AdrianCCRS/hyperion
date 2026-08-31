"""Dataset compacto, particiones por tamano, baselines y data card (Fase R1)."""
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import compact, datacard, sizes
from classifier.selector.compact import CompactDatasetError


def _candidate_row(
    config_id: str, operation: str, size: int, device: str, action_id: str,
    region: str, energy: float, time: float, *, spread: float = 0.0,
) -> dict:
    """Fila de `candidate_summary` sintetica y coherente (edp = energia*tiempo)."""
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


def _ref_only_candidates(**overrides) -> pd.DataFrame:
    """Una configuracion medida SOLO a REF, como las campanas `*_big_ref_*`.

    Con estas cuatro filas -- dos acciones x dos regiones -- el dataset
    compacto tiene que funcionar, aunque el catalogo base tenga 40 acciones.
    """
    costs = {
        ("cpu", "cold"): (10.0, 10.0), ("cpu", "warm"): (4.0, 4.0),
        ("gpu", "cold"): (50.0, 50.0), ("gpu", "warm"): (1.0, 1.0),
    }
    costs.update(overrides)
    rows = [
        _candidate_row(
            "gemm_N8192", "gemm", 8192, device,
            compact.CPU_REF_ACTION if device == "cpu" else compact.GPU_REF_ACTION,
            region, energy, time, spread=0.1,
        )
        for (device, region), (energy, time) in costs.items()
    ]
    return pd.DataFrame(rows)


def _two_operation_candidates() -> pd.DataFrame:
    """Catalogo pequeno: dos operaciones x cuatro tamanos, con cruce por tamano.

    La GPU gana en los tamanos grandes y pierde en los pequenos, que es la
    estructura que las baselines de umbral deben poder capturar.
    """
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 128, 256, 512):
            config_id = f"{operation}_N{size}"
            # CPU escala con size^3; GPU tiene un costo casi fijo y alto.
            cpu_warm = (size / 64.0) ** 3
            gpu_warm = 40.0 if operation == "gemm" else 4000.0
            for device, warm, cold in (
                ("cpu", cpu_warm, cpu_warm * 2),
                ("gpu", gpu_warm, gpu_warm * 5),
            ):
                action = compact.CPU_REF_ACTION if device == "cpu" else compact.GPU_REF_ACTION
                rows.append(_candidate_row(config_id, operation, size, device, action, "cold",
                                           cold, cold, spread=0.05))
                rows.append(_candidate_row(config_id, operation, size, device, action, "warm",
                                           warm, warm, spread=0.05))
                # Una accion no-REF por dispositivo, mas barata: da headroom DVFS.
                other = "cpu:F3" if device == "cpu" else "gpu:REF:F3"
                rows.append(_candidate_row(config_id, operation, size, device, other, "cold",
                                           cold * 0.9, cold * 0.9, spread=0.05))
                rows.append(_candidate_row(config_id, operation, size, device, other, "warm",
                                           warm * 0.9, warm * 0.9, spread=0.05))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Dataset compacto
# --------------------------------------------------------------------------


def test_config_medida_solo_a_ref_no_se_descarta_en_silencio():
    # Regresion del bug central: `dataset._complete_candidate_slice` exige las
    # 40 acciones y descartaria esta configuracion sin avisar.
    candidates = _ref_only_candidates()
    assert compact.ref_configurations(candidates) == ["gemm_N8192"]
    frame = compact.build_compact_dataset(candidates)
    assert set(frame["resource_state"]) == set(compact.RESOURCE_STATES)
    assert len(frame) == 3


def test_target_es_el_log_de_la_razon_de_edp_en_la_region_de_cada_estado():
    frame = compact.build_compact_dataset(_ref_only_candidates()).set_index("resource_state")

    # none_ready: los dos frios -> log((50*50) / (10*10)).
    assert frame.loc["none_ready", "y_log_edp_ratio"] == pytest.approx(math.log(2500.0 / 100.0))
    assert frame.loc["none_ready", "device_label"] == "cpu"
    # gpu_ready: GPU caliente contra CPU fria -> log((1*1) / (10*10)).
    assert frame.loc["gpu_ready", "y_log_edp_ratio"] == pytest.approx(math.log(1.0 / 100.0))
    assert frame.loc["gpu_ready", "device_label"] == "gpu"
    # cpu_ready: CPU caliente contra GPU fria -> log((50*50) / (4*4)).
    assert frame.loc["cpu_ready", "y_log_edp_ratio"] == pytest.approx(math.log(2500.0 / 16.0))
    assert frame.loc["cpu_ready", "device_label"] == "cpu"


def test_una_fila_por_config_id_y_estado_sin_multiplicar_por_accion():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    # 8 configuraciones x 3 estados; las 4 acciones por config NO inflan filas.
    assert len(frame) == 24
    assert frame.groupby(["config_id", "resource_state"]).size().max() == 1


def test_edp_no_positivo_es_error_explicito_y_no_un_log_de_nan():
    candidates = _ref_only_candidates()
    candidates.loc[candidates["device"] == "gpu", "edp_mean"] = 0.0
    with pytest.raises(CompactDatasetError, match="EDP REF no positivo"):
        compact.build_compact_dataset(candidates)


def test_configuracion_sin_las_dos_acciones_ref_no_entra():
    candidates = _ref_only_candidates()
    candidates = candidates[candidates["device"] != "gpu"]
    assert compact.ref_configurations(candidates) == []
    with pytest.raises(CompactDatasetError, match="ninguna configuracion"):
        compact.build_compact_dataset(candidates)


# --------------------------------------------------------------------------
# Prevencion de fuga (seccion 7.3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("column", [
    "cpu_ref_edp_js", "gpu_ref_edp_js", "oracle_ref_edp_js", "device_margin_pct",
    "device_label", "y_log_edp_ratio", "is_optimal", "action_id", "run_id",
    "repetition", "margin_edp_pct",
])
def test_columnas_prohibidas_se_detectan_como_fuga(column):
    assert compact.leaking_columns([column]) == [column]
    with pytest.raises(CompactDatasetError, match="fuga"):
        compact.assert_no_leakage([column])


def test_las_features_declaradas_no_contienen_ninguna_columna_con_fuga():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    columns = compact.feature_columns(frame)
    assert compact.leaking_columns(columns) == []
    assert "operation" in columns and "arithmetic_intensity_analytic" in columns
    # El EDP que define la etiqueta se conserva en el frame para auditoria,
    # pero jamas aparece en la lista de caracteristicas.
    assert "cpu_ref_edp_js" in frame.columns
    assert "cpu_ref_edp_js" not in columns


def test_variante_con_sondeo_marca_ausencia_estructural_en_none_ready():
    run_regions = pd.DataFrame([
        {"config_id": "gemm_N8192", "action_id": action, "region": "cold",
         "repetition": rep, "run_id": f"r{rep}", "device": device,
         "time_per_dispatch_s": 2.0, "energy_per_dispatch_j": 8.0,
         "region_to_sampling_ratio": 100.0, "ipc": 1.5, "gpu_power_mw": 90_000.0}
        for device, action in (("cpu", compact.CPU_REF_ACTION), ("gpu", compact.GPU_REF_ACTION))
        for rep in (0, 1, 2)
    ])
    frame = compact.attach_probe_features(
        compact.build_compact_dataset(_ref_only_candidates()), run_regions,
    ).set_index("resource_state")

    assert frame.loc["cpu_ready", "probe_avg_power_w"] == pytest.approx(4.0)
    assert frame.loc["cpu_ready", "probe_avg_power_w_missing"] == 0
    # `none_ready` no ha ejecutado nada: el sondeo no existe y se declara.
    assert np.isnan(frame.loc["none_ready", "probe_avg_power_w"])
    assert frame.loc["none_ready", "probe_avg_power_w_missing"] == 1
    assert compact.leaking_columns(compact.feature_columns(frame.reset_index(), with_probe=True)) == []


# --------------------------------------------------------------------------
# Amortizacion K_break_even (seccion 6.2)
# --------------------------------------------------------------------------


def test_edp_total_es_el_producto_de_las_sumas_no_la_suma_de_productos():
    # K=3: E = 10 + 2*4 = 18, T = 20 + 2*5 = 30 -> 540.
    assert compact.edp_total(10.0, 4.0, 20.0, 5.0, 3) == pytest.approx(540.0)
    # K=1 deja solo el termino frio.
    assert compact.edp_total(10.0, 4.0, 20.0, 5.0, 1) == pytest.approx(200.0)
    with pytest.raises(CompactDatasetError, match="K debe ser"):
        compact.edp_total(1.0, 1.0, 1.0, 1.0, 0)


def test_break_even_es_finito_si_y_solo_si_la_gpu_gana_en_caliente():
    cpu_barata_en_frio = {"e_cold": 1.0, "t_cold": 1.0, "e_warm": 2.0, "t_warm": 2.0}
    gpu_cara_en_frio = {"e_cold": 100.0, "t_cold": 100.0, "e_warm": 1.0, "t_warm": 1.0}
    # GPU cara en frio pero barata en caliente -> cruce finito.
    assert math.isfinite(compact.break_even_k(cpu_barata_en_frio, gpu_cara_en_frio))
    # Sentido inverso: la GPU pierde en caliente Y tambien en el primer
    # despacho, asi que ningun horizonte la amortiza.
    gpu_peor_siempre = {"e_cold": 200.0, "t_cold": 200.0, "e_warm": 2.0, "t_warm": 2.0}
    cpu_mejor_siempre = {"e_cold": 1.0, "t_cold": 1.0, "e_warm": 1.0, "t_warm": 1.0}
    assert compact.break_even_k(cpu_mejor_siempre, gpu_peor_siempre) == float("inf")


def test_gpu_que_gana_en_frio_pero_pierde_en_caliente_cruza_en_uno_y_luego_deja_de_ganar():
    # Caso limite que la nota analitica no cubre: el "menor K con GPU mejor"
    # es 1, pero NO es un punto de amortizacion -- la ventaja se pierde al
    # crecer K en lugar de consolidarse. Se documenta para que un K=1 no se
    # lea como "conviene GPU para cualquier horizonte".
    cpu = {"e_cold": 100.0, "t_cold": 100.0, "e_warm": 1.0, "t_warm": 1.0}
    gpu = {"e_cold": 1.0, "t_cold": 1.0, "e_warm": 2.0, "t_warm": 2.0}
    assert compact.break_even_k(cpu, gpu) == 1.0
    grande = 10_000
    assert compact.edp_total(gpu["e_cold"], gpu["e_warm"], gpu["t_cold"], gpu["t_warm"], grande) > \
           compact.edp_total(cpu["e_cold"], cpu["e_warm"], cpu["t_cold"], cpu["t_warm"], grande)


def test_break_even_vale_uno_cuando_la_gpu_ya_gana_en_el_primer_despacho():
    cpu = {"e_cold": 100.0, "t_cold": 100.0, "e_warm": 100.0, "t_warm": 100.0}
    gpu = {"e_cold": 1.0, "t_cold": 1.0, "e_warm": 1.0, "t_warm": 1.0}
    assert compact.break_even_k(cpu, gpu) == 1.0


def test_break_even_coincide_con_una_busqueda_lineal_directa():
    cpu = {"e_cold": 2.0, "t_cold": 2.0, "e_warm": 2.0, "t_warm": 2.0}
    gpu = {"e_cold": 60.0, "t_cold": 60.0, "e_warm": 1.0, "t_warm": 1.0}
    expected = next(
        k for k in range(1, 10_000)
        if compact.edp_total(gpu["e_cold"], gpu["e_warm"], gpu["t_cold"], gpu["t_warm"], k)
        < compact.edp_total(cpu["e_cold"], cpu["e_warm"], cpu["t_cold"], cpu["t_warm"], k)
    )
    assert compact.break_even_k(cpu, gpu) == float(expected)


def test_la_banda_de_incertidumbre_encierra_el_valor_central():
    frame = compact.amortization_map(_ref_only_candidates()).iloc[0]
    assert frame["k_break_even_low"] <= frame["k_break_even"] <= frame["k_break_even_high"]
    assert frame["gpu_wins_warm"] == 1


def test_el_mapa_verifica_su_propia_prediccion_analitica():
    frame = compact.amortization_map(_two_operation_candidates())
    assert frame.attrs["analytic_inconsistencies"] == []
    finite = np.isfinite(frame["k_break_even"])
    assert (finite == frame["gpu_wins_warm"].astype(bool)).all()
    # stencil: la GPU nunca gana en caliente en este catalogo sintetico.
    assert not np.isfinite(frame.loc[frame["operation"] == "stencil", "k_break_even"]).any()


def test_resolucion_baja_es_desconocida_sin_run_regions_no_falsamente_nominal():
    sin_runs = compact.amortization_map(_ref_only_candidates())
    assert sin_runs["cold_low_resolution"].isna().all()

    run_regions = pd.DataFrame([{
        "config_id": "gemm_N8192", "action_id": compact.CPU_REF_ACTION,
        "region": "cold", "region_to_sampling_ratio": 0.4,
    }])
    con_runs = compact.amortization_map(_ref_only_candidates(), run_regions)
    assert con_runs["cold_low_resolution"].tolist() == [1]


# --------------------------------------------------------------------------
# Headroom DVFS (seccion 11.5)
# --------------------------------------------------------------------------


def test_headroom_se_mide_dentro_del_dispositivo_ganador_de_cada_estado():
    frame = compact.dvfs_headroom(_two_operation_candidates())
    assert len(frame) == 24
    # Las acciones no-REF sinteticas cuestan 0.9x -> headroom de 1 - 0.81.
    assert frame["dvfs_headroom_pct"].tolist() == pytest.approx([19.0] * len(frame))
    # La accion elegida siempre pertenece al dispositivo ganador declarado.
    for row in frame.to_dict("records"):
        assert row["best_action_same_device"].startswith(row["winner_device_at_ref"])


def test_headroom_por_estado_coincide_con_la_etiqueta_del_dataset_compacto():
    candidates = _two_operation_candidates()
    clave = ["config_id", "resource_state"]
    etiquetas = compact.build_compact_dataset(candidates).set_index(clave).sort_index()
    headroom = compact.dvfs_headroom(candidates).set_index(clave).sort_index()
    assert headroom.index.equals(etiquetas.index)
    assert (headroom["winner_device_at_ref"] == etiquetas["device_label"]).all()


# --------------------------------------------------------------------------
# Particiones por tamano (seccion 8.1)
# --------------------------------------------------------------------------


def test_interpolacion_deja_siempre_un_tamano_menor_y_uno_mayor_en_entrenamiento():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    folds = sizes.interpolation_folds(frame, n_folds=2)
    assert folds
    for _, train, test in folds:
        sizes.assert_no_config_leak(train, test)
        for row in test.to_dict("records"):
            train_sizes = set(train.loc[train["operation"] == row["operation"], "size"])
            assert any(size < row["size"] for size in train_sizes)
            assert any(size > row["size"] for size in train_sizes)


def test_extrapolacion_prueba_solo_por_encima_del_rango_de_entrenamiento():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    for name, train, test in sizes.extrapolation_folds(frame, n_largest=2):
        sizes.assert_no_config_leak(train, test)
        for operation, group in test.groupby("operation"):
            maximo_entrenado = train.loc[train["operation"] == operation, "size"].max()
            assert group["size"].min() > maximo_entrenado, name


def test_una_config_nunca_aparece_en_train_y_test_pese_a_tener_tres_estados():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    for _, train, test in sizes.extrapolation_folds(frame):
        # Las 3 filas de estado de una config viajan juntas.
        assert test.groupby("config_id").size().unique().tolist() == [3]
        sizes.assert_no_config_leak(train, test)


# --------------------------------------------------------------------------
# Baselines (seccion 9)
# --------------------------------------------------------------------------


def test_estan_las_once_baselines_obligatorias():
    # Ocho de la seccion 6 del protocolo + tres de la enmienda 2026-08-30-A
    # (seccion 12.4): stay_on_ready_device_k, k_break_even_table_train, oracle_k.
    assert len(sizes.BASELINES) == 11
    assert "oracle" in sizes.BASELINES and "always_cpu_ref" in sizes.BASELINES
    assert {"stay_on_ready_device_k", "k_break_even_table_train", "oracle_k"} <= set(sizes.BASELINES)


def test_el_oraculo_alcanza_exactamente_el_edp_minimo_y_las_constantes_no():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    gpu_ready = frame[frame["resource_state"] == "gpu_ready"]
    oracle = sizes.evaluate_devices(gpu_ready, sizes.BASELINES["oracle"](gpu_ready)(gpu_ready))
    assert oracle["edp_sum_ratio_vs_oracle"] == pytest.approx(1.0)
    assert oracle["balanced_accuracy"] == pytest.approx(1.0)
    for name in ("always_cpu_ref", "always_gpu_ref"):
        constante = sizes.evaluate_devices(gpu_ready, sizes.BASELINES[name](gpu_ready)(gpu_ready))
        assert constante["edp_sum_ratio_vs_oracle"] > 1.0


def test_el_umbral_se_ajusta_solo_con_entrenamiento():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    gpu_ready = frame[frame["resource_state"] == "gpu_ready"]
    train = gpu_ready[gpu_ready["size"] <= 128]
    predict = sizes.BASELINES["size_threshold_train"](train)

    # El corte queda congelado en `fit`: predecir sobre un superconjunto no
    # lo reajusta. Si el umbral se recalculara con los datos de prueba, la
    # prediccion sobre `train` cambiaria al pasar tambien los tamanos grandes.
    solo_train = list(predict(train))
    con_todo = list(predict(gpu_ready))
    assert con_todo[: len(solo_train)] == solo_train
    assert len(con_todo) == len(gpu_ready)


def test_un_umbral_global_de_tamano_degenera_a_constante_cuando_las_operaciones_se_contradicen():
    # gemm cruza a GPU en los tamanos grandes; stencil nunca cruza. Un solo
    # corte sobre `log10_n` no puede servir a ambas, y el ajuste lo reconoce
    # degenerando a "siempre CPU" en vez de inventar un corte intermedio.
    # Es justamente el fallo que la tabla de cruce POR OPERACION corrige.
    frame = compact.build_compact_dataset(_two_operation_candidates())
    gpu_ready = frame[frame["resource_state"] == "gpu_ready"]
    umbral = sizes.BASELINES["size_threshold_train"](gpu_ready)(gpu_ready)
    assert set(umbral) == {"cpu"}

    tabla = sizes.BASELINES["operation_crossover_table_train"](gpu_ready)(gpu_ready)
    elegido = pd.Series(tabla, index=gpu_ready.index)
    assert (elegido[gpu_ready["operation"] == "stencil"] == "cpu").all()
    # En gemm si recupera el cruce que el umbral global no podia expresar.
    assert set(elegido[gpu_ready["operation"] == "gemm"]) == {"cpu", "gpu"}


def test_stay_on_ready_device_es_cpu_cuando_no_hay_dispositivo_preparado():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    devices = sizes.BASELINES["stay_on_ready_device"](frame)(frame)
    elegido = pd.Series(devices, index=frame.index)
    assert (elegido[frame["resource_state"] == "none_ready"] == "cpu").all()
    assert (elegido[frame["resource_state"] == "gpu_ready"] == "gpu").all()


def test_metricas_de_una_decision_perfecta_e_invertida():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    gpu_ready = frame[frame["resource_state"] == "gpu_ready"]
    invertido = np.where(gpu_ready["device_label"] == "gpu", "cpu", "gpu")
    metricas = sizes.evaluate_devices(gpu_ready, invertido)
    assert metricas["accuracy"] == pytest.approx(0.0)
    assert metricas["balanced_accuracy"] == pytest.approx(0.0)
    assert metricas["regret_ratio_median"] > 1.0
    assert metricas["edp_sum_ratio_vs_oracle"] > 1.0
    # Elegir siempre lo peor no captura nada del ahorro del oraculo.
    assert metricas["oracle_savings_captured_pct"] == pytest.approx(0.0)

    perfecto = sizes.evaluate_devices(gpu_ready, gpu_ready["device_label"].to_numpy())
    assert perfecto["accuracy"] == pytest.approx(1.0)
    assert perfecto["regret_ratio_max"] == pytest.approx(1.0)
    assert perfecto["oracle_savings_captured_pct"] == pytest.approx(100.0)


def test_run_baselines_no_mezcla_estados_y_reporta_el_regimen():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    folds = sizes.extrapolation_folds(frame, n_largest=1)
    resultados = sizes.run_baselines(folds)
    assert set(resultados["regime"]) == {"extrapolation"}
    assert set(resultados["resource_state"]) <= set(compact.RESOURCE_STATES)
    assert set(resultados["baseline"]) == set(sizes.BASELINES)


def test_reporte_de_senal_aprendible_usa_el_piso_de_ruido():
    frame = compact.build_compact_dataset(_two_operation_candidates())
    resultados = sizes.run_baselines(sizes.extrapolation_folds(frame, n_largest=1))
    reporte = sizes.baseline_headroom_report(resultados)
    assert "oracle" not in set(reporte["best_baseline"])
    assert set(reporte.columns) >= {
        "above_individual_action_cv_reference",
        "oracle_headroom_over_best_baseline_pct",
        "strict_frozen_protocol_pass_possible",
        "inference_status",
    }


# --------------------------------------------------------------------------
# Data card (seccion 10.1)
# --------------------------------------------------------------------------


def test_data_card_reune_los_apartados_obligatorios(tmp_path):
    candidates = _two_operation_candidates()
    frame = compact.build_compact_dataset(candidates)
    folds = sizes.extrapolation_folds(frame, n_largest=1)
    card = datacard.build_datacard(
        frame, candidates,
        amortization=compact.amortization_map(candidates),
        headroom=compact.dvfs_headroom(candidates),
        folds=folds,
    )
    for section in ("counts", "sizes_by_operation", "device_winner", "device_margin",
                    "winning_actions_by_region", "repetition_cv", "missing_by_feature",
                    "amortization", "fold_balance", "frequency_margin"):
        assert section in card, section
    assert card["counts"]["config_ids"] == 8
    assert card["sizes_by_operation"]["gemm"] == [64, 128, 256, 512]

    paths = datacard.write_datacard(card, tmp_path)
    reread = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert reread["counts"]["config_ids"] == 8
    texto = paths["markdown"].read_text(encoding="utf-8")
    assert "Data card" in texto and "K_break_even" in texto


def test_el_markdown_declara_si_la_prediccion_analitica_se_cumplio():
    candidates = _two_operation_candidates()
    card = datacard.build_datacard(
        compact.build_compact_dataset(candidates), candidates,
        amortization=compact.amortization_map(candidates),
    )
    assert "sin inconsistencias" in datacard.render_markdown(card)
