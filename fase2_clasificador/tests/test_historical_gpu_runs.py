import numpy as np
import pandas as pd
import pytest

from fase2_clasificador.analysis.historical_gpu_runs import build_historical_run_dataset


def _row(run_id="r1", *, accepted=True, quality="gpu_telemetry", label="compute_bound", dt=10, util=80):
    return {
        "run_id": run_id, "kernel_ref": "rodinia_gaussian", "gpu_freq_level_id": "F1",
        "phase_label_train": label, "run_accepted": accepted, "quality_status": quality,
        "delta_t_ns": dt, "gpu_util_pct": util, "gpu_mem_util_pct": 40,
        "gpu_power_mw": 200000, "gpu_sm_clock_mhz": 1000,
    }


def test_agregador_descarta_warmup_y_corridas_rechazadas():
    source = pd.DataFrame([
        _row(util=10), _row(util=30),
        _row(quality="warmup_excluded", util=999),
        _row(run_id="rejected", accepted="False", util=999),
    ])
    runs, report = build_historical_run_dataset([source])
    assert len(runs) == 1
    assert runs.loc[0, "gpu_util_pct_median"] == 20
    assert report["excluded_rows"] == 2
    assert runs.loc[0, "aggregation_granularity"] == "historical_run"


def test_media_ponderada_respeta_duracion_de_cada_muestra():
    source = pd.DataFrame([_row(dt=1, util=0), _row(dt=9, util=100)])
    runs, _ = build_historical_run_dataset([source])
    assert np.isclose(runs.loc[0, "gpu_util_pct_time_weighted_mean"], 90)


def test_media_ponderada_usa_diferencia_de_timestamps_cuando_no_hay_delta():
    source = pd.DataFrame([
        {**_row(util=0), "delta_t_ns": np.nan, "t_end_ns": 100},
        {**_row(util=100), "delta_t_ns": np.nan, "t_end_ns": 200},
        {**_row(util=100), "delta_t_ns": np.nan, "t_end_ns": 1100},
    ])
    runs, _ = build_historical_run_dataset([source])
    assert np.isclose(runs.loc[0, "gpu_util_pct_time_weighted_mean"], 1400 / 15)


def test_agregador_no_acepta_etiquetas_inconsistentes_en_una_corrida():
    source = pd.DataFrame([_row(label="compute_bound"), _row(label="memory_bound")])
    with pytest.raises(ValueError, match="etiqueta histórica debe ser una sola"):
        build_historical_run_dataset([source])


def test_agregador_rechaza_colision_de_run_id_entre_campanas():
    source = pd.DataFrame([
        {**_row(), "source_campaign_id": "campaign_a"},
        {**_row(), "source_campaign_id": "campaign_b"},
    ])
    with pytest.raises(ValueError, match="identidad de corrida inconsistente"):
        build_historical_run_dataset([source])
