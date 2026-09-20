import pandas as pd
import pytest

from fase2_clasificador.analysis.merge_historical_gpu_phase_runs import load_and_merge


def _row(run_id="campaign_a__kernel__REF__gpuF1__rep01", **changes):
    base = {
        "run_id": run_id, "kernel_ref": "kernel", "kernel_family": "family", "gpu_freq_level_id": "F1",
        "phase_label_train": "compute_bound", "granularity": "run", "phase_quality_status": "ok",
        "training_eligible": True, "gpu_frequency_quality_status": "valid",
        "gpu_util_pct_median": 90, "gpu_mem_util_pct_median": 20,
        "gpu_power_mw_median": 200000, "gpu_sm_clock_mhz_median": 1200,
    }
    return {**base, **changes}


def test_fusion_acepta_solo_corridas_historicas_validas(tmp_path):
    good = tmp_path / "a.csv"
    pd.DataFrame([_row(), _row(run_id="campaign_a__warmup", training_eligible=False)]).to_csv(good, index=False)
    merged, report = load_and_merge([good])
    assert len(merged) == 1
    assert report["campaigns"] == ["campaign_a"]
    assert report["rejected"]["not_eligible"] == 1


def test_fusion_rechaza_time_window_para_no_mezclar_con_cupti(tmp_path):
    path = tmp_path / "window.csv"
    pd.DataFrame([_row(granularity="time_window")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ninguna corrida pasó"):
        load_and_merge([path])


def test_fusion_rechaza_run_id_duplicado(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame([_row()]).to_csv(a, index=False)
    pd.DataFrame([_row()]).to_csv(b, index=False)
    with pytest.raises(ValueError, match="run_id duplicado"):
        load_and_merge([a, b])
