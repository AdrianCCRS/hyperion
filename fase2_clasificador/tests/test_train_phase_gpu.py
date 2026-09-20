import pandas as pd
import pytest

from fase1_telemetria.gpu_phases import GPU_PHASE_DATASET_FILENAME
from fase2_clasificador.training.train_phase_gpu import load


def _row(granularity="time_window"):
    return {
        "gpu_util_pct_median": 80, "gpu_mem_util_pct_median": 50,
        "gpu_power_mw_median": 200000, "gpu_sm_clock_mhz_median": 1400,
        "phase_label_train": "compute_bound", "kernel_ref": "rodinia_gaussian",
        "kernel_family": "gaussian", "gpu_freq_level_id": "REF",
        "training_eligible": True, "phase_quality_status": "ok",
        "gpu_frequency_quality_status": "not_applicable_native", "granularity": granularity,
    }


def test_carga_gpu_acepta_solo_dataset_por_ventanas(tmp_path):
    pd.DataFrame([_row()]).to_csv(tmp_path / GPU_PHASE_DATASET_FILENAME, index=False)
    frame = load(tmp_path)
    assert len(frame) == 1


def test_carga_gpu_rechaza_mezcla_historica_por_corrida(tmp_path):
    pd.DataFrame([_row("run")]).to_csv(tmp_path / GPU_PHASE_DATASET_FILENAME, index=False)
    with pytest.raises(ValueError, match="granularidad time_window"):
        load(tmp_path)
