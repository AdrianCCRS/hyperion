from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.edp import compute_window_edp, median_observed_frequency


def test_compute_window_edp_cpu_y_gpu():
    df = pd.DataFrame([
        {
            "device": "cpu", "delta_t_ns": 1_000_000_000, "pkg_delta_uj": 1_000_000,
            "dram_delta_uj": 0, "energy_valid": True, "gpu_energy_delta_mj": None,
        },
        {
            "device": "gpu", "delta_t_ns": 1_000_000_000, "pkg_delta_uj": None,
            "dram_delta_uj": None, "energy_valid": None, "gpu_energy_delta_mj": 2000,
        },
    ])
    edp = compute_window_edp(df)
    # CPU: 1_000_000 uJ = 1 J, 1s -> EDP = 1 * 1 = 1.0
    assert edp.iloc[0] == pytest.approx(1.0)
    # GPU: 2000 mJ = 2 J, 1s -> EDP = 2 * 1 = 2.0
    assert edp.iloc[1] == pytest.approx(2.0)


def test_compute_window_edp_energia_invalida_da_nan():
    df = pd.DataFrame([{
        "device": "cpu", "delta_t_ns": 1_000_000_000, "pkg_delta_uj": 1_000_000,
        "dram_delta_uj": 0, "energy_valid": False, "gpu_energy_delta_mj": None,
    }])
    edp = compute_window_edp(df)
    assert pd.isna(edp.iloc[0])


def test_median_observed_frequency_cpu():
    df = pd.DataFrame([
        {"device": "cpu", "phase_label_train": "compute_bound", "freq_level_id": "F0", "freq_khz_observed": 3600000},
        {"device": "cpu", "phase_label_train": "compute_bound", "freq_level_id": "F0", "freq_khz_observed": 3550000},
        {"device": "cpu", "phase_label_train": "compute_bound", "freq_level_id": "F4", "freq_khz_observed": 800000},
    ])
    assert median_observed_frequency(df, "cpu", "compute_bound", "F0") == pytest.approx(3575000)


def test_median_observed_frequency_sin_datos_da_none():
    df = pd.DataFrame([
        {"device": "cpu", "phase_label_train": "compute_bound", "freq_level_id": "F0", "freq_khz_observed": 3600000},
    ])
    assert median_observed_frequency(df, "cpu", "memory_bound", "F0") is None
