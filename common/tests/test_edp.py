from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.edp import compute_window_edp, load_windows, median_observed_frequency


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


def test_compute_window_edp_gpu_energia_invalida_da_nan():
    """Regresión: compute_window_edp no filtraba por gpu_energy_valid (el
    equivalente GPU de energy_valid, ARC-95) -- una ventana GPU marcada
    como no válida entraba igual en el EDP agregado, sin que nada lo
    señalara. Corregido -- ver el docstring de _valid_mask()."""
    df = pd.DataFrame([{
        "device": "gpu", "delta_t_ns": 1_000_000_000, "pkg_delta_uj": None,
        "dram_delta_uj": None, "energy_valid": None,
        "gpu_energy_delta_mj": 2000, "gpu_energy_valid": False,
    }])
    edp = compute_window_edp(df)
    assert pd.isna(edp.iloc[0])


def test_compute_window_edp_sin_columna_de_validez_no_lanza():
    """_valid_mask() debe tratar la ausencia de la columna como "todo
    válido" sin lanzar -- a diferencia de df.get(col, True).astype(bool),
    que fallaba con AttributeError cuando la columna no existía."""
    df = pd.DataFrame([{
        "device": "cpu", "delta_t_ns": 1_000_000_000, "pkg_delta_uj": 1_000_000,
        "dram_delta_uj": 0, "gpu_energy_delta_mj": None,
        # sin columna "energy_valid" -- caso real: fixtures minimalistas
        # que no cubren energía, no un caso de producción (postprocess.py
        # siempre la incluye).
    }])
    edp = compute_window_edp(df)
    assert edp.iloc[0] == pytest.approx(1.0)


def _window_row(*, quality_status, frequency_quality_status, phase_label_train="compute_bound"):
    return {
        "quality_status": quality_status,
        "frequency_quality_status": frequency_quality_status,
        "phase_label_train": phase_label_train,
        "kernel_ref": "k1", "freq_level_id": "REF", "gpu_freq_level_id": None,
        "delta_t_ns": 1_000_000_000, "pkg_delta_uj": 1_000_000, "dram_delta_uj": 0,
        "energy_valid": True, "gpu_energy_delta_mj": None, "gpu_energy_valid": None,
    }


def test_load_windows_excluye_cpu_con_frecuencia_no_verificada(tmp_path):
    """Regresión: load_windows() no filtraba por frequency_quality_status
    -- una ventana cuyo freq_level_id decía "F4" pero cuyo reloj real no
    había convergido (observation_unreliable/observation_unverified_grace)
    contaminaba el EDP agregado de ese nivel sin que nada lo señalara.
    fase2_clasificador/training/train_phase.py ya exigía este mismo filtro
    -- load_windows() ahora hace lo mismo."""
    rows = [
        _window_row(quality_status="ok", frequency_quality_status="valid"),
        _window_row(quality_status="ok", frequency_quality_status="observation_unreliable"),
        _window_row(quality_status="ok", frequency_quality_status="not_applicable_native"),
    ]
    path = tmp_path / "windows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    df = load_windows([path])
    assert len(df) == 2  # solo "valid" y "not_applicable_native" sobreviven
    assert set(df["frequency_quality_status"]) == {"valid", "not_applicable_native"}


def test_load_windows_no_filtra_gpu_por_frequency_quality_status(tmp_path):
    """frequency_quality_status queda vacía en filas GPU (ver
    postprocess.py) -- load_windows() no debe usarla para excluir filas GPU."""
    rows = [{
        "quality_status": "gpu_telemetry", "frequency_quality_status": np.nan,
        "phase_label_train": "memory_bound", "kernel_ref": "g1",
        "freq_level_id": None, "gpu_freq_level_id": "REF",
        "delta_t_ns": 1_000_000_000, "pkg_delta_uj": None, "dram_delta_uj": None,
        "energy_valid": None, "gpu_energy_delta_mj": 2000, "gpu_energy_valid": True,
    }]
    path = tmp_path / "windows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    df = load_windows([path])
    assert len(df) == 1
    assert df.iloc[0]["device"] == "gpu"


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
