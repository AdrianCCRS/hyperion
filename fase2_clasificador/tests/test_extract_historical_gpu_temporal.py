import numpy as np
import pandas as pd

from fase2_clasificador.analysis.extract_historical_gpu_temporal import temporal_features


def test_temporal_features_recorta_colas_inactivas_y_detecta_rafagas():
    frame = pd.DataFrame({
        "timestamp_ns": np.arange(8) * 1_000_000,
        "gpu_util_pct": [0, 0, 80, 80, 0, 70, 0, 0],
        "gpu_mem_util_pct": [0, 0, 10, 20, 0, 30, 0, 0],
        "gpu_power_mw": [40_000, 40_000, 120_000, 130_000, 60_000, 110_000, 40_000, 40_000],
        "gpu_sm_clock_mhz": [1000] * 8,
    })
    result = temporal_features(frame)
    assert result["temporal_n_samples"] == 4
    assert result["temporal_gpu_active_fraction"] == 0.75
    assert result["temporal_gpu_burst_count_per_1k"] == 500
    assert result["temporal_gpu_mem_util_pct_mean_abs_diff"] > 0
    assert all(np.isfinite(value) for value in result.values())
