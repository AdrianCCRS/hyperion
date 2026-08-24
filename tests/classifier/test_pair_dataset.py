from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.features import pair_dataset


def _runs():
    # Dos kernels, dos repeticiones, tres niveles (REF = referencia).
    # feat_a duplica el kernel en el nivel REF, para poder verificar que
    # build_pair_dataset copia exactamente ese valor y no otro.
    return pd.DataFrame({
        "kernel_ref": ["k1", "k1", "k1", "k1", "k2", "k2", "k2", "k2"],
        "repetition": [1, 1, 2, 2, 1, 1, 2, 2],
        "freq_level_id": ["REF", "F1", "REF", "F1", "REF", "F1", "REF", "F1"],
        "feat_a": [10.0, 99.0, 12.0, 98.0, 50.0, 199.0, 52.0, 198.0],
        "energy_j": [100.0, 60.0, 110.0, 65.0, 200.0, 260.0, 210.0, 270.0],
        "elapsed_s": [1.0, 1.5, 1.1, 1.6, 2.0, 1.8, 2.1, 1.9],
    })


def test_build_pair_dataset_copia_features_de_la_referencia_no_del_candidato():
    out = pair_dataset.build_pair_dataset(
        _runs(), feature_cols=["feat_a"], ref_level="REF",
    )

    row = out[(out["kernel_ref"] == "k1") & (out["repetition"] == 1)].iloc[0]
    assert row["ref_feat_a"] == pytest.approx(10.0)  # el valor en REF, no en F1 (99.0)
    assert row["freq_level_id"] == "F1"


def test_build_pair_dataset_calcula_ratios_relativos_a_la_referencia():
    out = pair_dataset.build_pair_dataset(
        _runs(), feature_cols=["feat_a"], ref_level="REF",
    )

    row = out[(out["kernel_ref"] == "k2") & (out["repetition"] == 2)].iloc[0]
    assert row["energy_ratio"] == pytest.approx(270.0 / 210.0)
    assert row["time_ratio"] == pytest.approx(1.9 / 2.1)


def test_build_pair_dataset_descarta_candidatos_sin_referencia_y_lo_reporta():
    runs = _runs()
    # Se elimina la fila de referencia de k1/rep1 -- su candidato F1 debe
    # descartarse, no emparejarse con otra repeticion por error.
    runs = runs[~((runs["kernel_ref"] == "k1") & (runs["repetition"] == 1)
                  & (runs["freq_level_id"] == "REF"))]

    out = pair_dataset.build_pair_dataset(runs, feature_cols=["feat_a"], ref_level="REF")

    assert not ((out["kernel_ref"] == "k1") & (out["repetition"] == 1)).any()
    assert out.attrs["dropped_no_ref"] == 1


def test_assert_no_target_leak_pasa_con_features_legitimas():
    pair_dataset.assert_no_target_leak(["gpu_util_pct", "gpu_mem_util_pct"])


def test_assert_no_target_leak_detecta_columna_objetivo():
    with pytest.raises(AssertionError, match="fuga"):
        pair_dataset.assert_no_target_leak(["gpu_util_pct", "energy_ratio"])


def test_assert_no_target_leak_detecta_fuente_de_etiqueta():
    with pytest.raises(AssertionError, match="fuga"):
        pair_dataset.assert_no_target_leak(["ipc", "operational_intensity"])


def test_build_pair_dataset_rechaza_feature_cols_con_fuga_antes_de_construir():
    with pytest.raises(AssertionError, match="fuga"):
        pair_dataset.build_pair_dataset(
            _runs(), feature_cols=["energy_j"], ref_level="REF",
        )


def test_aggregate_cpu_runs_suma_energia_solo_de_ventanas_validas():
    df = pd.DataFrame({
        "kernel_ref": ["k1"] * 3,
        "repetition": [1] * 3,
        "freq_level_id": ["F0"] * 3,
        "t_start_ns": [0, 1_000_000_000, 2_000_000_000],
        "t_end_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000],
        "energy_valid": ["1", "1", "0"],  # la 3ra ventana no cuenta
        "pkg_delta_uj": [1_000_000, 1_000_000, 9_999_999],
        "dram_delta_uj": [100_000, 100_000, 9_999_999],
        "ipc": [1.0, 1.2, 1.4],
        "mpki": [5.0, 5.0, 5.0],
        "llc_miss_rate": [0.1, 0.1, 0.1],
        "stall_backend_ratio": [0.2, 0.2, 0.2],
        "ips": [1e9, 1e9, 1e9],
        "running_ratio": [1.0, 1.0, 1.0],
        "freq_khz_observed": [3200000, 3200000, 3200000],
    })

    out = pair_dataset.aggregate_cpu_runs(df)

    assert len(out) == 1
    assert out.iloc[0]["energy_j"] == pytest.approx(2.2)  # (1.0+1.0+0.1+0.1) J
    assert out.iloc[0]["elapsed_s"] == pytest.approx(3.0)  # rango completo, no solo validas
    assert out.iloc[0]["ipc"] == pytest.approx((1.0 + 1.2 + 1.4) / 3)


def test_honest_constant_baseline_import_smoke():
    # No repite las pruebas de eval/protocol.py; solo confirma que
    # pair_dataset y protocol se usan juntos sin romper import circular.
    from classifier.eval import protocol
    assert callable(protocol.honest_constant_baseline)
