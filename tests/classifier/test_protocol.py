from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.eval import protocol


def _dataset():
    return pd.DataFrame({
        "kernel_ref": ["npb_cg"] * 3 + ["npb_mg"] * 2 + ["dgemm_n2048"] * 2,
        "freq_level_id": ["F0", "F1", "F2", "F0", "F1", "F0", "F4"],
        "value": [1, 2, 3, 4, 5, 6, 7],
    })


def test_loko_genera_un_pliegue_por_kernel_en_orden_estable():
    df = _dataset()

    folds = list(protocol.leave_one_kernel_out(df))

    assert [k for _, _, k in folds] == ["dgemm_n2048", "npb_cg", "npb_mg"]


def test_loko_excluye_todas_las_filas_del_kernel_de_prueba():
    df = _dataset()

    for idx_train, idx_test, kernel in protocol.leave_one_kernel_out(df):
        # ninguna repeticion ni nivel del kernel excluido sobrevive en train
        assert kernel not in set(df.iloc[idx_train]["kernel_ref"])
        assert set(df.iloc[idx_test]["kernel_ref"]) == {kernel}
        assert len(idx_train) + len(idx_test) == len(df)


def test_loko_cubre_todas_las_filas_exactamente_una_vez_como_prueba():
    df = _dataset()

    seen = np.concatenate([idx_test for _, idx_test, _ in protocol.leave_one_kernel_out(df)])

    assert sorted(seen) == list(range(len(df)))


def test_loko_falla_cerrado_con_un_solo_kernel():
    df = pd.DataFrame({"kernel_ref": ["npb_cg"] * 5})

    with pytest.raises(ValueError):
        list(protocol.leave_one_kernel_out(df))


def test_guardarrail_detecta_la_fuga_de_kernel():
    df = _dataset()
    # split aleatorio deliberadamente mal hecho: es EXACTAMENTE el error
    # que infla las metricas y que este guardarrail existe para atrapar.
    idx_train = np.array([0, 1, 3, 5])
    idx_test = np.array([2, 4, 6])

    with pytest.raises(AssertionError, match="fuga de kernel"):
        protocol.assert_no_kernel_leak(df, idx_train, idx_test)


def test_guardarrail_acepta_un_split_loko_valido():
    df = _dataset()
    idx_train, idx_test, _ = next(iter(protocol.leave_one_kernel_out(df)))

    protocol.assert_no_kernel_leak(df, idx_train, idx_test)


def test_fold_summary_reporta_el_peor_kernel_no_solo_la_media():
    scores = {"npb_cg": 0.95, "npb_mg": 0.60, "dgemm_n2048": 0.98}

    summary = protocol.fold_summary(scores)

    assert summary["mean"] == pytest.approx((0.95 + 0.60 + 0.98) / 3)
    assert summary["min"] == pytest.approx(0.60)
    assert summary["worst_kernel"] == "npb_mg"
    assert summary["best_kernel"] == "dgemm_n2048"
    assert summary["n_folds"] == 3


def test_fold_summary_falla_sin_pliegues():
    with pytest.raises(ValueError):
        protocol.fold_summary({})


def test_edp_loss_vale_uno_cuando_se_elige_el_optimo():
    oracle = np.array([100.0, 200.0, 50.0])

    assert protocol.edp_loss(oracle, oracle) == pytest.approx(1.0)


def test_edp_loss_mide_el_exceso_agregado():
    oracle = np.array([100.0, 100.0])
    chosen = np.array([110.0, 130.0])

    assert protocol.edp_loss(chosen, oracle) == pytest.approx(1.20)


def test_edp_loss_ignora_filas_no_finitas():
    oracle = np.array([100.0, np.nan, 0.0])
    chosen = np.array([120.0, 500.0, 500.0])

    assert protocol.edp_loss(chosen, oracle) == pytest.approx(1.20)


def test_edp_loss_exige_misma_forma():
    with pytest.raises(ValueError):
        protocol.edp_loss(np.array([1.0]), np.array([1.0, 2.0]))


def test_baseline_siempre_maxima_es_perfecta_si_el_optimo_es_la_maxima():
    # Situacion real de la CPU hoy: el optimo es F0 en todos los tramos, asi
    # que la linea base tonta logra EDP loss 1.0 y cualquier modelo tiene
    # que empatarla para justificar su existencia.
    edp = pd.DataFrame({
        "F0": [100.0, 200.0],
        "F1": [120.0, 260.0],
        "F4": [800.0, 1600.0],
    })

    baselines = protocol.trivial_baselines(edp, max_level="F0")

    assert baselines["siempre_maxima"] == pytest.approx(1.0)
    assert baselines["oraculo"] == 1.0
    assert baselines["al_azar"] > 1.0


def test_baseline_siempre_maxima_pierde_si_hay_optimo_interior():
    # Situacion de la GPU: rodinia_lud minimiza en el nivel mas bajo.
    edp = pd.DataFrame({
        "REF": [5669.0],
        "F1": [4448.0],
        "F4": [3942.0],
    })

    baselines = protocol.trivial_baselines(edp, max_level="REF")

    assert baselines["siempre_maxima"] == pytest.approx(5669.0 / 3942.0)
