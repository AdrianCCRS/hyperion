from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.features import targets


def test_b_vale_exactamente_medio_en_el_ridge():
    # OI == ridge es la frontera que Fase 1 usa. Si esto no da 0.5 exacto,
    # el target continuo deja de generalizar al binario.
    scores = targets.boundedness_score([8.733], [8.733])

    assert scores[0] == pytest.approx(0.5)


def test_b_respeta_la_convencion_cero_compute_uno_memory():
    # Con k=1 la sigmoide es deliberadamente suave: UNA decada por encima
    # del ridge da b~0.26, no ~0. Para saturar hacen falta ~3 decadas
    # (sigma(-3) = 0.047). Se comprueba la convencion y la saturacion con
    # separaciones realistas, no con expectativas mas duras que la formula.
    ridge = 8.733
    scores = targets.boundedness_score(
        [ridge * 1000, ridge * 10, ridge, ridge / 10, ridge / 1000],
        [ridge] * 5,
    )

    assert scores[0] < 0.05          # 3 decadas compute: sigma(-3)=0.047
    assert scores[1] == pytest.approx(0.2689, abs=1e-3)  # 1 decada: sigma(-1)
    assert scores[2] == pytest.approx(0.5)               # en el ridge
    assert scores[3] == pytest.approx(0.7311, abs=1e-3)  # 1 decada: sigma(1)
    assert scores[4] > 0.95          # 3 decadas memory


def test_b_es_monotono_decreciente_en_la_intensidad():
    scores = targets.boundedness_score([0.1, 1.0, 5.0, 20.0], [5.0] * 4)

    assert list(scores) == sorted(scores, reverse=True)


def test_k_solo_cambia_la_pendiente_no_la_frontera():
    oi = [1.0, 5.0, 25.0]
    ridge = [5.0] * 3

    suave = targets.boundedness_score(oi, ridge, k=0.5)
    abrupto = targets.boundedness_score(oi, ridge, k=4.0)

    # la frontera no se mueve
    assert suave[1] == pytest.approx(0.5)
    assert abrupto[1] == pytest.approx(0.5)
    # pero k grande satura mas
    assert abrupto[0] > suave[0]
    assert abrupto[2] < suave[2]


def test_b_es_nan_donde_el_logaritmo_no_existe():
    # Fabricar 0.5 aqui seria lo peor posible: los haria pasar por "justo
    # en la frontera" cuando en realidad no hay medicion.
    scores = targets.boundedness_score([0.0, -1.0, np.nan, 5.0], [5.0] * 4)

    assert np.isnan(scores[0])
    assert np.isnan(scores[1])
    assert np.isnan(scores[2])
    assert np.isfinite(scores[3])


def test_ridge_por_fila_no_uno_fijo():
    # El ridge cae de 8.733 a 2.992 entre 3200 y 800 MHz. La MISMA
    # intensidad cambia de lado segun la frecuencia -- eso es fisica real
    # (ARC-175), y el score tiene que reflejarlo.
    oi = [6.4, 6.4]
    ridge = [8.733, 2.992]

    scores = targets.boundedness_score(oi, ridge)

    assert scores[0] > 0.5  # a 3200 MHz: memory_bound
    assert scores[1] < 0.5  # a 800 MHz: compute_bound


def test_calibrate_k_es_el_inverso_de_la_dispersion():
    rng = np.random.default_rng(0)
    ridge = np.full(500, 10.0)
    oi = 10.0 * (10 ** rng.normal(0.0, 0.5, 500))

    k = targets.calibrate_k(oi, ridge)

    assert k == pytest.approx(1 / 0.5, rel=0.1)


def test_calibrate_k_falla_cerrado_sin_dispersion():
    with pytest.raises(ValueError):
        targets.calibrate_k([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    with pytest.raises(ValueError):
        targets.calibrate_k([0.0, -1.0], [5.0, 5.0])


def test_binary_from_score_empata_hacia_compute_como_fase_1():
    # Fase 1: memory_bound if OI < ridge. En OI == ridge (b == 0.5) la
    # etiqueta es compute_bound.
    labels = targets.binary_from_score(np.array([0.9, 0.5, 0.1]))

    assert list(labels) == ["memory_bound", "compute_bound", "compute_bound"]


def test_umbralizar_b_reproduce_la_etiqueta_de_fase_1():
    df = pd.DataFrame({
        "operational_intensity_uncore_real": [0.02, 6.4, 21.5, 0.35, 8.733],
        "i_ridge_used": [8.733, 8.733, 8.733, 8.733, 8.733],
        "phase_label_train": [
            "memory_bound", "memory_bound", "compute_bound",
            "memory_bound", "compute_bound",
        ],
    })

    agreement, compared = targets.agreement_with_binary_label(df)

    assert agreement == pytest.approx(1.0)
    assert compared == 5


def test_agreement_ignora_filas_sin_etiqueta_o_sin_medicion():
    df = pd.DataFrame({
        "operational_intensity_uncore_real": [0.02, np.nan, 21.5],
        "i_ridge_used": [8.733, 8.733, 8.733],
        "phase_label_train": ["memory_bound", "memory_bound", ""],
    })

    agreement, compared = targets.agreement_with_binary_label(df)

    assert compared == 1
    assert agreement == pytest.approx(1.0)


def test_add_targets_calibra_k_solo_si_no_se_da():
    df = pd.DataFrame({
        "operational_intensity_uncore_real": [1.0, 5.0, 25.0],
        "i_ridge_used": [5.0] * 3,
    })

    con_k = targets.add_targets(df, k=1.0)
    auto = targets.add_targets(df)

    assert con_k["b"].iloc[1] == pytest.approx(0.5)
    assert auto["b"].iloc[1] == pytest.approx(0.5)
    # con k calibrado la pendiente cambia, la frontera no
    assert not np.isclose(con_k["b"].iloc[0], auto["b"].iloc[0])
