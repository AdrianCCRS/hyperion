from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.features import align


def _run(kernel, level, rep, instr, **extra):
    """Construye las ventanas de una corrida con los conteos dados."""
    n = len(instr)
    base = {
        "kernel_ref": [kernel] * n,
        "freq_level_id": [level] * n,
        "repetition": [rep] * n,
        "delta_instructions": instr,
    }
    base.update({k: v for k, v in extra.items()})
    return pd.DataFrame(base)


def test_progress_por_instrucciones_termina_en_uno_por_corrida():
    df = pd.concat([
        _run("npb_cg", "F0", 1, [10, 30, 60]),
        _run("npb_cg", "F4", 1, [5, 5, 10, 30, 50]),
    ], ignore_index=True)

    out = align.add_instruction_progress(df)

    f0 = out[out["freq_level_id"] == "F0"]["progress"].tolist()
    f4 = out[out["freq_level_id"] == "F4"]["progress"].tolist()
    assert f0 == pytest.approx([0.1, 0.4, 1.0])
    assert f4 == pytest.approx([0.05, 0.10, 0.20, 0.50, 1.0])


def test_progress_alinea_el_mismo_punto_logico_entre_frecuencias():
    # La corrida lenta tiene el DOBLE de ventanas, pero ejecuta el mismo
    # trabajo -- a mitad de programa ambas deben marcar progress 0.5.
    df = pd.concat([
        _run("k", "F0", 1, [50, 50]),
        _run("k", "F4", 1, [25, 25, 25, 25]),
    ], ignore_index=True)

    out = align.add_instruction_progress(df)
    rapida = out[out["freq_level_id"] == "F0"]["progress"].tolist()
    lenta = out[out["freq_level_id"] == "F4"]["progress"].tolist()

    assert rapida[0] == pytest.approx(0.5)
    assert lenta[1] == pytest.approx(0.5)


def test_progress_es_nan_si_la_corrida_no_retiro_instrucciones():
    # Caso real: filas de GPU passthrough, sin PMU de CPU. No se debe
    # fabricar un progreso uniforme.
    df = _run("gpu_kernel", "F0", 1, [0, 0, 0])

    out = align.add_instruction_progress(df)

    assert out["progress"].isna().all()


def test_progress_por_tiempo_como_respaldo():
    df = pd.DataFrame({
        "kernel_ref": ["g"] * 4,
        "freq_level_id": ["F0"] * 4,
        "repetition": [1] * 4,
        "t_start_ns": [0, 100, 200, 300],
        "t_end_ns": [100, 200, 300, 400],
    })

    out = align.add_time_progress(df)

    assert out["progress"].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_bins_incluyen_el_extremo_uno_en_el_ultimo_bin():
    # progress == 1.0 * n_bins daria el bin n (inexistente): la ultima
    # ventana de CADA corrida se perderia sin el clip.
    df = pd.DataFrame({"progress": [0.0, 0.049, 0.05, 0.99, 1.0]})

    out = align.assign_progress_bins(df, n_bins=20)

    assert out["progress_bin"].tolist() == [0, 0, 1, 19, 19]


def test_bins_rechaza_n_bins_invalido():
    with pytest.raises(ValueError):
        align.assign_progress_bins(pd.DataFrame({"progress": [0.5]}), n_bins=0)


def test_aggregate_suma_extensivas_y_promedia_features():
    df = pd.DataFrame({
        "kernel_ref": ["k"] * 4,
        "repetition": [1] * 4,
        "freq_level_id": ["F0"] * 4,
        "progress_bin": [0, 0, 1, 1],
        "ipc": [1.0, 3.0, 2.0, 2.0],
        "pkg_delta_uj": [100, 200, 50, 50],
        "delta_t_ns": [10, 10, 10, 10],
    })

    out = align.aggregate_cells(df, feature_cols=["ipc"]).sort_values("progress_bin")

    # ipc se promedia (intensiva), energia y duracion se suman (extensivas).
    assert out["ipc"].tolist() == pytest.approx([2.0, 2.0])
    assert out["energy_uj"].tolist() == pytest.approx([300, 100])
    assert out["duration_ns"].tolist() == pytest.approx([20, 20])
    assert out["n_windows"].tolist() == [2, 2]


def test_aggregate_descarta_filas_sin_bin():
    df = pd.DataFrame({
        "kernel_ref": ["k"] * 3,
        "repetition": [1] * 3,
        "freq_level_id": ["F0"] * 3,
        "progress_bin": pd.array([0, None, 1], dtype="Int64"),
        "ipc": [1.0, 9.9, 2.0],
        "pkg_delta_uj": [100, 999, 200],
        "delta_t_ns": [10, 99, 10],
    })

    out = align.aggregate_cells(df, feature_cols=["ipc"])

    assert len(out) == 2
    assert 9.9 not in out["ipc"].tolist()


def test_fit_alpha_recupera_una_carga_perfectamente_sensible():
    # alpha = 1: el tiempo escala exactamente con el inverso del reloj.
    durations = {3200: 1.0, 1600: 2.0, 800: 4.0}

    alpha, r2 = align.fit_alpha(durations, f_ref_mhz=3200)

    assert alpha == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_fit_alpha_recupera_una_carga_insensible_al_reloj():
    # alpha = 0: bajar el reloj no alarga nada (todo es espera a memoria).
    durations = {3200: 5.0, 1600: 5.0, 800: 5.0}

    alpha, _ = align.fit_alpha(durations, f_ref_mhz=3200)

    assert alpha == pytest.approx(0.0)


def test_fit_alpha_caso_intermedio_conocido():
    # Mitad del tiempo sensible: a la mitad del reloj, T = 0.5 + 0.5*2 = 1.5
    durations = {3200: 1.0, 1600: 1.5, 800: 2.5}

    alpha, r2 = align.fit_alpha(durations, f_ref_mhz=3200)

    assert alpha == pytest.approx(0.5)
    assert r2 == pytest.approx(1.0)


def test_fit_alpha_exige_la_referencia_y_al_menos_dos_puntos():
    with pytest.raises(ValueError):
        align.fit_alpha({1600: 2.0}, f_ref_mhz=3200)
    with pytest.raises(ValueError):
        align.fit_alpha({3200: 1.0}, f_ref_mhz=3200)
