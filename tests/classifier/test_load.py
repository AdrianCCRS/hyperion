from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.features import load


def _write_windows(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "windows.csv", index=False)


def test_load_campaign_windows_concatena_todas_las_corridas(tmp_path):
    _write_windows(tmp_path / "run1", [{"quality_status": "ok", "window_index": 0}])
    _write_windows(tmp_path / "run2", [{"quality_status": "ok", "window_index": 0}])

    df = load.load_campaign_windows(tmp_path)

    assert len(df) == 2


def test_load_campaign_windows_falla_cerrado_si_no_hay_nada(tmp_path):
    with pytest.raises(FileNotFoundError):
        load.load_campaign_windows(tmp_path)


def test_load_run_matrix_construye_la_grilla_explicita_sin_glob(tmp_path, capsys):
    # Simula exactamente el caso real: telemetría + un __baseline al lado,
    # que la matriz explícita nunca debe tocar porque no está en la
    # plantilla.
    _write_windows(
        tmp_path / "camp__npb_cg__F1__rep01", [{"kernel_ref": "npb_cg", "window_index": 0}],
    )
    _write_windows(
        tmp_path / "camp__npb_cg__F1__rep02", [{"kernel_ref": "npb_cg", "window_index": 0}],
    )
    _write_windows(
        tmp_path / "camp__npb_cg__F1__rep01__baseline", [{"kernel_ref": "npb_cg", "window_index": 0}],
    )
    # rep03 falta a propósito -- debe omitirse con aviso, no reventar.

    df = load.load_run_matrix(
        tmp_path, "camp__{kernel}__{level}__rep{rep:02d}",
        kernel=["npb_cg"], level=["F1"], rep=[1, 2, 3],
    )

    assert len(df) == 2
    captured = capsys.readouterr()
    assert "1 corridas sin windows.csv" in captured.out
    assert "rep03" in captured.out


def test_load_run_matrix_falla_cerrado_si_nada_de_la_matriz_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        load.load_run_matrix(
            tmp_path, "camp__{kernel}__rep{rep:02d}", kernel=["npb_cg"], rep=[1],
        )


def test_filter_cpu_trainable_exige_calidad_frecuencia_y_etiqueta():
    df = pd.DataFrame([
        # cumple los tres criterios -- se conserva.
        {"quality_status": "ok", "frequency_quality_status": "valid", "phase_label_train": "compute_bound"},
        # REF (not_applicable_native) también cuenta.
        {"quality_status": "ok", "frequency_quality_status": "not_applicable_native", "phase_label_train": "memory_bound"},
        # frecuencia no confiable -- se descarta aunque el resto esté bien.
        {"quality_status": "ok", "frequency_quality_status": "observation_unreliable", "phase_label_train": "compute_bound"},
        # calidad general no-ok -- se descarta.
        {"quality_status": "pmu_degraded", "frequency_quality_status": "valid", "phase_label_train": "compute_bound"},
        # sin etiqueta -- se descarta.
        {"quality_status": "ok", "frequency_quality_status": "valid", "phase_label_train": ""},
    ])

    result = load.filter_cpu_trainable(df)

    assert len(result) == 2
    assert set(result["phase_label_train"]) == {"compute_bound", "memory_bound"}


def test_filter_gpu_trainable_exige_piso_de_utilizacion_inclusive():
    df = pd.DataFrame([
        # exactamente en el piso (>=, no >) -- se conserva.
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "5.0", "phase_label_train": "compute_bound"},
        # por debajo del piso -- se descarta.
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "4.9", "phase_label_train": "compute_bound"},
        # quality_status distinto de gpu_telemetry -- se descarta aunque tenga utilización alta.
        {"quality_status": "ok", "gpu_util_pct": "50.0", "phase_label_train": "compute_bound"},
        # utilización vacía/no numérica -- se descarta, no explota.
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "", "phase_label_train": "compute_bound"},
    ])

    result = load.filter_gpu_trainable(df)

    assert len(result) == 1
    assert result.iloc[0]["gpu_util_pct"] == "5.0"


def test_filter_gpu_trainable_criterio_de_potencia_reemplaza_al_piso_de_utilizacion():
    # ARC-185: espejo de validate_windows() -- con las lineas de reposo
    # provistas, gpu_util_pct deja de decidir. La fila con util alto pero
    # sin exceso real de potencia se descarta; la de util bajo con exceso
    # real se conserva.
    df = pd.DataFrame([
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "50", "gpu_freq_level_id": "F4",
         "gpu_power_mw": "60500", "phase_label_train": "compute_bound"},
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "1", "gpu_freq_level_id": "F4",
         "gpu_power_mw": "180000", "phase_label_train": "compute_bound"},
    ])

    result = load.filter_gpu_trainable(
        df, idle_power_mw_by_level={"F4": 60000.0}, active_power_margin_mw=50000.0,
    )

    assert len(result) == 1
    assert result.iloc[0]["gpu_power_mw"] == "180000"


def test_filter_gpu_trainable_nivel_sin_linea_de_reposo_se_descarta():
    df = pd.DataFrame([
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "80", "gpu_freq_level_id": "F2",
         "gpu_power_mw": "300000", "phase_label_train": "compute_bound"},
    ])

    result = load.filter_gpu_trainable(
        df, idle_power_mw_by_level={"F4": 60000.0}, active_power_margin_mw=50000.0,
    )

    assert len(result) == 0


def test_filter_gpu_trainable_sin_parametros_preserva_el_piso_de_utilizacion():
    df = pd.DataFrame([
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "5.0", "phase_label_train": "compute_bound"},
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "4.9", "phase_label_train": "compute_bound"},
    ])

    result = load.filter_gpu_trainable(df)

    assert len(result) == 1


def test_filter_gpu_trainable_margen_por_nivel_arc189():
    # ARC-189: mismo bug/arreglo que en validate_windows -- un margen unico
    # rechaza F4 pese a exceso real; el margen por nivel lo acepta.
    df = pd.DataFrame([
        {"quality_status": "gpu_telemetry", "gpu_util_pct": "90", "gpu_freq_level_id": "F4",
         "gpu_power_mw": "38000", "phase_label_train": "compute_bound"},
    ])

    con_margen_unico = load.filter_gpu_trainable(
        df, idle_power_mw_by_level={"F4": 33804.0}, active_power_margin_mw=20000.0,
    )
    assert len(con_margen_unico) == 0

    con_margen_por_nivel = load.filter_gpu_trainable(
        df, idle_power_mw_by_level={"F4": 33804.0},
        active_power_margin_mw={"F4": 800.0},
    )
    assert len(con_margen_por_nivel) == 1
