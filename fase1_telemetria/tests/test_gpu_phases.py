"""F1-GPU-003: pruebas del dataset intermedio GPU por fase/corrida.

Lo más importante: varias muestras NVML de una misma corrida NO se convierten
en ejemplos ML independientes -- producen UNA fila con agregados robustos.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import gpu_phases


def _gpu_win(run_id, t_end_ns, *, util, power, sm_clock, mem_util=50, temp=60,
            label="compute_bound", kernel_ref="rodinia_lud", quality="gpu_telemetry",
            energy_delta=None, energy_valid=False, freq_level="REF", gpu_freq_level="REF",
            rep=1):
    return {
        "run_id": run_id, "repetition": rep, "kernel_ref": kernel_ref,
        "node_id": "paccaA100", "freq_level_id": freq_level,
        "gpu_freq_level_id": gpu_freq_level, "binary_checksum": "sha256:abc",
        "roofline_calibration_ref": "/cal/roofline_calibration_gpu_fp64.json",
        "operational_intensity": 12.0, "i_ridge_used": 3.3,
        "phase_label_train": label, "quality_status": quality,
        "t_end_ns": t_end_ns,
        "gpu_util_pct": util, "gpu_mem_util_pct": mem_util, "gpu_power_mw": power,
        "gpu_sm_clock_mhz": sm_clock, "gpu_temperature_c": temp,
        "gpu_energy_delta_mj": energy_delta, "gpu_energy_valid": energy_valid,
    }


def test_muchas_muestras_de_una_corrida_producen_una_sola_fila():
    # 30 muestras NVML, muchas con el mismo valor (escalón del sensor)
    wins = []
    for i in range(30):
        wins.append(_gpu_win("run_A", 1_000_000_000 + i * 5_000_000,
                             util=95 if i < 20 else 96,
                             power=250_000 if i < 15 else 255_000,
                             sm_clock=1410))
    rows = gpu_phases.build_gpu_phase_rows(wins)
    assert len(rows) == 1
    r = rows[0]
    assert r["run_id"] == "run_A"
    assert r["granularity"] == "run"
    assert r["n_nvml_samples"] == 30
    assert r["training_eligible"] is True
    assert r["phase_quality_status"] == "ok"
    # agregados robustos
    assert r["gpu_sm_clock_mhz_median"] == 1410
    assert r["gpu_util_pct_min"] == 95 and r["gpu_util_pct_max"] == 96
    # frescura: potencia tomó 2 valores distintos, util 2, sm_clock 1
    assert r["gpu_power_mw_n_distinct"] == 2
    assert r["gpu_sm_clock_mhz_n_distinct"] == 1
    assert r["gpu_util_pct_valid_frac"] == 1.0
    # duración cubierta = (29 * 5ms)
    assert r["covered_duration_ns"] == 29 * 5_000_000


def test_dos_corridas_dan_dos_filas_agrupadas_por_run_id():
    wins = [_gpu_win("run_A", 1000 + i * 100, util=90, power=200_000, sm_clock=1400) for i in range(10)]
    wins += [_gpu_win("run_B", 2000 + i * 100, util=40, power=120_000, sm_clock=900) for i in range(10)]
    rows = gpu_phases.build_gpu_phase_rows(wins, min_nvml_samples=5)
    assert {r["run_id"] for r in rows} == {"run_A", "run_B"}
    a = next(r for r in rows if r["run_id"] == "run_A")
    b = next(r for r in rows if r["run_id"] == "run_B")
    assert a["gpu_util_pct_median"] == 90
    assert b["gpu_sm_clock_mhz_median"] == 900


def test_muestras_de_warmup_se_cuentan_aparte_y_bajan_la_fraccion_usable():
    wins = [_gpu_win("run_A", 100 + i, util=90, power=200_000, sm_clock=1400,
                     quality="warmup_excluded") for i in range(15)]
    wins += [_gpu_win("run_A", 1000 + i * 100, util=90, power=200_000, sm_clock=1400)
             for i in range(15)]
    rows = gpu_phases.build_gpu_phase_rows(wins, min_nvml_samples=5,
                                          min_usable_sample_fraction=0.6)
    r = rows[0]
    assert r["n_nvml_samples"] == 15
    assert r["n_nvml_samples_warmup_excluded"] == 15
    assert r["usable_sample_fraction"] == 0.5
    # 0.5 < 0.6 -> no elegible
    assert r["training_eligible"] is False
    assert r["phase_quality_status"] == "insufficient_samples"


def test_pocas_muestras_no_es_elegible():
    wins = [_gpu_win("run_A", 1000 + i * 100, util=90, power=2e5, sm_clock=1400) for i in range(4)]
    rows = gpu_phases.build_gpu_phase_rows(wins, min_nvml_samples=8)
    assert rows[0]["training_eligible"] is False
    assert rows[0]["phase_quality_status"] == "insufficient_samples"


def test_reloj_gpu_fijo_se_verifica_sobre_muestras_bajo_carga():
    wins = [_gpu_win("run_A", 1000 + i * 100, util=90, power=2e5, sm_clock=1200)
            for i in range(20)]
    rows = gpu_phases.build_gpu_phase_rows(
        wins, gpu_freq_mhz_requested=900, gpu_freq_mhz_applied=900,
        gpu_freq_tolerance_fraction=0.05,
    )
    row = rows[0]
    assert row["gpu_frequency_quality_status"] == "invalid"
    assert row["gpu_frequency_valid_fraction"] == 0.0
    assert row["training_eligible"] is False
    assert row["phase_quality_status"] == "gpu_frequency_invalid"


def test_sin_etiqueta_no_es_elegible():
    wins = [_gpu_win("run_A", 1000 + i * 100, util=90, power=2e5, sm_clock=1400, label="")
            for i in range(20)]
    rows = gpu_phases.build_gpu_phase_rows(wins)
    assert rows[0]["training_eligible"] is False
    assert rows[0]["phase_quality_status"] == "label_missing"


def test_gpu_phasic_es_control_diagnostico_no_entrenamiento():
    wins = [_gpu_win("run_P", 1000 + i * 100, util=90, power=2e5, sm_clock=1400,
                     kernel_ref="gpu_phasic_p1000") for i in range(50)]
    rows = gpu_phases.build_gpu_phase_rows(wins)
    r = rows[0]
    assert r["training_eligible"] is False
    assert r["phase_quality_status"] == "phasic_control_needs_marks"
    # aún así produce agregados para diagnóstico
    assert r["n_nvml_samples"] == 50
    assert r["gpu_util_pct_median"] == 90


def test_energia_solo_suma_deltas_validos():
    wins = []
    for i in range(20):
        valid = i >= 2  # las 2 primeras sin energía válida
        wins.append(_gpu_win("run_A", 1000 + i * 100, util=90, power=2e5, sm_clock=1400,
                             energy_delta=100 if valid else 0, energy_valid=valid))
    rows = gpu_phases.build_gpu_phase_rows(wins)
    r = rows[0]
    assert r["gpu_energy_covered"] is True
    assert r["gpu_energy_delta_mj_sum"] == 18 * 100


def test_filas_no_gpu_se_ignoran():
    wins = [
        {"run_id": "run_cpu", "quality_status": "ok", "kernel_ref": "npb_bt"},
        _gpu_win("run_A", 1000, util=90, power=2e5, sm_clock=1400),
    ]
    # 1 muestra GPU sola -> fila con no_usable/insufficient, pero la fila CPU no aparece
    rows = gpu_phases.build_gpu_phase_rows(wins, min_nvml_samples=1)
    assert [r["run_id"] for r in rows] == ["run_A"]


def test_contrato_de_granularidad_declara_lo_esencial():
    c = gpu_phases.granularity_contract()
    assert c["row_unit"] == "run"
    assert c["nvml_sample_is_independent_example"] is False
    assert "phase_label_train" in c["label_and_truth_columns_forbidden_as_features"]
    assert c["phasic_kernels_training_eligible"] is False


def test_csv_y_contrato_se_escriben(tmp_path):
    wins = [_gpu_win("run_A", 1000 + i * 100, util=90, power=2e5, sm_clock=1400) for i in range(20)]
    rows = gpu_phases.build_gpu_phase_rows(wins)
    csv_path = gpu_phases.write_gpu_phases_csv(rows, tmp_path / gpu_phases.GPU_PHASE_DATASET_FILENAME)
    contract_path = gpu_phases.write_contract(tmp_path / gpu_phases.GPU_PHASE_CONTRACT_FILENAME)
    assert csv_path.exists() and contract_path.exists()
    header = csv_path.read_text().splitlines()[0].split(",")
    assert "training_eligible" in header
    assert "gpu_power_mw_median" in header
    assert "phase_label_train" in header  # trazabilidad, no feature
