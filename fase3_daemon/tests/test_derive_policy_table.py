"""Tests de fase3_daemon/policy/derive_policy_table.py con windows.csv
sintéticos -- no hay campaña real disponible en este entorno de
reconstrucción, así que se construyen filas mínimas que cumplen el
contrato de columnas de fase1_telemetria/postprocess.py.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.policy import derive_policy_table as dpt


def _cpu_rows(kernel_ref, level, label, pkg_uj, run_suffix, n=5, delta_t_ns=1_000_000, freq_khz_observed=800000):
    """n ventanas CPU idénticas (ruido mínimo) para un (kernel, nivel)."""
    rows = []
    for rep in range(n):
        rows.append({
            "run_id": f"{kernel_ref}__{level}__{run_suffix}",
            "kernel_ref": kernel_ref,
            "freq_level_id": level,
            "gpu_freq_level_id": None,
            "phase_label_train": label,
            "quality_status": "ok",
            "delta_t_ns": delta_t_ns,
            "pkg_delta_uj": pkg_uj,
            "dram_delta_uj": 0,
            "energy_valid": True,
            "gpu_energy_delta_mj": None,
            "freq_khz_observed": freq_khz_observed,
            "gpu_sm_clock_mhz": None,
        })
    return rows


def _gpu_rows(kernel_ref, level, label, gpu_energy_mj, run_suffix, n=5, delta_t_ns=1_000_000, gpu_sm_clock_mhz=900):
    rows = []
    for rep in range(n):
        rows.append({
            "run_id": f"{kernel_ref}__{level}__{run_suffix}",
            "kernel_ref": kernel_ref,
            "freq_level_id": None,
            "gpu_freq_level_id": level,
            "phase_label_train": label,
            "quality_status": "gpu_telemetry",
            "delta_t_ns": delta_t_ns,
            "pkg_delta_uj": None,
            "dram_delta_uj": None,
            "energy_valid": None,
            "gpu_energy_delta_mj": gpu_energy_mj,
            "freq_khz_observed": None,
            "gpu_sm_clock_mhz": gpu_sm_clock_mhz,
        })
    return rows


def test_derive_policy_table_elige_nivel_que_reduce_edp_significativamente():
    # 6 kernels CPU compute_bound: REF gasta bastante más EDP que F4 en
    # todos, de forma consistente -> debe elegir F4.
    rows = []
    for i, kernel in enumerate(["k1", "k2", "k3", "k4", "k5", "k6"]):
        rows += _cpu_rows(kernel, "REF", "compute_bound", pkg_uj=2_000_000, run_suffix="ref", n=3, freq_khz_observed=3600000)
        rows += _cpu_rows(kernel, "F4", "compute_bound", pkg_uj=500_000, run_suffix="f4", n=3, freq_khz_observed=800000)
    df = pd.DataFrame(rows)
    is_gpu_row = df["quality_status"] == "gpu_telemetry"
    df["device"] = np.where(is_gpu_row, "gpu", "cpu")

    result = dpt.derive_policy_table(df, t_transicion_gpu_ns=None, alpha=0.05, campaign_id="test")
    entry = result["policy"]["cpu-compute_bound"]
    assert entry["action"] == "actuar"
    assert entry["chosen_level"] == "F4"
    assert entry["edp_relative_gain"] > 0
    # Autocontenida: la frecuencia REAL observada a F4, no solo su ID.
    assert entry["resolved_freq_khz"] == pytest.approx(800000)


def test_derive_policy_table_no_actuar_si_ningun_nivel_mejora():
    # EDP prácticamente idéntico entre REF y F4, con ruido de medición
    # realista (run a run) que domina la diferencia de ~0.1% -- ambos
    # ingredientes hacen falta: sin ruido, hasta una diferencia mínima pero
    # perfectamente consistente sale "significativa" con un test pareado
    # (correcto mecánicamente, pero irreal: ninguna medición real tiene
    # varianza cero entre repeticiones).
    rng = np.random.default_rng(0)
    rows = []
    for kernel in ["k1", "k2", "k3", "k4", "k5", "k6"]:
        ref_pkg = int(1_000_000 * (1 + rng.normal(0, 0.05)))
        f4_pkg = int(999_000 * (1 + rng.normal(0, 0.05)))
        rows += _cpu_rows(kernel, "REF", "compute_bound", pkg_uj=ref_pkg, run_suffix="ref", n=3)
        rows += _cpu_rows(kernel, "F4", "compute_bound", pkg_uj=f4_pkg, run_suffix="f4", n=3)
    df = pd.DataFrame(rows)
    df["device"] = "cpu"

    result = dpt.derive_policy_table(df, t_transicion_gpu_ns=None, alpha=0.05, campaign_id="test")
    entry = result["policy"]["cpu-compute_bound"]
    assert entry["action"] == "no_actuar"
    assert entry["chosen_level"] is None


def test_gpu_sin_t_transicion_medido_siempre_no_actuar():
    # Incluso con una mejora de EDP GPU enorme y significativa, sin
    # T_transición_gpu medido la política debe quedar en no_actuar --
    # es el estado real del proyecto (§2.4.1), no algo que se pueda asumir.
    rows = []
    for kernel in ["g1", "g2", "g3", "g4", "g5", "g6"]:
        rows += _gpu_rows(kernel, "REF", "memory_bound", gpu_energy_mj=5000, run_suffix="ref", n=3)
        rows += _gpu_rows(kernel, "F4", "memory_bound", gpu_energy_mj=500, run_suffix="f4", n=3)
    df = pd.DataFrame(rows)
    df["device"] = "gpu"

    result = dpt.derive_policy_table(df, t_transicion_gpu_ns=None, alpha=0.05, campaign_id="test")
    entry = result["policy"]["gpu-memory_bound"]
    assert entry["action"] == "no_actuar"
    assert entry["reason"] == "t_transicion_gpu_no_medido"


def test_gpu_con_t_transicion_medido_excluye_fases_no_asentadas():
    # Una corrida GPU de duración total menor a t_transicion_gpu_ns debe
    # excluirse del EDP agregado -- se simula con delta_t_ns pequeño.
    rows = _gpu_rows("g1", "F4", "memory_bound", gpu_energy_mj=100, run_suffix="corta", n=2, delta_t_ns=100)
    rows += _gpu_rows("g1", "REF", "memory_bound", gpu_energy_mj=5000, run_suffix="ref", n=2, delta_t_ns=1_000_000_000)
    df = pd.DataFrame(rows)
    df["device"] = "gpu"

    usable, excluded = dpt.filter_gpu_transition_not_settled(df, t_transicion_gpu_ns=1_000_000)
    assert len(excluded) == 2  # las 2 ventanas de la corrida corta
    assert (excluded["run_id"] == "g1__F4__corta").all()
    assert len(usable) == len(df) - 2


def test_filter_gpu_transition_none_no_excluye_nada():
    rows = _gpu_rows("g1", "F4", "memory_bound", gpu_energy_mj=100, run_suffix="corta", n=2, delta_t_ns=100)
    df = pd.DataFrame(rows)
    df["device"] = "gpu"
    usable, excluded = dpt.filter_gpu_transition_not_settled(df, t_transicion_gpu_ns=None)
    assert len(excluded) == 0
    assert len(usable) == len(df)


def test_main_escribe_yaml_valido(tmp_path):
    rows = []
    for kernel in ["k1", "k2", "k3", "k4", "k5", "k6"]:
        rows += _cpu_rows(kernel, "REF", "compute_bound", pkg_uj=2_000_000, run_suffix="ref", n=3)
        rows += _cpu_rows(kernel, "F4", "compute_bound", pkg_uj=500_000, run_suffix="f4", n=3)
    df = pd.DataFrame(rows)
    is_gpu_row = df["quality_status"] == "gpu_telemetry"
    df["device"] = np.where(is_gpu_row, "gpu", "cpu")
    windows_csv = tmp_path / "windows.csv"
    df.to_csv(windows_csv, index=False)

    output = tmp_path / "policy_table.yaml"
    argv = [
        "derive_policy_table.py", str(windows_csv),
        "--campaign-id", "test_campaign", "--output", str(output),
    ]
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = argv
    try:
        dpt.main()
    finally:
        _sys.argv = old_argv

    assert output.exists()
    loaded = yaml.safe_load(output.read_text())
    assert loaded["campaign_id"] == "test_campaign"
    assert set(loaded["policy"].keys()) == {"cpu-compute_bound", "cpu-memory_bound", "gpu-compute_bound", "gpu-memory_bound"}
