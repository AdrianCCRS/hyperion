from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase4_evaluacion.edp_report import compare_scenarios, format_report, median_edp_by_kernel_class


def _rows(kernel_ref, label, pkg_uj, n=3, delta_t_ns=1_000_000_000):
    return [
        {
            "kernel_ref": kernel_ref, "phase_label_train": label, "quality_status": "ok",
            "delta_t_ns": delta_t_ns, "pkg_delta_uj": pkg_uj, "dram_delta_uj": 0,
            "energy_valid": True, "gpu_energy_delta_mj": None,
        }
        for _ in range(n)
    ]


def _df(rows):
    df = pd.DataFrame(rows)
    df["device"] = "cpu"
    return df


def test_median_edp_by_kernel_class():
    df = _df(_rows("k1", "compute_bound", pkg_uj=1_000_000))
    edp = median_edp_by_kernel_class(df, "cpu", "compute_bound")
    assert list(edp.index) == ["k1"]
    assert edp.iloc[0] == pytest.approx(1.0)  # 1_000_000 uJ = 1 J, 1s -> EDP = 1.0


def test_compare_scenarios_detecta_mejora_del_agente():
    kernels = ["k1", "k2", "k3", "k4", "k5", "k6"]
    rng = np.random.default_rng(0)

    agent_rows, performance_rows = [], []
    for kernel in kernels:
        agent_pkg = int(500_000 * (1 + rng.normal(0, 0.03)))
        perf_pkg = int(2_000_000 * (1 + rng.normal(0, 0.03)))
        agent_rows += _rows(kernel, "compute_bound", pkg_uj=agent_pkg)
        performance_rows += _rows(kernel, "compute_bound", pkg_uj=perf_pkg)

    windows_by_scenario = {
        "agente": _df(agent_rows),
        "performance": _df(performance_rows),
    }

    comparisons = compare_scenarios(
        windows_by_scenario, agent_scenario="agente", baseline_scenarios=["performance"],
        devices=("cpu",), labels=("compute_bound",),
    )
    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.baseline_scenario == "performance"
    assert c.n_kernels_compared == 6
    assert c.agent_edp_relative_change < -0.5  # el agente gastó bastante menos EDP
    assert c.significance.significant


def test_compare_scenarios_omite_clase_con_menos_de_2_kernels_comunes():
    windows_by_scenario = {
        "agente": _df(_rows("k1", "compute_bound", pkg_uj=500_000)),
        "performance": _df(_rows("k1", "compute_bound", pkg_uj=2_000_000)),
    }
    comparisons = compare_scenarios(
        windows_by_scenario, agent_scenario="agente", baseline_scenarios=["performance"],
        devices=("cpu",), labels=("compute_bound",),
    )
    assert comparisons == []  # solo 1 kernel en común -- sin pares suficientes para la prueba


def test_compare_scenarios_agent_scenario_ausente_lanza():
    with pytest.raises(ValueError, match="agente_no_existe"):
        compare_scenarios({}, agent_scenario="agente_no_existe", baseline_scenarios=[])


def test_format_report_produce_tabla_legible():
    kernels = ["k1", "k2", "k3", "k4", "k5", "k6"]
    rng = np.random.default_rng(1)
    agent_rows, ondemand_rows = [], []
    for kernel in kernels:
        agent_rows += _rows(kernel, "memory_bound", pkg_uj=int(800_000 * (1 + rng.normal(0, 0.02))))
        ondemand_rows += _rows(kernel, "memory_bound", pkg_uj=int(800_000 * (1 + rng.normal(0, 0.02))))
    windows_by_scenario = {"agente": _df(agent_rows), "ondemand": _df(ondemand_rows)}

    comparisons = compare_scenarios(
        windows_by_scenario, agent_scenario="agente", baseline_scenarios=["ondemand"],
        devices=("cpu",), labels=("memory_bound",),
    )
    report = format_report(comparisons)
    assert "ondemand" in report
    assert "memory_bound" in report
