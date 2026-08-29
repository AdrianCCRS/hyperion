from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import dataset


def _windows() -> pd.DataFrame:
    return pd.DataFrame({
        "t_start_ns": [0, 10, 0, 10],
        "t_end_ns": [10, 20, 10, 20],
        "energy_valid": [1, 1, 0, 0],
        "pkg_delta_uj": [900_000, 900_000, np.nan, np.nan],
        "dram_delta_uj": [100_000, 100_000, np.nan, np.nan],
        "gpu_energy_valid": [0, 0, 1, 1],
        "gpu_energy_delta_mj": [np.nan, np.nan, 1000.0, 1000.0],
        "ipc": [1.0, 3.0, np.nan, np.nan],
        "gpu_power_mw": [np.nan, np.nan, 100_000.0, 200_000.0],
    })


def test_integrate_region_prorratea_ventanas_en_los_dos_bordes():
    result = dataset.integrate_region(
        _windows(), t0_ns=5, t1_ns=15, device="gpu", iterations=2,
        region="cold", interval_ns=10, gpu_interval_ns=10,
    )

    assert result["rapl_energy_total_j"] == pytest.approx(1.0)
    assert result["gpu_energy_total_j"] == pytest.approx(1.0)
    assert result["total_energy_j"] == pytest.approx(2.0)
    assert result["ipc"] == pytest.approx(2.0)
    assert result["gpu_power_mw"] == pytest.approx(150_000.0)
    assert result["rapl_coverage_fraction"] == pytest.approx(1.0)
    assert result["gpu_coverage_fraction"] == pytest.approx(1.0)


def test_reconstruye_inicio_de_intervalos_gpu_persistidos_solo_con_timestamp():
    windows = _windows()
    windows.loc[2:, "t_start_ns"] = np.nan
    result = dataset.integrate_region(
        windows, t0_ns=10, t1_ns=20, device="gpu", iterations=1,
        region="cold", interval_ns=10, gpu_interval_ns=10,
    )
    # La segunda muestra GPU representa el delta 10..20.
    assert result["gpu_energy_total_j"] == pytest.approx(1.0)
    assert result["gpu_coverage_fraction"] == pytest.approx(1.0)


def test_cpu_reemplaza_nvml_observador_por_linea_base_sin_hacerla_cero():
    result = dataset.integrate_region(
        _windows(), t0_ns=0, t1_ns=20, device="cpu", iterations=2,
        region="cold", interval_ns=10, gpu_interval_ns=10,
        idle_gpu_power_w=10.0,
    )

    assert result["gpu_energy_source"] == "idle_baseline"
    assert result["gpu_energy_total_j"] == pytest.approx(10.0 * 20e-9)
    assert result["gpu_energy_raw_observer_j"] == pytest.approx(2.0)
    assert result["total_energy_j"] == pytest.approx(2.0 + 10.0 * 20e-9)


def test_warm_normaliza_energia_y_tiempo_por_iteraciones():
    result = dataset.integrate_region(
        _windows(), t0_ns=0, t1_ns=20, device="gpu", iterations=4,
        region="warm", interval_ns=10, gpu_interval_ns=10,
    )

    assert result["dispatch_count"] == 4
    assert result["time_per_dispatch_s"] == pytest.approx(20e-9 / 4)
    assert result["energy_per_dispatch_j"] == pytest.approx(4.0 / 4)
    assert result["edp_per_dispatch_js"] == pytest.approx(1.0 * 5e-9)


def test_region_menor_que_intervalo_queda_marcada_como_baja_resolucion():
    result = dataset.integrate_region(
        _windows(), t0_ns=1, t1_ns=5, device="cpu", iterations=1,
        region="cold", interval_ns=10, gpu_interval_ns=10,
    )
    assert result["energy_resolution_status"] == "low"
    assert result["region_to_sampling_ratio"] == pytest.approx(0.4)


def test_action_space_tiene_8_cpu_y_32_gpu():
    cpu = dataset.expected_actions("cpu-provisional")
    final = dataset.expected_actions("final")
    assert len(cpu) == 8
    assert len(final) == 40
    assert len([action for action in final if action.startswith("gpu:")]) == 32


def _candidate_rows() -> pd.DataFrame:
    rows = []
    for config_id, operation, size in (("gemm_N64", "gemm", 64), ("axpy_N10000", "axpy", 10000)):
        for region in ("cold", "warm"):
            for action, level, edp in (("cpu:REF", "REF", 2.0), ("cpu:F0", "F0", 1.0)):
                rows.append({
                    "config_id": config_id, "operation": operation, "size": size,
                    "family": "vector" if operation == "axpy" else "matrix",
                    "device": "cpu", "action_id": action, "cpu_level": level,
                    "gpu_level": np.nan, "region": region, "n_repetitions": 3,
                    "eligible_repetitions": True, "edp_mean": edp if region == "cold" else edp / 2,
                    "edp_std": 0.01, "energy_mean": edp, "energy_std": 0.01,
                    "time_mean": 1.0, "time_std": 0.01,
                })
    return pd.DataFrame(rows)


def _probe_runs() -> pd.DataFrame:
    rows = []
    for config_id, operation, size in (("gemm_N64", "gemm", 64), ("axpy_N10000", "axpy", 10000)):
        for repetition in (1, 2, 3):
            rows.append({
                "config_id": config_id, "operation": operation, "size": size,
                "run_id": f"{config_id}_{repetition}",
                "action_id": "cpu:REF", "region": "cold", "repetition": repetition,
                "time_per_dispatch_s": float(repetition), "energy_per_dispatch_j": 2.0,
                "rapl_energy_total_j": 1.5, "gpu_energy_total_j": 0.5,
                "region_to_sampling_ratio": 2.0, "ipc": 1.2, "mpki": 3.0,
            })
    return pd.DataFrame(rows)


def test_strategy_a_usa_cold_y_un_optimo_por_configuracion():
    result = dataset.build_strategy_a(_candidate_rows(), ("cpu:REF", "cpu:F0"))
    assert set(result["region"]) == {"cold"}
    assert len(result) == 4
    assert result.groupby("decision_group_id")["is_optimal"].sum().eq(1).all()
    assert set(result[result["is_optimal"] == 1]["action_id"]) == {"cpu:F0"}


def test_strategy_c_cpu_ready_usa_warm_para_candidatos_cpu():
    result = dataset.build_strategy_c(
        _candidate_rows(), _probe_runs(), ("cpu:REF", "cpu:F0"), probe_devices=("cpu",),
    )
    assert set(result["probe_device"]) == {"cpu"}
    assert set(result["target_region"]) == {"warm"}
    assert "probe_ipc" in result
    assert set(result["probe_time_per_dispatch_s"]) == {1.0}
    assert set(result["probe_avg_power_w"]) == {2.0}
    assert result.groupby("decision_group_id")["is_optimal"].sum().eq(1).all()


def test_strategy_c_no_inventa_telemetria_de_sonda_submuestreada():
    probes = _probe_runs()
    probes["region_to_sampling_ratio"] = 0.5
    result = dataset.build_strategy_c(
        _candidate_rows(), probes, ("cpu:REF", "cpu:F0"), probe_devices=("cpu",),
    )
    assert result["probe_ipc"].isna().all()
    assert result["probe_ipc_missing"].eq(1).all()
    assert result["probe_time_per_dispatch_s"].notna().all()


def test_aggregate_candidates_conserva_cv_y_repeticiones():
    rows = []
    for repetition, edp in enumerate((1.0, 1.1, 0.9), start=1):
        rows.append({
            "config_id": "gemm_N64", "operation": "gemm", "size": 64,
            "family": "matrix", "device": "cpu", "action_id": "cpu:F0",
            "cpu_level": "F0", "gpu_level": np.nan, "region": "cold",
            "repetition": repetition, "time_per_dispatch_s": 1.0,
            "energy_per_dispatch_j": edp, "edp_per_dispatch_js": edp,
            "rapl_energy_total_j": edp, "gpu_energy_total_j": 0.1,
            "gpu_energy_source": "idle_baseline",
        })
    result = dataset.aggregate_candidates(pd.DataFrame(rows), 3).iloc[0]
    assert result["n_repetitions"] == 3
    assert bool(result["eligible_repetitions"])
    assert result["edp_mean"] == pytest.approx(1.0)
    assert result["edp_std"] == pytest.approx(0.1)


def test_iterations_del_launcher_en_cero_usa_exec_args_del_catalogo():
    entry = {"id": "dual_gemm_cpu_N64", "exec_args": "--size 64 --iterations 975"}
    assert dataset._iterations(entry, {"iterations": 0}) == 975
