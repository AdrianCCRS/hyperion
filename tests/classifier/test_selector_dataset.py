from pathlib import Path
import json
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload))


def test_accepted_run_dirs_incluye_corridas_saltadas_de_sesiones_previas(tmp_path):
    # ARC: una campana reanudada en varias sesiones (CAM-11) solo agrega a
    # accepted_run_ids lo que la sesion actual acepto de nuevo; las corridas
    # ya aceptadas antes quedan en skipped_run_ids, no duplicadas. Si la
    # ultima sesion no acepta nada nuevo (todo ya estaba hecho),
    # accepted_run_ids queda vacio aunque la campana este completa -- leer
    # solo esa lista deja el dataset vacio en silencio.
    campaign_dir = tmp_path / "campaign"
    run_new = "run_new"
    run_old = "run_old"
    _write_json(campaign_dir / "campaign_metadata.json", {
        "accepted_run_ids": [run_new],
        "skipped_run_ids": [run_old],
    })
    _write_json(campaign_dir / run_new / "verdict.json", {"accepted": True})
    _write_json(campaign_dir / run_old / "verdict.json", {"accepted": True})

    result = dataset._accepted_run_dirs(campaign_dir)

    assert {p.name for p in result} == {run_new, run_old}


def test_accepted_run_dirs_falla_si_skipped_run_ids_ausente(tmp_path):
    campaign_dir = tmp_path / "campaign"
    _write_json(campaign_dir / "campaign_metadata.json", {"accepted_run_ids": []})

    with pytest.raises(dataset.DatasetContractError, match="skipped_run_ids"):
        dataset._accepted_run_dirs(campaign_dir)


def test_accepted_run_dirs_falla_si_un_run_esta_en_ambas_listas(tmp_path):
    campaign_dir = tmp_path / "campaign"
    run_id = "run_dup"
    _write_json(campaign_dir / "campaign_metadata.json", {
        "accepted_run_ids": [run_id],
        "skipped_run_ids": [run_id],
    })
    _write_json(campaign_dir / run_id / "verdict.json", {"accepted": True})

    with pytest.raises(dataset.DatasetContractError, match="a la vez"):
        dataset._accepted_run_dirs(campaign_dir)


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml as _yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_yaml.safe_dump(payload), encoding="utf-8")


def _catalog_entry(kernel_ref: str, config_id: str, device: str = "cpu") -> dict:
    entry = {
        "id": kernel_ref,
        "config_id": config_id,
        "exec_args": "--size 64 --iterations 5",
    }
    if device == "gpu":
        entry["device"] = "gpu"
    return entry


def _write_accepted_run(
    campaign_dir: Path, *, run_id: str, kernel_ref: str, freq_level_id: str,
    repetition_index: int, campaign_id: str, device: str = "cpu",
    gpu_freq_level_id: str | None = None,
) -> None:
    run_dir = campaign_dir / run_id
    metadata = {
        "run_id": run_id,
        "campaign_id": campaign_id,
        "kernel_ref": kernel_ref,
        "freq_level_id": freq_level_id,
        "gpu_freq_level_id": gpu_freq_level_id,
        "repetition_index": repetition_index,
        "iterations": 0,
        "dispatch_timing_contract_valid": True,
        "dispatch_timing": {
            "contract_version": dataset.CONTRACT_VERSION,
            "cold_t0_ns": 0, "setup_complete_ns": 1000,
            "cold_t1_ns": 2000, "warm_t0_ns": 3000, "warm_t1_ns": 4000,
        },
        "interval_ns": 1_000_000,
        "gpu_interval_ns": 5_000_000,
    }
    _write_json(run_dir / "metadata.json", metadata)
    windows = pd.DataFrame({
        "t_start_ns": [0], "t_end_ns": [5000],
        "energy_valid": [1], "pkg_delta_uj": [900_000], "dram_delta_uj": [100_000],
        "gpu_energy_valid": [1 if device == "gpu" else 0],
        "gpu_energy_delta_mj": [1000.0 if device == "gpu" else np.nan],
    })
    windows.to_csv(run_dir / "windows.csv", index=False)
    _write_json(run_dir / "verdict.json", {"accepted": True})


def _write_campaign(
    campaign_dir: Path, *, campaign_id: str, config_ids: list[str], catalog: dict[str, dict],
    device: str = "cpu", levels: tuple[str, ...] = dataset.CPU_LEVELS, repetitions: int = 3,
) -> None:
    accepted_run_ids: list[str] = []
    for config_id in config_ids:
        kernel_ref = f"dual_{config_id}_{device}"
        for level in levels:
            for repetition in range(1, repetitions + 1):
                run_id = f"{campaign_id}__{kernel_ref}__{level}__rep{repetition:02d}"
                _write_accepted_run(
                    campaign_dir, run_id=run_id, kernel_ref=kernel_ref, freq_level_id=level,
                    repetition_index=repetition, campaign_id=campaign_id, device=device,
                )
                accepted_run_ids.append(run_id)
    _write_json(campaign_dir / "campaign_metadata.json", {
        "accepted_run_ids": accepted_run_ids, "skipped_run_ids": [],
    })


def _write_manifest(path: Path, kernel_refs: list[str]) -> None:
    _write_yaml(path, {
        "kernels": [{"kernel_ref": ref} for ref in kernel_refs],
        "frequency_levels": [{"id": level} for level in dataset.CPU_LEVELS],
    })


def test_expected_config_ids_deriva_de_manifiestos_combinados(tmp_path):
    catalog = {
        "dual_gemm_N64_cpu": _catalog_entry("dual_gemm_N64_cpu", "gemm_N64"),
        "dual_gemm_N128_cpu": _catalog_entry("dual_gemm_N128_cpu", "gemm_N128"),
        "dual_gemm_N8192_cpu": _catalog_entry("dual_gemm_N8192_cpu", "gemm_N8192"),
    }
    manifest_a = tmp_path / "manifest_a.yaml"
    manifest_b = tmp_path / "manifest_b.yaml"
    _write_manifest(manifest_a, ["dual_gemm_N64_cpu", "dual_gemm_N128_cpu"])
    _write_manifest(manifest_b, ["dual_gemm_N8192_cpu"])

    combined = dataset.expected_config_ids((manifest_a, manifest_b), catalog)

    assert combined == {"gemm_N64", "gemm_N128", "gemm_N8192"}


def test_build_selector_datasets_un_directorio_por_eje_deriva_conteo_dinamico(tmp_path):
    config_ids = ["gemm_N64", "gemm_N128", "gemm_N192"]
    catalog = {
        f"dual_{cid}_cpu": _catalog_entry(f"dual_{cid}_cpu", cid) for cid in config_ids
    }
    catalog_path = tmp_path / "catalog.yaml"
    _write_yaml(catalog_path, {"kernels": list(catalog.values())})

    campaign_dir = tmp_path / "campaign_full"
    _write_campaign(campaign_dir, campaign_id="full", config_ids=config_ids, catalog=catalog)
    manifest_path = tmp_path / "manifest_full.yaml"
    _write_manifest(manifest_path, [f"dual_{cid}_cpu" for cid in config_ids])

    output_dir = tmp_path / "out"
    result = dataset.build_selector_datasets(dataset.BuildConfig(
        cpu_campaign_dir=campaign_dir,
        catalog_path=catalog_path,
        output_dir=output_dir,
        cpu_manifest_path=manifest_path,
    ))

    completeness = json.loads((output_dir / "completeness.json").read_text())
    assert completeness["complete_config_count"] == len(config_ids)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["expected_config_count"] == len(config_ids)
    assert result["run_regions"].exists()


def test_build_selector_datasets_combina_dos_directorios_del_mismo_eje(tmp_path):
    base_ids = ["gemm_N64", "gemm_N128"]
    big_ids = ["gemm_N8192"]
    all_ids = base_ids + big_ids
    catalog = {
        f"dual_{cid}_cpu": _catalog_entry(f"dual_{cid}_cpu", cid) for cid in all_ids
    }
    catalog_path = tmp_path / "catalog.yaml"
    _write_yaml(catalog_path, {"kernels": list(catalog.values())})

    base_dir = tmp_path / "campaign_full"
    big_dir = tmp_path / "campaign_big"
    _write_campaign(base_dir, campaign_id="full", config_ids=base_ids, catalog=catalog)
    _write_campaign(big_dir, campaign_id="big", config_ids=big_ids, catalog=catalog)

    manifest_full = tmp_path / "manifest_full.yaml"
    manifest_big = tmp_path / "manifest_big.yaml"
    _write_manifest(manifest_full, [f"dual_{cid}_cpu" for cid in base_ids])
    _write_manifest(manifest_big, [f"dual_{cid}_cpu" for cid in big_ids])

    output_dir = tmp_path / "out"
    dataset.build_selector_datasets(dataset.BuildConfig(
        cpu_campaign_dir=[base_dir, big_dir],
        catalog_path=catalog_path,
        output_dir=output_dir,
        cpu_manifest_path=[manifest_full, manifest_big],
    ))

    completeness = json.loads((output_dir / "completeness.json").read_text())
    assert completeness["complete_config_count"] == len(all_ids)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["expected_config_count"] == len(all_ids)
    assert set(provenance["campaigns"][0]["campaign_dir"] for _ in [0]) or True  # sanity: no explota
    assert len(provenance["campaigns"]) == 2


def test_build_selector_datasets_detecta_run_id_duplicado_entre_campanas(tmp_path):
    config_ids = ["gemm_N64"]
    catalog = {f"dual_{cid}_cpu": _catalog_entry(f"dual_{cid}_cpu", cid) for cid in config_ids}
    catalog_path = tmp_path / "catalog.yaml"
    _write_yaml(catalog_path, {"kernels": list(catalog.values())})

    dir_a = tmp_path / "campaign_a"
    dir_b = tmp_path / "campaign_b"
    # Misma campaign_id/kernel_ref/nivel/repeticion -> mismos run_id: debe
    # fallar en vez de mezclar silenciosamente.
    _write_campaign(dir_a, campaign_id="dup", config_ids=config_ids, catalog=catalog)
    _write_campaign(dir_b, campaign_id="dup", config_ids=config_ids, catalog=catalog)

    with pytest.raises(dataset.DatasetContractError, match="duplicado"):
        dataset.build_selector_datasets(dataset.BuildConfig(
            cpu_campaign_dir=[dir_a, dir_b],
            catalog_path=catalog_path,
            output_dir=tmp_path / "out",
        ))


def test_accepted_run_dirs_falla_si_verdict_contradice_skipped(tmp_path):
    campaign_dir = tmp_path / "campaign"
    run_id = "run_bad"
    _write_json(campaign_dir / "campaign_metadata.json", {
        "accepted_run_ids": [],
        "skipped_run_ids": [run_id],
    })
    _write_json(campaign_dir / run_id / "verdict.json", {"accepted": False})

    with pytest.raises(dataset.DatasetContractError, match="contradice verdict"):
        dataset._accepted_run_dirs(campaign_dir)
