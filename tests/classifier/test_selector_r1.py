from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import r1


def _candidates() -> pd.DataFrame:
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 128, 256, 512):
            config_id = f"{operation}_N{size}"
            cpu = (size / 64) ** 2
            gpu = 20.0 if operation == "gemm" else 200.0
            for device, ref, other in (("cpu", cpu, cpu * 0.9), ("gpu", gpu, gpu * 0.9)):
                for action, cost in (
                    (("cpu:REF" if device == "cpu" else "gpu:REF:REF"), ref),
                    (("cpu:F0" if device == "cpu" else "gpu:REF:F0"), other),
                ):
                    for region, multiplier in (("cold", 3.0), ("warm", 1.0)):
                        energy = cost * multiplier
                        time = cost * multiplier
                        rows.append({
                            "config_id": config_id, "operation": operation, "size": size,
                            "family": "matrix", "device": device, "action_id": action,
                            "region": region, "n_repetitions": 3,
                            "eligible_repetitions": True,
                            "energy_mean": energy, "energy_min": energy * 0.99,
                            "energy_max": energy * 1.01, "energy_std": energy * 0.01,
                            "time_mean": time, "time_min": time * 0.99,
                            "time_max": time * 1.01, "time_std": time * 0.01,
                            "edp_mean": energy * time, "edp_std": energy * time * 0.02,
                            "edp_cv_pct": 2.0, "time_cv_pct": 1.0, "energy_cv_pct": 1.0,
                        })
    return pd.DataFrame(rows)


def _runs() -> pd.DataFrame:
    rows = []
    for operation in ("gemm", "stencil"):
        for size in (64, 128, 256, 512):
            config_id = f"{operation}_N{size}"
            for device, action in (("cpu", "cpu:REF"), ("gpu", "gpu:REF:REF")):
                for repetition in (1, 2, 3):
                    rows.append({
                        "config_id": config_id, "action_id": action, "region": "cold",
                        "repetition": repetition, "run_id": f"{config_id}-{device}-{repetition}",
                        "device": device, "time_per_dispatch_s": 1.0,
                        "energy_per_dispatch_j": 1.0, "region_to_sampling_ratio": 10.0,
                        "rapl_coverage_fraction": 1.0, "gpu_coverage_fraction": 1.0,
                        "ipc": 1.5, "gpu_power_mw": 100_000.0,
                    })
    return pd.DataFrame(rows)


def test_run_r1_escribe_un_solo_contrato_canonico(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _candidates().to_csv(dataset_dir / "candidate_summary.csv", index=False)
    _runs().to_csv(dataset_dir / "run_regions.csv", index=False)

    paths = r1.run_r1_analysis(dataset_dir)

    expected = {
        "compact_static", "compact_with_probe", "amortization_map", "dvfs_headroom",
        "size_folds", "interpolation_baselines", "extrapolation_baselines",
        "baseline_metrics", "baseline_oracle_headroom", "datacard_json",
        "datacard_markdown", "summary",
    }
    assert set(paths) == expected
    assert all(path.exists() for path in paths.values())
    compact = pd.read_csv(paths["compact_static"])
    assert len(compact) == 8 * 3
    assert compact.groupby(["config_id", "resource_state"]).size().max() == 1
    summary = json.loads(paths["summary"].read_text())
    assert summary["effective_config_count"] == 8
    assert summary["interpretation_contract"]["k_is_supplied_not_predicted"] is True


def test_r1_no_exige_los_datasets_antiguos_de_40_acciones(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _candidates().to_csv(dataset_dir / "candidate_summary.csv", index=False)
    _runs().to_csv(dataset_dir / "run_regions.csv", index=False)

    paths = r1.run_r1_analysis(dataset_dir)

    assert np.isfinite(pd.read_csv(paths["compact_static"])["y_log_edp_ratio"]).all()
    assert not (dataset_dir / "strategy_a_candidates.csv").exists()
    assert not (dataset_dir / "strategy_c_candidates.csv").exists()
