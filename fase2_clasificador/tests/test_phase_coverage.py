"""Pruebas del diagnóstico F2-XDEV-001; no requiere hardware."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fase1_telemetria.postprocess import TRAINING_CPU_INTERVALS_FILENAME
from fase2_clasificador.analysis import phase_coverage


def _run_dir(root: Path, campaign_id: str, kernel: str, level: str = "F2") -> Path:
    path = root / f"{campaign_id}__{kernel}__{level}__rep01"
    path.mkdir()
    return path


def test_cpu_coverage_usa_solo_intervalos_validos_y_registra_rechazos(tmp_path):
    campaign_id = "cpu_screen"
    run = _run_dir(tmp_path, campaign_id, "npb_bt")
    pd.DataFrame([
        {
            "kernel_ref": "npb_bt", "freq_level_id": "F2", "phase_label_train": "compute_bound",
            "training_quality_status": "ok", "training_quality_reason": "",
            "operational_intensity_uncore_real": 8.0, "i_ridge_used": 4.0,
        },
        {
            "kernel_ref": "npb_bt", "freq_level_id": "F2", "phase_label_train": "memory_bound",
            "training_quality_status": "ok", "training_quality_reason": "",
            "operational_intensity_uncore_real": 2.0, "i_ridge_used": 4.0,
        },
        {
            "kernel_ref": "npb_bt", "freq_level_id": "F2", "phase_label_train": "memory_bound",
            "training_quality_status": "rejected", "training_quality_reason": "source_window_not_ok",
            "operational_intensity_uncore_real": 2.0, "i_ridge_used": 4.0,
        },
    ]).to_csv(run / TRAINING_CPU_INTERVALS_FILENAME, index=False)

    report = phase_coverage.analyze_campaign(tmp_path, campaign_id, tmp_path / "report")

    assert report["row_semantics"] == "one_uncore_interval_per_row"
    assert report["eligible_for_training_without_additional_aggregation"] is True
    assert report["diagnostics"]["rows_usable"] == 2
    assert report["diagnostics"]["rows_rejected"] == 1
    assert report["diagnostics"]["families_by_coverage"]["npb_bt"] == "mixed"
    coverage = pd.read_csv(tmp_path / "report" / "family_class_frequency_summary.csv")
    assert coverage.loc[0, "n_compute_bound"] == 1
    assert coverage.loc[0, "n_memory_bound"] == 1
    assert coverage.loc[0, "n_near_ridge"] == 2
    quality = pd.read_csv(tmp_path / "report" / "kernel_quality_summary.csv")
    assert quality.loc[0, "primary_rejection_reason"] == "source_window_not_ok"
    assert json.loads((tmp_path / "report" / "phase_coverage_report.json").read_text())["schema_version"] == 1


def test_gpu_coverage_se_declara_no_elegible_hasta_agregar_por_fase(tmp_path):
    campaign_id = "gpu_screen"
    run = _run_dir(tmp_path, campaign_id, "rodinia_gaussian", "REF")
    pd.DataFrame([
        {
            "kernel_ref": "rodinia_gaussian", "freq_level_id": "REF", "gpu_freq_level_id": "F3",
            "phase_label_train": "memory_bound", "quality_status": "gpu_telemetry",
            "operational_intensity": 1.0, "i_ridge_used": 4.0,
        },
        {
            "kernel_ref": "rodinia_gaussian", "freq_level_id": "REF", "gpu_freq_level_id": "F3",
            "phase_label_train": "memory_bound", "quality_status": "warmup_excluded",
            "operational_intensity": 1.0, "i_ridge_used": 4.0,
        },
    ]).to_csv(run / "windows.csv", index=False)

    report = phase_coverage.analyze_campaign(
        tmp_path, campaign_id, tmp_path / "report", device="gpu"
    )

    assert report["eligible_for_training_without_additional_aggregation"] is False
    assert report["diagnostics"]["rows_usable"] == 1
    assert report["pending"]
    quality = pd.read_csv(tmp_path / "report" / "kernel_quality_summary.csv")
    assert quality.loc[0, "primary_rejection_reason"] == "warmup_excluded"
