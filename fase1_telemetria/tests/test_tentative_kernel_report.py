from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from fase1_telemetria.tentative_kernel_report import build_report, write_report


def test_informe_distingue_candidato_y_control_sintetico(tmp_path):
    catalog = {"kernels": [
        {"id": "npb_bt", "suite": "NPB", "role": "dataset"},
        {"id": "phasic_p010", "suite": "HYPERION-PHASE", "role": "dataset"},
    ]}
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump(catalog))
    manifest = {
        "kernels": [{"kernel_ref": "npb_bt"}, {"kernel_ref": "phasic_p010"}],
    }
    (tmp_path / "cpu.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "gpu.yaml").write_text(yaml.safe_dump({"kernels": []}))
    campaign = tmp_path / "cpu_campaign"
    for ref in ("npb_bt", "phasic_p010"):
        for rep in (1, 2, 3):
            run = campaign / f"cid__{ref}__F{rep}__rep{rep:02d}"
            run.mkdir(parents=True)
            (run / "verdict.json").write_text('{"accepted": true}')
            pd.DataFrame([{
                "kernel_ref": ref, "repetition": rep, "freq_level_id": f"F{rep}",
                "training_quality_status": "ok", "training_quality_reason": "",
                "phase_label_train": "compute_bound", "operational_intensity_uncore_real": 8,
                "i_ridge_used": 4,
            }]).to_csv(run / "training_cpu_intervals.csv", index=False)
    warmup = tmp_path / "warmup/cpu"
    warmup.mkdir(parents=True)
    (warmup / "warmup_calibration.json").write_text(json.dumps({
        "per_kernel": {ref: {"status": "measured", "warmup_seconds": 0.1}
                       for ref in ("npb_bt", "phasic_p010")}
    }))
    wf = {
        "catalog": str(tmp_path / "catalog.yaml"), "cpu_manifest": str(tmp_path / "cpu.yaml"),
        "gpu_candidates_manifest": str(tmp_path / "gpu.yaml"),
        "cpu_campaign_dir": str(campaign), "cpu_campaign_id": "cid",
        "gpu_campaign_dir": str(tmp_path / "gpu_campaign"), "gpu_campaign_id": "gid",
        "ncu_reports_dir": str(tmp_path / "ncu"),
        "warmup_cpu_dir": str(warmup), "warmup_gpu_dir": str(tmp_path / "warmup/gpu"),
        "utility_dir": str(tmp_path / "utility"),
    }
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps(wf))
    report = build_report(workflow)
    by_ref = {row["kernel_ref"]: row for row in report["kernels"]}
    assert by_ref["npb_bt"]["recommended_action"] == "candidate_for_final_campaign"
    assert by_ref["phasic_p010"]["recommended_action"] == "keep_as_diagnostic_control"
    paths = write_report(report, tmp_path / "out")
    assert all(path.exists() for path in paths)
