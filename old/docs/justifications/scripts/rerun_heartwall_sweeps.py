#!/usr/bin/env python3
"""Re-corre rodinia_heartwall para los sweeps B (gpu_interval_ns) y C/D
(repetitions) despues de corregir el video sintetico truncado (ARC-89,
ver seccion 3 del reporte). Los resultados anteriores para este kernel en
esos dos barridos midieron una corrida que fallaba de inmediato (video de
25 cuadros, exec_args pedia 1000) y deben descartarse."""
import sys
import os

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
os.chdir("/home/latorresn/hyperion-kernels")

import csv
import json
import statistics
from pathlib import Path
from dataclasses import replace
from collections import Counter

from orchestrator import manifest as manifest_module
from orchestrator import runner as runner_module
from orchestrator import postprocess as postprocess_module
from orchestrator.catalog import load_catalog

BASE_MANIFEST_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/campaign_pacca_gpu_ref.yaml"
BASE_CALIBRATION_DIR = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_ref_20260807"
GPU_INTERVAL_VALUES_NS = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]
KERNEL_REF = "rodinia_heartwall"


def run_and_collect(manifest, entry, rep, freq_level_id="REF"):
    result = runner_module.run_single(entry, manifest, KERNEL_REF, freq_level_id, rep, node_id="pacca-a100")
    row = {"success": result.success, "elapsed_seconds": result.elapsed_seconds}
    if result.success:
        windows_path = postprocess_module.run_postprocess(
            result.run_dir, run_id=result.run_id, repetition=rep, kernel_ref=KERNEL_REF,
            kernel_entry=entry, node_id="pacca-a100", freq_level_id=freq_level_id,
            calibration_dir=BASE_CALIBRATION_DIR,
            freq_khz_observed=2261000, warmup_seconds=entry.warmup_seconds or 0.0,
            running_ratio_min=manifest.running_ratio_min, rapl_enabled=False,
        )
        with open(windows_path, newline="") as f:
            wrows = list(csv.DictReader(f))
        quality = Counter(r["quality_status"] for r in wrows)
        row["n_windows_total"] = len(wrows)
        row["n_gpu_telemetry"] = quality.get("gpu_telemetry", 0)
        row["quality_json"] = json.dumps(dict(quality))
    else:
        row["postprocess_error"] = "run_failed"
    return row


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))
    entry = catalog[KERNEL_REF]

    # --- gpu_interval_ns sweep ---
    interval_rows = []
    for gpu_interval_ns in GPU_INTERVAL_VALUES_NS:
        sweep_manifest = replace(
            base_manifest,
            output_dir=Path(f"/home/latorresn/hyperion-results/sweeps/gpu_interval_ns/gpu_interval_{gpu_interval_ns}"),
            gpu_interval_ns=gpu_interval_ns, campaign_id=f"sweep_gpu_interval_{gpu_interval_ns}",
        )
        for rep in range(1, 4):
            row = run_and_collect(sweep_manifest, entry, rep)
            row.update({"gpu_interval_ns": gpu_interval_ns, "kernel_ref": KERNEL_REF, "repetition": rep})
            interval_rows.append(row)
            print(f"[gpu_interval_ns={gpu_interval_ns}] heartwall rep{rep}: {row}")

    out1 = Path("/home/latorresn/hyperion-results/sweeps/gpu_interval_ns/rodinia_heartwall_fix.csv")
    fieldnames = sorted({k for r in interval_rows for k in r})
    with open(out1, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interval_rows)
    print(f"Escrito: {out1}")

    # --- repetitions sweep ---
    rep_manifest = replace(
        base_manifest,
        output_dir=Path("/home/latorresn/hyperion-results/sweeps/repetitions/runs"),
        campaign_id="sweep_repetitions",
    )
    rep_rows = []
    for rep in range(1, 11):
        row = run_and_collect(rep_manifest, entry, rep)
        row.update({"kernel_ref": KERNEL_REF, "device": "gpu", "repetition": rep, "target_windows_per_repetition": 5})
        row["windows_achieved"] = row.pop("n_gpu_telemetry", None)
        rep_rows.append(row)
        print(f"[repetitions] heartwall rep{rep}: {row}")

    out2 = Path("/home/latorresn/hyperion-results/sweeps/repetitions/rodinia_heartwall_fix.csv")
    fieldnames = sorted({k for r in rep_rows for k in r})
    with open(out2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rep_rows)
    print(f"Escrito: {out2}")


if __name__ == "__main__":
    main()
