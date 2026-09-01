#!/usr/bin/env python3
"""Re-corre dgemm_n2048 (unico kernel que fallo en los sweeps A y C/D por
libopenblas.so.0 ausente del LD_LIBRARY_PATH bajo sbatch --wrap) con la
ruta de OpenBLAS del sistema (modulo gnu12/openblas 0.3.21) fijada
explicitamente antes de invocar el harness."""
import sys
import os

os.environ["LD_LIBRARY_PATH"] = (
    "/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)

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
INTERVAL_VALUES_NS = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
KERNEL_REF = "dgemm_n2048"


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
        ok_rows = [r for r in wrows if r["quality_status"] == "ok"]
        ipc_vals = [float(r["ipc"]) for r in ok_rows if r.get("ipc") not in (None, "")]
        row["n_windows_total"] = len(wrows)
        row["n_windows_ok"] = quality.get("ok", 0)
        row["ipc_mean"] = statistics.fmean(ipc_vals) if ipc_vals else None
        row["quality_json"] = json.dumps(dict(quality))
        row["sampling_interval_cv_pct"] = result.metadata.get("sampling_interval_cv_pct")
    else:
        row["postprocess_error"] = "run_failed"
    return row


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))
    entry = catalog[KERNEL_REF]

    # --- interval_ns sweep ---
    interval_rows = []
    for interval_ns in INTERVAL_VALUES_NS:
        sweep_manifest = replace(
            base_manifest,
            output_dir=Path(f"/home/latorresn/hyperion-results/sweeps/interval_ns/interval_{interval_ns}"),
            interval_ns=interval_ns, campaign_id=f"sweep_interval_{interval_ns}",
        )
        for rep in range(1, 4):
            row = run_and_collect(sweep_manifest, entry, rep)
            row.update({"interval_ns": interval_ns, "kernel_ref": KERNEL_REF, "repetition": rep})
            interval_rows.append(row)
            print(f"[interval_ns={interval_ns}] dgemm_n2048 rep{rep}: {row}")

    out1 = Path("/home/latorresn/hyperion-results/sweeps/interval_ns/dgemm_n2048_fix.csv")
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
        row.update({"kernel_ref": KERNEL_REF, "device": "cpu", "repetition": rep, "target_windows_per_repetition": 50})
        row["windows_achieved"] = row.pop("n_windows_ok", None)
        rep_rows.append(row)
        print(f"[repetitions] dgemm_n2048 rep{rep}: {row}")

    out2 = Path("/home/latorresn/hyperion-results/sweeps/repetitions/dgemm_n2048_fix.csv")
    fieldnames = sorted({k for r in rep_rows for k in r})
    with open(out2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rep_rows)
    print(f"Escrito: {out2}")


if __name__ == "__main__":
    main()
