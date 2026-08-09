#!/usr/bin/env python3
"""Sweep C/D (docs/justifications): convergencia de repetitions_per_combination
y suficiencia de target_windows_per_repetition.

Corre cada kernel de dataset (CPU+GPU, catalogo completo) 10 veces bajo la
cadencia por defecto (no se toca interval_ns/gpu_interval_ns aqui, ya
cubierto en los sweeps A/B) -- el analisis de convergencia de CV% en
funcion de n (3,4,...,10 repeticiones) y de ventanas logradas vs. el
objetivo configurado se hace despues, en Python puro, sobre este mismo CSV
(no hace falta una corrida nueva por cada n).
"""
import sys
import os
import csv
import json
import statistics
from pathlib import Path
from dataclasses import replace
from collections import Counter

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator import manifest as manifest_module
from orchestrator import runner as runner_module
from orchestrator import postprocess as postprocess_module
from orchestrator.catalog import load_catalog

BASE_MANIFEST_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/campaign_pacca_gpu_ref.yaml"
SWEEP_OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/repetitions")
SUMMARY_CSV = SWEEP_OUTPUT_ROOT / "summary.csv"

DATASET_KERNELS = [
    "npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu", "dgemm_n2048",
    "gpu_dgemm_n4096", "rodinia_hotspot", "rodinia_backprop", "rodinia_lavamd",
    "rodinia_heartwall", "rodinia_lud", "rodinia_myocyte", "rodinia_dwt2d",
]
N_REPETITIONS = 10


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))

    sweep_manifest = replace(
        base_manifest,
        output_dir=SWEEP_OUTPUT_ROOT / "runs",
        campaign_id="sweep_repetitions",
    )
    Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for kernel_ref in DATASET_KERNELS:
        entry = catalog[kernel_ref]
        target_windows = 5 if entry.device == "gpu" else 50
        for rep in range(1, N_REPETITIONS + 1):
            result = runner_module.run_single(
                entry, sweep_manifest, kernel_ref, "REF", rep, node_id="pacca-a100",
            )
            row = {
                "kernel_ref": kernel_ref,
                "device": entry.device,
                "repetition": rep,
                "target_windows_per_repetition": target_windows,
                "success": result.success,
                "elapsed_seconds": result.elapsed_seconds,
            }
            if result.success:
                try:
                    windows_path = postprocess_module.run_postprocess(
                        result.run_dir, run_id=result.run_id, repetition=rep, kernel_ref=kernel_ref,
                        kernel_entry=entry, node_id="pacca-a100", freq_level_id="REF",
                        calibration_dir=str(base_manifest.output_dir),
                        freq_khz_observed=None, warmup_seconds=entry.warmup_seconds or 0.0,
                        running_ratio_min=sweep_manifest.running_ratio_min, rapl_enabled=False,
                    )
                    with open(windows_path, newline="") as f:
                        wrows = list(csv.DictReader(f))
                    quality = Counter(r["quality_status"] for r in wrows)
                    if entry.device == "gpu":
                        achieved = quality.get("gpu_telemetry", 0)
                    else:
                        achieved = quality.get("ok", 0)
                        ok_rows = [r for r in wrows if r["quality_status"] == "ok"]
                        ipc_vals = [float(r["ipc"]) for r in ok_rows if r.get("ipc") not in (None, "")]
                        row["ipc_mean"] = statistics.fmean(ipc_vals) if ipc_vals else None
                    row["windows_achieved"] = achieved
                    row["quality_json"] = json.dumps(dict(quality))
                except Exception as exc:  # noqa: BLE001
                    row["postprocess_error"] = str(exc)
            else:
                row["postprocess_error"] = "run_failed"

            rows.append(row)
            print(f"{kernel_ref} rep{rep}/{N_REPETITIONS}: elapsed={result.elapsed_seconds:.3f}s "
                  f"success={result.success} windows_achieved={row.get('windows_achieved')} "
                  f"(target={target_windows})")

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
