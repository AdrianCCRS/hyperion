#!/usr/bin/env python3
"""Sweep B (docs/justifications): gpu_interval_ns (cadencia de muestreo NVML).

Mismo patron que sweep_interval_ns.py pero para el eje de GPU: corre cada
kernel de dataset GPU bajo distintas cadencias de muestreo NVML, mostrando
cuantas muestras utiles (quality_status="gpu_telemetry") se obtienen a cada
cadencia -- la justificacion empirica de por que el default de produccion
(antes 100ms, ahora 5ms tras ARC-88) es necesario para los kernels cortos.
"""
import sys
import os
import csv
import json
import time
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
SWEEP_OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/gpu_interval_ns")
SUMMARY_CSV = SWEEP_OUTPUT_ROOT / "summary.csv"

GPU_INTERVAL_VALUES_NS = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]
GPU_DATASET_KERNELS = [
    "gpu_dgemm_n4096", "rodinia_hotspot", "rodinia_backprop", "rodinia_lavamd",
    "rodinia_heartwall", "rodinia_lud", "rodinia_myocyte", "rodinia_dwt2d",
]
REPETITIONS = 3


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))

    SWEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for gpu_interval_ns in GPU_INTERVAL_VALUES_NS:
        sweep_manifest = replace(
            base_manifest,
            output_dir=SWEEP_OUTPUT_ROOT / f"gpu_interval_{gpu_interval_ns}",
            gpu_interval_ns=gpu_interval_ns,
            campaign_id=f"sweep_gpu_interval_{gpu_interval_ns}",
        )
        Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)

        for kernel_ref in GPU_DATASET_KERNELS:
            entry = catalog[kernel_ref]
            for rep in range(1, REPETITIONS + 1):
                result = runner_module.run_single(
                    entry, sweep_manifest, kernel_ref, "REF", rep, node_id="pacca-a100",
                )
                row = {
                    "gpu_interval_ns": gpu_interval_ns,
                    "kernel_ref": kernel_ref,
                    "repetition": rep,
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
                        gpu_rows = [r for r in wrows if r["quality_status"] == "gpu_telemetry"]
                        row["n_windows_total"] = len(wrows)
                        row["n_gpu_telemetry"] = len(gpu_rows)
                        row["quality_json"] = json.dumps(dict(quality))
                        if gpu_rows:
                            row["phase_label_train"] = gpu_rows[0].get("phase_label_train")
                    except Exception as exc:  # noqa: BLE001
                        row["postprocess_error"] = str(exc)
                else:
                    row["postprocess_error"] = "run_failed"

                rows.append(row)
                print(f"[gpu_interval_ns={gpu_interval_ns}] {kernel_ref} rep{rep}: "
                      f"elapsed={result.elapsed_seconds:.3f}s success={result.success} "
                      f"n_gpu_telemetry={row.get('n_gpu_telemetry')}")

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
