#!/usr/bin/env python3
"""Sweep A (docs/justifications): interval_ns (cadencia de muestreo CPU).

Corre cada kernel de dataset CPU del catalogo bajo distintos interval_ns,
usando runner.run_single() directamente (sin la orquestacion completa de
run_campaign: sin calibracion/preflight repetidos por combinacion) para
mantener el costo de computo acotado. Escribe en una carpeta de resultados
SEPARADA de las campanas reales del dataset (hyperion-results/sweeps/, no
hyperion-results/campaigns/), para no contaminarlas.
"""
import sys
import os
import csv
import json
import statistics
import time
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator import manifest as manifest_module
from orchestrator import runner as runner_module
from orchestrator import postprocess as postprocess_module
from orchestrator.catalog import load_catalog

BASE_MANIFEST_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/campaign_pacca_gpu_ref.yaml"
SWEEP_OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/interval_ns")
SUMMARY_CSV = SWEEP_OUTPUT_ROOT / "summary.csv"

INTERVAL_VALUES_NS = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
CPU_DATASET_KERNELS = ["npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu", "dgemm_n2048"]
REPETITIONS = 3


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))

    SWEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for interval_ns in INTERVAL_VALUES_NS:
        sweep_manifest = replace(
            base_manifest,
            output_dir=SWEEP_OUTPUT_ROOT / f"interval_{interval_ns}",
            interval_ns=interval_ns,
            campaign_id=f"sweep_interval_{interval_ns}",
        )
        Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)

        for kernel_ref in CPU_DATASET_KERNELS:
            entry = catalog[kernel_ref]
            for rep in range(1, REPETITIONS + 1):
                t0 = time.monotonic()
                result = runner_module.run_single(
                    entry, sweep_manifest, kernel_ref, "REF", rep, node_id="pacca-a100",
                )
                t1 = time.monotonic()

                row = {
                    "interval_ns": interval_ns,
                    "kernel_ref": kernel_ref,
                    "repetition": rep,
                    "success": result.success,
                    "elapsed_seconds": result.elapsed_seconds,
                    "wall_seconds": t1 - t0,
                    "samples_collected": result.metadata.get("samples_collected"),
                    "sampling_interval_mean_ns": result.metadata.get("sampling_interval_mean_ns"),
                    "sampling_interval_cv_pct": result.metadata.get("sampling_interval_cv_pct"),
                    "push_retries": result.metadata.get("push_retries"),
                    "perf_running_ratio_min": result.metadata.get("perf_running_ratio_min"),
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
                        from collections import Counter
                        quality = Counter(r["quality_status"] for r in wrows)
                        ok_rows = [r for r in wrows if r["quality_status"] == "ok"]
                        ipc_vals = [float(r["ipc"]) for r in ok_rows if r.get("ipc") not in (None, "")]
                        row["n_windows_total"] = len(wrows)
                        row["n_windows_ok"] = len(ok_rows)
                        row["ipc_mean"] = statistics.fmean(ipc_vals) if ipc_vals else None
                        row["ipc_cv_pct"] = (
                            (statistics.pstdev(ipc_vals) / statistics.fmean(ipc_vals) * 100.0)
                            if len(ipc_vals) > 1 and statistics.fmean(ipc_vals) != 0 else None
                        )
                        row["quality_json"] = json.dumps(dict(quality))
                    except Exception as exc:  # noqa: BLE001 -- registrar y seguir con el resto del barrido
                        row["postprocess_error"] = str(exc)
                else:
                    row["postprocess_error"] = "run_failed"

                rows.append(row)
                print(f"[interval_ns={interval_ns}] {kernel_ref} rep{rep}: "
                      f"elapsed={result.elapsed_seconds:.3f}s success={result.success} "
                      f"windows_ok={row.get('n_windows_ok')} cv_interval={row.get('sampling_interval_cv_pct')}")

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
