#!/usr/bin/env python3
"""Sweep G (docs/justifications): distribucion empirica de la calibracion
Roofline (BW_pico/P_pico), repetida N veces, para fundamentar D03_TOLERANCE_FRACTION
(CPU, 40%) y la tolerancia relativa entre niveles de GPU (5%) con una
distribucion real de desviacion en vez de un solo punto.
"""
import sys
import os
import re
import csv
import statistics
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator import manifest as manifest_module
from orchestrator import runner as runner_module
from orchestrator.catalog import load_catalog

BASE_MANIFEST_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/campaign_pacca_gpu_ref.yaml"
SWEEP_OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/calibration_repeats")
SUMMARY_CSV = SWEEP_OUTPUT_ROOT / "summary.csv"
N_REPEATS = 10

CALIBRATION_KERNELS = ["stream_official", "ert_probe", "gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]


def _extract(entry, stdout_text: str):
    pattern = entry.bandwidth_stdout_pattern or entry.flops_stdout_pattern
    if not pattern:
        return None
    match = re.search(pattern, stdout_text)
    if not match:
        return None
    value = float(match.group(1))
    if entry.bandwidth_stdout_pattern:
        value *= entry.bandwidth_stdout_unit_multiplier
    else:
        value *= entry.flops_stdout_unit_multiplier
    return value


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))

    sweep_manifest = replace(
        base_manifest, output_dir=SWEEP_OUTPUT_ROOT / "runs", campaign_id="sweep_calibration_repeats",
    )
    Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for kernel_ref in CALIBRATION_KERNELS:
        entry = catalog[kernel_ref]
        values = []
        for rep in range(1, N_REPEATS + 1):
            result = runner_module.run_single(
                entry, sweep_manifest, kernel_ref, "REF", rep, node_id="pacca-a100",
            )
            value = None
            if result.success:
                stdout_text = Path(result.stdout_path).read_text(errors="replace")
                value = _extract(entry, stdout_text)
            values.append(value)
            rows.append({
                "kernel_ref": kernel_ref, "repetition": rep, "success": result.success, "value": value,
            })
            print(f"{kernel_ref} rep{rep}/{N_REPEATS}: value={value}")

        clean = [v for v in values if v is not None]
        if len(clean) > 1:
            mean = statistics.fmean(clean)
            cv_pct = statistics.pstdev(clean) / mean * 100.0 if mean else None
            print(f"  -> {kernel_ref}: mean={mean:.6e} cv%={cv_pct:.4f}")

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
