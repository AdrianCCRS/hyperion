#!/usr/bin/env python3
"""Confirmacion aleatorizada e intercalada de los casos frontera del
informe (ARC-90 -> ARC-91): los barridos originales (interval_ns,
gpu_interval_ns) corren TODAS las repeticiones de un valor de parametro
antes de pasar al siguiente ("primero todo 0.5ms, luego todo 1ms, etc."),
sin aleatorizacion ni control de carga externa del nodo compartido -- una
deriva temporal del nodo podria confundirse con el efecto del parametro.

Este script corre una confirmacion pequena e intercalada al azar (no
reemplaza los barridos principales, solo verifica que sus conclusiones se
sostienen bajo un orden distinto) sobre los dos casos frontera mas citados
del informe:
  - interval_ns: 0.5ms vs 1ms (npb_cg, npb_mg, npb_bt), 3 reps c/u,
    orden aleatorio -- se espera que sampling_interval_cv_pct siga
    mostrando el mismo patron (0.5ms con cola de jitter mas larga).
  - gpu_interval_ns: 50ms vs 100ms (rodinia_backprop), 3 reps c/u,
    orden aleatorio -- se espera n_gpu_telemetry ~17 y ~9 respectivamente,
    consistente con el barrido original no aleatorizado.
"""
import sys
import os
import csv
import json
import random
import statistics
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator import manifest as manifest_module
from orchestrator import runner as runner_module
from orchestrator import postprocess as postprocess_module
from orchestrator.catalog import load_catalog

BASE_MANIFEST_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/campaign_pacca_gpu_ref.yaml"
SWEEP_OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/randomized_confirmation")
SUMMARY_CSV = SWEEP_OUTPUT_ROOT / "summary.csv"

RNG_SEED = 20260808
REPETITIONS = 3

CPU_KERNELS = ["npb_cg", "npb_mg", "npb_bt"]
CPU_INTERVALS_NS = [500_000, 1_000_000]

GPU_KERNEL = "rodinia_backprop"
GPU_INTERVALS_NS = [50_000_000, 100_000_000]


def run_cpu_confirmation(base_manifest, catalog, rows):
    rng = random.Random(RNG_SEED)
    plan = []
    for kernel_ref in CPU_KERNELS:
        for interval_ns in CPU_INTERVALS_NS:
            for rep in range(1, REPETITIONS + 1):
                plan.append((kernel_ref, interval_ns, rep))
    rng.shuffle(plan)
    print("Orden aleatorio CPU:", plan)

    counters = {}
    for kernel_ref, interval_ns, rep in plan:
        entry = catalog[kernel_ref]
        key = (kernel_ref, interval_ns)
        counters[key] = counters.get(key, 0) + 1
        seq = counters[key]
        sweep_manifest = replace(
            base_manifest,
            output_dir=SWEEP_OUTPUT_ROOT / "cpu" / f"interval_{interval_ns}",
            interval_ns=interval_ns,
            campaign_id=f"confirm_cpu_{interval_ns}",
        )
        Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)
        result = runner_module.run_single(
            entry, sweep_manifest, kernel_ref, "REF", seq, node_id="pacca-a100",
        )
        row = {
            "domain": "cpu", "interval_ns": interval_ns, "kernel_ref": kernel_ref,
            "repetition": seq, "success": result.success,
            "sampling_interval_cv_pct": result.metadata.get("sampling_interval_cv_pct"),
        }
        rows.append(row)
        print(f"[randomized cpu] {kernel_ref} interval={interval_ns} seq={seq}: "
              f"success={result.success} cv_interval={row['sampling_interval_cv_pct']}")


def run_gpu_confirmation(base_manifest, catalog, rows):
    rng = random.Random(RNG_SEED + 1)
    plan = []
    for interval_ns in GPU_INTERVALS_NS:
        for rep in range(1, REPETITIONS + 1):
            plan.append((interval_ns, rep))
    rng.shuffle(plan)
    print("Orden aleatorio GPU:", plan)

    entry = catalog[GPU_KERNEL]
    counters = {}
    for interval_ns, rep in plan:
        counters[interval_ns] = counters.get(interval_ns, 0) + 1
        seq = counters[interval_ns]
        sweep_manifest = replace(
            base_manifest,
            output_dir=SWEEP_OUTPUT_ROOT / "gpu" / f"gpu_interval_{interval_ns}",
            gpu_interval_ns=interval_ns,
            campaign_id=f"confirm_gpu_{interval_ns}",
        )
        Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)
        result = runner_module.run_single(
            entry, sweep_manifest, GPU_KERNEL, "REF", seq, node_id="pacca-a100",
        )
        row = {"domain": "gpu", "interval_ns": interval_ns, "kernel_ref": GPU_KERNEL,
               "repetition": seq, "success": result.success}
        if result.success:
            try:
                windows_path = postprocess_module.run_postprocess(
                    result.run_dir, run_id=result.run_id, repetition=seq, kernel_ref=GPU_KERNEL,
                    kernel_entry=entry, node_id="pacca-a100", freq_level_id="REF",
                    calibration_dir=str(base_manifest.output_dir),
                    freq_khz_observed=2261000, warmup_seconds=entry.warmup_seconds or 0.0,
                    running_ratio_min=sweep_manifest.running_ratio_min, rapl_enabled=False,
                )
                with open(windows_path, newline="") as f:
                    wrows = list(csv.DictReader(f))
                gpu_rows = [r for r in wrows if r["quality_status"] == "gpu_telemetry"]
                row["n_gpu_telemetry"] = len(gpu_rows)
            except Exception as exc:  # noqa: BLE001
                row["postprocess_error"] = str(exc)
        rows.append(row)
        print(f"[randomized gpu] interval={interval_ns} seq={seq}: "
              f"success={result.success} n_gpu_telemetry={row.get('n_gpu_telemetry')}")


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))
    SWEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    run_cpu_confirmation(base_manifest, catalog, rows)
    run_gpu_confirmation(base_manifest, catalog, rows)

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
