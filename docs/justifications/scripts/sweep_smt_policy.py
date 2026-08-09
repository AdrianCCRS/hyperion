#!/usr/bin/env python3
"""Sweep H (docs/justifications): smt_policy (all_threads vs. one_thread_per_physical_core).

Compara, sobre el mismo conjunto de 6 nucleos fisicos de paccaA100 (0-5,
con hermanos SMT reales 16-21 confirmados via `lscpu -e`), correr npb_cg
con 6 hilos (un hilo por nucleo fisico, politica vigente en produccion)
contra correrlo con 12 hilos (ambos hermanos SMT de cada nucleo ocupados).
No requiere el permiso de escritura de frecuencia (P1/P4): smt_policy solo
cambia --pin-workload-cpus/OMP_NUM_THREADS via cores.delegated_cpus, y
run_single() no pasa por el preflight de campana completa (_requires_
frequency_control) que bloquea F0-F4 hoy.
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
SWEEP_OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/smt_policy")
SUMMARY_CSV = SWEEP_OUTPUT_ROOT / "summary.csv"

KERNEL_REF = "npb_cg"
REPETITIONS = 5

# Confirmado en paccaA100 via `lscpu -e`: core fisico N (socket 0) = cpu N
# y su hermano SMT es cpu N+16, para N en 0..7.
POLICIES = {
    "one_thread_per_physical_core": [0, 1, 2, 3, 4, 5],
    "all_threads": [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21],
}


def main():
    base_manifest = manifest_module.load(BASE_MANIFEST_PATH)
    catalog = load_catalog(str(base_manifest.catalog_path))
    entry = catalog[KERNEL_REF]

    SWEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for policy_name, cpu_list in POLICIES.items():
        sweep_manifest = replace(
            base_manifest,
            output_dir=SWEEP_OUTPUT_ROOT / policy_name,
            smt_policy=policy_name,
            cores=replace(base_manifest.cores, delegated_cpus=tuple(cpu_list)),
            campaign_id=f"sweep_smt_{policy_name}",
        )
        Path(sweep_manifest.output_dir).mkdir(parents=True, exist_ok=True)

        for rep in range(1, REPETITIONS + 1):
            t0 = time.monotonic()
            result = runner_module.run_single(
                entry, sweep_manifest, KERNEL_REF, "REF", rep, node_id="pacca-a100",
            )
            t1 = time.monotonic()

            row = {
                "smt_policy": policy_name,
                "n_threads": len(cpu_list),
                "cpu_list": json.dumps(cpu_list),
                "kernel_ref": KERNEL_REF,
                "repetition": rep,
                "success": result.success,
                "elapsed_seconds": result.elapsed_seconds,
                "wall_seconds": t1 - t0,
            }

            if result.success:
                try:
                    windows_path = postprocess_module.run_postprocess(
                        result.run_dir, run_id=result.run_id, repetition=rep, kernel_ref=KERNEL_REF,
                        kernel_entry=entry, node_id="pacca-a100", freq_level_id="REF",
                        calibration_dir=str(base_manifest.output_dir),
                        freq_khz_observed=2261000, warmup_seconds=entry.warmup_seconds or 0.0,
                        running_ratio_min=sweep_manifest.running_ratio_min, rapl_enabled=False,
                    )
                    with open(windows_path, newline="") as f:
                        wrows = list(csv.DictReader(f))
                    ok_rows = [r for r in wrows if r["quality_status"] == "ok"]
                    ipc_vals = [float(r["ipc"]) for r in ok_rows if r.get("ipc") not in (None, "")]
                    mpki_vals = [float(r["mpki"]) for r in ok_rows if r.get("mpki") not in (None, "")]
                    llc_vals = [float(r["llc_miss_rate"]) for r in ok_rows if r.get("llc_miss_rate") not in (None, "")]
                    row["n_windows_total"] = len(wrows)
                    row["n_windows_ok"] = len(ok_rows)
                    row["ipc_mean"] = statistics.fmean(ipc_vals) if ipc_vals else None
                    row["mpki_mean"] = statistics.fmean(mpki_vals) if mpki_vals else None
                    row["llc_miss_rate_mean"] = statistics.fmean(llc_vals) if llc_vals else None
                except Exception as exc:  # noqa: BLE001 -- registrar y seguir con el resto del barrido
                    row["postprocess_error"] = str(exc)
            else:
                row["postprocess_error"] = "run_failed"

            rows.append(row)
            print(f"[{policy_name}] rep{rep}: elapsed={result.elapsed_seconds:.3f}s "
                  f"success={result.success} ipc_mean={row.get('ipc_mean')}")

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
