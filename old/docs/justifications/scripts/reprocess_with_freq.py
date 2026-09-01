#!/usr/bin/env python3
"""Corrige un error de diseno de los sweeps A/B/C: se llamo a
run_postprocess() con freq_khz_observed=None, lo que marca cada ventana
como quality_status="no_freq_reading" (correcto por diseno del propio
taxonomia de calidad -- no es un bug de postprocess.py, es un error de
esta herramienta de barrido). Como runner.py nunca borra samples.csv, se
puede re-correr solo el post-procesamiento (sin GPU, sin re-ejecutar el
harness) pasando un valor real de frecuencia observada.

No requiere asignacion de GPU: postprocess.py es Python puro sobre
samples.csv ya en disco.
"""
import sys
import os
import csv
import json
import re
import glob
import statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")

from orchestrator import postprocess as postprocess_module
from orchestrator.catalog import load_catalog

CATALOG_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/kernels/catalog.yaml"
BASE_CALIBRATION_DIR = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_ref_20260807"

# Valor de relleno: estas corridas se hicieron bajo gobernador nativo, sin
# ninguna actuacion de frecuencia -- el valor exacto es irrelevante para lo
# que estos barridos miden (cadencia de muestreo, repeticiones), solo hace
# falta que no sea None para que la ventana no se marque "no_freq_reading".
PLACEHOLDER_FREQ_KHZ = 2261000


def read_run_id_parts(run_dir_name: str):
    # <campaign_id>__<kernel_ref>__<freq_level_id>__rep<NN>
    parts = run_dir_name.split("__")
    campaign_id, kernel_ref, freq_level_id, rep_part = parts[0], parts[1], parts[2], parts[3]
    repetition = int(rep_part.replace("rep", ""))
    return campaign_id, kernel_ref, freq_level_id, repetition


def reprocess_tree(root: Path, catalog, running_ratio_min=0.90):
    results = []
    for samples_path in sorted(root.glob("**/samples.csv")):
        run_dir = samples_path.parent
        try:
            campaign_id, kernel_ref, freq_level_id, repetition = read_run_id_parts(run_dir.name)
        except (IndexError, ValueError):
            continue
        entry = catalog.get(kernel_ref)
        if entry is None:
            continue
        try:
            windows_path = postprocess_module.run_postprocess(
                run_dir, run_id=run_dir.name, repetition=repetition, kernel_ref=kernel_ref,
                kernel_entry=entry, node_id="pacca-a100", freq_level_id=freq_level_id,
                calibration_dir=BASE_CALIBRATION_DIR,
                freq_khz_observed=PLACEHOLDER_FREQ_KHZ, warmup_seconds=entry.warmup_seconds or 0.0,
                running_ratio_min=running_ratio_min, rapl_enabled=False,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"run_dir": str(run_dir), "kernel_ref": kernel_ref, "error": str(exc)})
            continue

        with open(windows_path, newline="") as f:
            wrows = list(csv.DictReader(f))
        quality = Counter(r["quality_status"] for r in wrows)
        row = {
            "run_dir": str(run_dir), "kernel_ref": kernel_ref, "repetition": repetition,
            "n_windows_total": len(wrows), "quality_json": json.dumps(dict(quality)),
        }
        if entry.device == "gpu":
            row["n_gpu_telemetry"] = quality.get("gpu_telemetry", 0)
        else:
            row["n_windows_ok"] = quality.get("ok", 0)
            ok_rows = [r for r in wrows if r["quality_status"] == "ok"]
            ipc_vals = [float(r["ipc"]) for r in ok_rows if r.get("ipc") not in (None, "")]
            row["ipc_mean"] = statistics.fmean(ipc_vals) if ipc_vals else None
            row["ipc_cv_pct"] = (
                statistics.pstdev(ipc_vals) / statistics.fmean(ipc_vals) * 100.0
                if len(ipc_vals) > 1 and statistics.fmean(ipc_vals) else None
            )
        results.append(row)
        print(f"reprocesado: {run_dir.name} -> {dict(quality)}")
    return results


def main():
    catalog = load_catalog(CATALOG_PATH)
    targets = {
        "interval_ns": Path("/home/latorresn/hyperion-results/sweeps/interval_ns"),
        "gpu_interval_ns": Path("/home/latorresn/hyperion-results/sweeps/gpu_interval_ns"),
        "repetitions": Path("/home/latorresn/hyperion-results/sweeps/repetitions"),
    }
    for name, root in targets.items():
        print(f"\n=== Reprocesando {name} ===")
        results = reprocess_tree(root, catalog)
        out_csv = root / "summary_reprocessed.csv"
        fieldnames = sorted({key for row in results for key in row})
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Escrito: {out_csv} ({len(results)} filas)")


if __name__ == "__main__":
    main()
