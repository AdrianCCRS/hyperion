#!/usr/bin/env python3
"""Re-parsea los CSV crudos de ncu ya guardados en disco (sin volver a
perfilar) con la clave de metrica corregida para FP32 (ver ARC-89:
sweep_ncu_launch_count.py construia "..._fma_pred_on.sum" para FP32 en vez
de "..._ffma_pred_on.sum")."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
from orchestrator.catalog import load_catalog

CATALOG_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/kernels/catalog.yaml"
OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/ncu_launch_count")
LAUNCH_COUNTS = [5, 20, 50]
GPU_DATASET_KERNELS = [
    "rodinia_hotspot", "rodinia_backprop", "rodinia_lavamd",
    "rodinia_heartwall", "rodinia_lud", "rodinia_myocyte", "rodinia_dwt2d",
]


def _parse_flop_per_byte(csv_text: str, precision: str):
    lines = csv_text.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if header_idx is None:
        return None, 0
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)

    def to_num(v):
        v = v.replace(",", "").strip()
        return float(v) if v not in ("", "N/A") else 0.0

    totals = {}
    for r in rows:
        name = r["Metric Name"]
        totals[name] = totals.get(name, 0.0) + to_num(r["Metric Value"])

    prefix = "d" if precision == "fp64" else "f"
    total_bytes = totals.get("dram__bytes.sum", 0.0)
    fma_key = f"sm__sass_thread_inst_executed_op_{prefix}fma_pred_on.sum"
    add_key = f"sm__sass_thread_inst_executed_op_{prefix}add_pred_on.sum"
    mul_key = f"sm__sass_thread_inst_executed_op_{prefix}mul_pred_on.sum"
    total_flops = 2 * totals.get(fma_key, 0.0) + totals.get(add_key, 0.0) + totals.get(mul_key, 0.0)
    n_launches = len({r["ID"] for r in rows})
    if total_bytes <= 0:
        return None, n_launches
    return total_flops / total_bytes, n_launches


def main():
    catalog = load_catalog(CATALOG_PATH)
    rows = []
    for kernel_ref in GPU_DATASET_KERNELS:
        entry = catalog[kernel_ref]
        for launch_count in LAUNCH_COUNTS:
            raw_path = OUTPUT_ROOT / f"{kernel_ref}_lc{launch_count}.csv"
            raw = raw_path.read_text()
            oi, n_launches = _parse_flop_per_byte(raw, entry.gpu_precision)
            rows.append({
                "kernel_ref": kernel_ref, "requested_launch_count": launch_count,
                "actual_n_launches": n_launches, "operational_intensity": oi,
                "catalog_declared_oi": entry.operational_intensity_flops_per_byte,
            })
            print(f"{kernel_ref} launch_count={launch_count}: OI={oi} (catalogo declara {entry.operational_intensity_flops_per_byte})")

    out_csv = OUTPUT_ROOT / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {out_csv} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
