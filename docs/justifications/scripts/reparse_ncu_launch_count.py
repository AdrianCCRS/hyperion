#!/usr/bin/env python3
"""Re-parsea los CSV crudos de ncu ya guardados en disco (sin volver a
perfilar) con la logica compartida de ncu_gpu_precision.py.

ARC-110: los CSV generados por versiones ANTERIORES a esta correccion
solo contienen los contadores de UNA precision (la que `gpu_precision`
declaraba en el catalogo) -- este script ya no puede recuperar la
precision faltante de esos archivos viejos, porque `ncu` nunca la
recolecto en primer lugar. Solo tiene sentido reparsear CSV generados por
la version corregida de sweep_ncu_launch_count.py (que pide ambas
precisiones simultaneamente).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/home/latorresn/hyperion")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrator.catalog import load_catalog
from ncu_gpu_precision import compute_gpu_precision_result, is_mixed_precision, parse_ncu_csv_totals

CATALOG_PATH = "/home/latorresn/hyperion/orchestrator/schemas/kernels/catalog.yaml"
OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/ncu_launch_count")
LAUNCH_COUNTS = [5, 20, 50]
GPU_DATASET_KERNELS = [
    "rodinia_hotspot", "rodinia_backprop", "rodinia_lavamd",
    "rodinia_heartwall", "rodinia_lud", "rodinia_myocyte", "rodinia_dwt2d",
]


def main():
    catalog = load_catalog(CATALOG_PATH)
    rows = []
    for kernel_ref in GPU_DATASET_KERNELS:
        entry = catalog[kernel_ref]
        for launch_count in LAUNCH_COUNTS:
            raw_path = OUTPUT_ROOT / f"{kernel_ref}_lc{launch_count}.csv"
            raw = raw_path.read_text()
            totals, n_launches = parse_ncu_csv_totals(raw)
            result = compute_gpu_precision_result(totals, n_launches)
            rows.append({
                "kernel_ref": kernel_ref,
                "requested_launch_count": launch_count,
                "actual_n_launches": n_launches,
                "flops_fp32": result.flops_fp32,
                "flops_fp64": result.flops_fp64,
                "flops_total": result.flops_total,
                "dram_bytes": result.dram_bytes,
                "fraction_fp32": result.fraction_fp32,
                "fraction_fp64": result.fraction_fp64,
                "operational_intensity": result.operational_intensity,
                "catalog_declared_gpu_precision": entry.gpu_precision,
                "catalog_declared_oi": entry.operational_intensity_flops_per_byte,
                "mixed_precision_detected": is_mixed_precision(result),
            })
            print(
                f"{kernel_ref} launch_count={launch_count}: OI={result.operational_intensity} "
                f"(catalogo declara {entry.gpu_precision}, OI={entry.operational_intensity_flops_per_byte})"
            )

    out_csv = OUTPUT_ROOT / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {out_csv} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
