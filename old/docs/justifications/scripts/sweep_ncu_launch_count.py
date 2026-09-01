#!/usr/bin/env python3
"""Sweep F (docs/justifications): convergencia de FLOP/byte vs. numero de
lanzamientos de kernel perfilados con ncu.

Para cada kernel de dataset GPU, perfila con --launch-count creciente
(5, 20, 50) y calcula el FLOP/byte acumulado en cada caso -- si el valor ya
no cambia de forma material entre 20 y 50, el numero de lanzamientos usado
en el catalogo (que historicamente vario sin una regla declarada, ARC-88)
queda justificado por convergencia real, no por eleccion arbitraria.

ARC-110: ya NO usa `entry.gpu_precision` para decidir que contadores pedir
-- recolecta SIEMPRE ambas precisiones (FP32 y FP64) simultaneamente, para
poder detectar un kernel mixto (declarado en una precision pero con
operaciones reales en la otra). `gpu_precision` se usa unicamente DESPUES,
para contrastar la declaracion del catalogo contra lo observado.

No usa el harness (runner.run_single): invoca `ncu` directamente sobre el
binario, igual que las mediciones manuales de esta sesion (ARC-75/76/86).
"""
import sys
import os
import csv
import subprocess
from pathlib import Path

sys.path.insert(0, "/home/latorresn/hyperion")
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator.catalog import load_catalog
from orchestrator.gpu_shim import cuda_lib_dirs
from ncu_gpu_precision import ALL_METRICS, compute_gpu_precision_result, is_mixed_precision, parse_ncu_csv_totals

CATALOG_PATH = "/home/latorresn/hyperion/orchestrator/schemas/kernels/catalog.yaml"
OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/ncu_launch_count")
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
LAUNCH_COUNTS = [5, 20, 50]

GPU_DATASET_KERNELS = [
    "rodinia_hotspot", "rodinia_backprop", "rodinia_lavamd",
    "rodinia_heartwall", "rodinia_lud", "rodinia_myocyte", "rodinia_dwt2d",
]


def _run_ncu(entry, launch_count: int, env: dict) -> str:
    args = entry.exec_args.split() if entry.exec_args else []
    cmd = [
        "ncu", "--metrics", ",".join(ALL_METRICS), "--launch-count", str(launch_count), "--csv",
        entry.exec_path, *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    return result.stdout


def main():
    catalog = load_catalog(CATALOG_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    lib_dirs = cuda_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    rows = []
    mixed_precision_findings = []
    for kernel_ref in GPU_DATASET_KERNELS:
        entry = catalog[kernel_ref]
        for launch_count in LAUNCH_COUNTS:
            raw = _run_ncu(entry, launch_count, env)
            (OUTPUT_ROOT / f"{kernel_ref}_lc{launch_count}.csv").write_text(raw)
            totals, n_launches = parse_ncu_csv_totals(raw)
            result = compute_gpu_precision_result(totals, n_launches)
            mixed = is_mixed_precision(result)
            row = {
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
                "mixed_precision_detected": mixed,
            }
            rows.append(row)
            print(
                f"{kernel_ref} launch_count={launch_count}: "
                f"OI={result.operational_intensity} fp32={result.flops_fp32:.3e} fp64={result.flops_fp64:.3e} "
                f"(catalogo declara {entry.gpu_precision}, OI={entry.operational_intensity_flops_per_byte}) "
                f"{'*** MIXTO ***' if mixed else ''}"
            )
            if mixed:
                mixed_precision_findings.append((kernel_ref, launch_count, result))

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")

    if mixed_precision_findings:
        print("\n" + "=" * 70)
        print("ATENCION: se detectaron kernels con precision mixta (ARC-110, paso 6).")
        print("NO se les asigna un ridge unico automaticamente -- decision pendiente del usuario.")
        print("=" * 70)
        for kernel_ref, launch_count, result in mixed_precision_findings:
            print(
                f"  {kernel_ref} (launch_count={launch_count}): "
                f"fraction_fp32={result.fraction_fp32:.4f} fraction_fp64={result.fraction_fp64:.4f}"
            )


if __name__ == "__main__":
    main()
