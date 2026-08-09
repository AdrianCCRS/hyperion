#!/usr/bin/env python3
"""Sweep F (docs/justifications): convergencia de FLOP/byte vs. numero de
lanzamientos de kernel perfilados con ncu.

Para cada kernel de dataset GPU, perfila con --launch-count creciente
(5, 20, 50) y calcula el FLOP/byte acumulado en cada caso -- si el valor ya
no cambia de forma material entre 20 y 50, el numero de lanzamientos usado
en el catalogo (que historicamente vario sin una regla declarada, ARC-88)
queda justificado por convergencia real, no por eleccion arbitraria.

No usa el harness (runner.run_single): invoca `ncu` directamente sobre el
binario, igual que las mediciones manuales de esta sesion (ARC-75/76/86).
"""
import sys
import os
import csv
import subprocess
from pathlib import Path

sys.path.insert(0, "/home/latorresn/hyperion-gpu-fase1")
os.chdir("/home/latorresn/hyperion-kernels")

from orchestrator.catalog import load_catalog
from orchestrator.gpu_shim import cuda_lib_dirs

CATALOG_PATH = "/home/latorresn/hyperion-gpu-fase1/orchestrator/schemas/kernels/catalog.yaml"
OUTPUT_ROOT = Path("/home/latorresn/hyperion-results/sweeps/ncu_launch_count")
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
LAUNCH_COUNTS = [5, 20, 50]

GPU_DATASET_KERNELS = [
    "rodinia_hotspot", "rodinia_backprop", "rodinia_lavamd",
    "rodinia_heartwall", "rodinia_lud", "rodinia_myocyte", "rodinia_dwt2d",
]

METRICS_BY_PRECISION = {
    "fp32": (
        "dram__bytes.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    ),
    "fp64": (
        "dram__bytes.sum",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
    ),
}


def _run_ncu(entry, launch_count: int, env: dict) -> str:
    args = entry.exec_args.split() if entry.exec_args else []
    metrics = ",".join(METRICS_BY_PRECISION[entry.gpu_precision])
    cmd = [
        "ncu", "--metrics", metrics, "--launch-count", str(launch_count), "--csv",
        entry.exec_path, *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    return result.stdout


def _parse_flop_per_byte(csv_text: str, precision: str) -> tuple[float | None, int]:
    lines = csv_text.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if header_idx is None:
        return None, 0
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)

    def to_num(v):
        v = v.replace(",", "").strip()
        return float(v) if v not in ("", "N/A") else 0.0

    totals: dict[str, float] = {}
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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    lib_dirs = cuda_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    rows = []
    for kernel_ref in GPU_DATASET_KERNELS:
        entry = catalog[kernel_ref]
        for launch_count in LAUNCH_COUNTS:
            raw = _run_ncu(entry, launch_count, env)
            (OUTPUT_ROOT / f"{kernel_ref}_lc{launch_count}.csv").write_text(raw)
            oi, n_launches = _parse_flop_per_byte(raw, entry.gpu_precision)
            row = {
                "kernel_ref": kernel_ref, "requested_launch_count": launch_count,
                "actual_n_launches": n_launches, "operational_intensity": oi,
                "catalog_declared_oi": entry.operational_intensity_flops_per_byte,
            }
            rows.append(row)
            print(f"{kernel_ref} launch_count={launch_count}: OI={oi} (catalogo declara {entry.operational_intensity_flops_per_byte})")

    fieldnames = sorted({key for row in rows for key in row})
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {SUMMARY_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
