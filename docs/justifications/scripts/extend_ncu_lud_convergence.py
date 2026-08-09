#!/usr/bin/env python3
"""Extension del Sweep F para rodinia_lud unicamente: el critico de ARC-90
observo correctamente que confirmar "no convergio a 50" no implica que 200
(el valor ya usado en el catalogo, ARC-80) sea suficiente -- solo que 50 es
insuficiente. Esta extension perfila rodinia_lud con --launch-count
100, 150 y 200 (unico kernel, 3 corridas de ncu adicionales, costo bajo) y
aplica un criterio de convergencia declarado ANTES de ver el resultado:
convergencia = cambio relativo < 1% entre cada par consecutivo de puntos.
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
KERNEL_REF = "rodinia_lud"
LAUNCH_COUNTS = [100, 150, 200]
CONVERGENCE_THRESHOLD_PCT = 1.0

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


def _run_ncu(entry, launch_count, env):
    args = entry.exec_args.split() if entry.exec_args else []
    metrics = ",".join(METRICS_BY_PRECISION[entry.gpu_precision])
    cmd = [
        "ncu", "--metrics", metrics, "--launch-count", str(launch_count), "--csv",
        entry.exec_path, *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    return result.stdout


def _parse_flop_per_byte(csv_text, precision):
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
    entry = catalog[KERNEL_REF]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    lib_dirs = cuda_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    # Puntos previos (5, 20, 50) ya medidos en el Sweep F original -- se
    # reincorporan aqui solo para el reporte de convergencia continuo,
    # tomados del summary.csv ya existente, sin volver a perfilar.
    prior = {}
    prior_csv = OUTPUT_ROOT / "summary.csv"
    if prior_csv.exists():
        with open(prior_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r["kernel_ref"] == KERNEL_REF:
                    prior[int(r["requested_launch_count"])] = float(r["operational_intensity"])

    points = dict(prior)
    for launch_count in LAUNCH_COUNTS:
        raw = _run_ncu(entry, launch_count, env)
        (OUTPUT_ROOT / f"{KERNEL_REF}_lc{launch_count}.csv").write_text(raw)
        oi, n_launches = _parse_flop_per_byte(raw, entry.gpu_precision)
        points[launch_count] = oi
        print(f"{KERNEL_REF} launch_count={launch_count}: OI={oi} (n_launches reales={n_launches})")

    ordered = sorted(points.items())
    rows = []
    converged_at = None
    for i, (lc, oi) in enumerate(ordered):
        rel_change_pct = None
        if i > 0 and ordered[i - 1][1]:
            rel_change_pct = abs(oi - ordered[i - 1][1]) / ordered[i - 1][1] * 100.0
            if rel_change_pct < CONVERGENCE_THRESHOLD_PCT and converged_at is None:
                converged_at = lc
        rows.append({"launch_count": lc, "operational_intensity": oi, "rel_change_pct": rel_change_pct})

    out_csv = OUTPUT_ROOT / "rodinia_lud_convergence_extended.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["launch_count", "operational_intensity", "rel_change_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPuntos: {ordered}")
    print(f"Criterio declarado: cambio relativo < {CONVERGENCE_THRESHOLD_PCT}% entre puntos consecutivos")
    print(f"Convergencia alcanzada en launch_count={converged_at}" if converged_at else "NO converge dentro del rango medido (5..200)")
    print(f"Escrito: {out_csv}")


if __name__ == "__main__":
    main()
