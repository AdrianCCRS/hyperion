#!/usr/bin/env python3
"""Extension del Sweep F para rodinia_lud unicamente: el critico de ARC-90
observo correctamente que confirmar "no convergio a 50" no implica que 200
(el valor ya usado en el catalogo, ARC-80) sea suficiente -- solo que 50 es
insuficiente. Esta extension perfila rodinia_lud con --launch-count
100, 150 y 200 (unico kernel, 3 corridas de ncu adicionales, costo bajo) y
aplica un criterio de convergencia declarado ANTES de ver el resultado:
convergencia = cambio relativo < 1% entre cada par consecutivo de puntos.

ARC-110: ya NO usa `entry.gpu_precision` para elegir contadores -- pide
ambas precisiones simultaneamente (rodinia_lud es el kernel mas cercano al
ridge de todo el catalogo, cf. ARC-76/89 -- si hubiera mezcla de
precision, es aqui donde mas importaria).
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
KERNEL_REF = "rodinia_lud"
LAUNCH_COUNTS = [100, 150, 200]
CONVERGENCE_THRESHOLD_PCT = 1.0


def _run_ncu(entry, launch_count, env):
    args = entry.exec_args.split() if entry.exec_args else []
    cmd = [
        "ncu", "--metrics", ",".join(ALL_METRICS), "--launch-count", str(launch_count), "--csv",
        entry.exec_path, *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    return result.stdout


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
    # tomados del summary.csv ya existente, sin volver a perfilar. Solo
    # validos si vienen de la version corregida del sweep (con
    # operational_intensity ya calculado con ambas precisiones).
    prior = {}
    prior_csv = OUTPUT_ROOT / "summary.csv"
    if prior_csv.exists():
        with open(prior_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r["kernel_ref"] == KERNEL_REF and r.get("operational_intensity"):
                    prior[int(r["requested_launch_count"])] = float(r["operational_intensity"])

    points = dict(prior)
    mixed_findings = []
    for launch_count in LAUNCH_COUNTS:
        raw = _run_ncu(entry, launch_count, env)
        (OUTPUT_ROOT / f"{KERNEL_REF}_lc{launch_count}.csv").write_text(raw)
        totals, n_launches = parse_ncu_csv_totals(raw)
        result = compute_gpu_precision_result(totals, n_launches)
        points[launch_count] = result.operational_intensity
        mixed = is_mixed_precision(result)
        print(
            f"{KERNEL_REF} launch_count={launch_count}: OI={result.operational_intensity} "
            f"(n_launches reales={n_launches}, fp32={result.flops_fp32:.3e}, fp64={result.flops_fp64:.3e}) "
            f"{'*** MIXTO ***' if mixed else ''}"
        )
        if mixed:
            mixed_findings.append((launch_count, result))

    ordered = sorted((lc, oi) for lc, oi in points.items() if oi is not None)
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

    if mixed_findings:
        print("\n" + "=" * 70)
        print(f"ATENCION: {KERNEL_REF} muestra precision mixta (ARC-110, paso 6).")
        print("Kernel cercano al ridge -- una mezcla real aqui SI puede cambiar la clasificacion.")
        print("NO se le asigna un ridge unico automaticamente -- decision pendiente del usuario.")
        print("=" * 70)
        for launch_count, result in mixed_findings:
            print(
                f"  launch_count={launch_count}: fraction_fp32={result.fraction_fp32:.4f} "
                f"fraction_fp64={result.fraction_fp64:.4f}"
            )


if __name__ == "__main__":
    main()
