#!/usr/bin/env python3
"""Reanalisis del Sweep C/D: en vez de tomar solo el CV%% de las primeras 3
corridas (que depende del orden fijo en que se ejecutaron), calcula el CV%%
de tiempo de ejecucion sobre TODAS las C(10,3)=120 combinaciones posibles de
3 corridas por kernel, y reporta la distribucion completa (min/mediana/p95/
max), no solo el valor que dio la muestra de orden fijo.

No requiere computo nuevo en pacca: reusa los mismos datos ya recolectados
por sweep_repetitions.py (docs/justifications/data/repetitions/).
"""
import csv
import itertools
import re
import statistics
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "repetitions"
OUT_CSV = Path(__file__).resolve().parent.parent / "plots" / "repetitions" / "table_combinatoric_cv3.csv"

_RUN_DIR_RE = re.compile(r"sweep_repetitions__([a-zA-Z0-9_]+)__REF__rep(\d+)")


def cv_pct(values):
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / abs(mean) * 100.0


def _load_rows():
    with open(DATA_DIR / "summary.csv", newline="") as f:
        base_rows = list(csv.DictReader(f))
    rows = [r for r in base_rows if r["success"] == "True" and r["kernel_ref"] != "rodinia_heartwall"]
    fix_path = DATA_DIR / "dgemm_n2048_fix.csv"
    if fix_path.exists():
        with open(fix_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    hw_fix_path = DATA_DIR / "rodinia_heartwall_fix.csv"
    if hw_fix_path.exists():
        with open(hw_fix_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    rows = _load_rows()
    by_kernel = defaultdict(list)
    device_by_kernel = {}
    for r in rows:
        by_kernel[r["kernel_ref"]].append((int(r["repetition"]), float(r["elapsed_seconds"])))
        device_by_kernel[r["kernel_ref"]] = r["device"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table = []
    for kernel in sorted(by_kernel):
        entries = sorted(by_kernel[kernel])
        elapsed = [e[1] for e in entries]
        n = len(elapsed)
        if n < 3:
            continue
        combo_cvs = [cv_pct(list(c)) for c in itertools.combinations(elapsed, 3)]
        first3_cv = cv_pct(elapsed[:3])
        combo_cvs_sorted = sorted(combo_cvs)
        p95_idx = min(len(combo_cvs_sorted) - 1, round(0.95 * (len(combo_cvs_sorted) - 1)))
        row = {
            "kernel_ref": kernel,
            "device": device_by_kernel[kernel],
            "n_combinations": len(combo_cvs),
            "cv3_first_order": round(first3_cv, 3),
            "cv3_min": round(combo_cvs_sorted[0], 3),
            "cv3_median": round(statistics.median(combo_cvs_sorted), 3),
            "cv3_p95": round(combo_cvs_sorted[p95_idx], 3),
            "cv3_max": round(combo_cvs_sorted[-1], 3),
            "cv10_all": round(cv_pct(elapsed), 3),
        }
        table.append(row)
        print(f"{kernel:20s} first3={row['cv3_first_order']:6.2f}%  "
              f"median={row['cv3_median']:6.2f}%  p95={row['cv3_p95']:6.2f}%  "
              f"max={row['cv3_max']:6.2f}%  cv10={row['cv10_all']:6.2f}%")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\nEscrito: {OUT_CSV}")


if __name__ == "__main__":
    main()
