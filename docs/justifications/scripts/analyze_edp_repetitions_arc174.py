#!/usr/bin/env python3
"""Convergencia de CV%% de EDP (energia x tiempo) en funcion de n
repeticiones, sobre los 9 kernels x 6 niveles de la campana CPU final ya
corregida (arc174).

Cierra la laguna que `repetitions.tex` (2026-08-14) dejo declarada
explicitamente: aquel analisis solo caracterizo convergencia de TIEMPO de
ejecucion, nunca de energia/EDP -- "no hay evidencia todavia de que tres
repeticiones sean suficientes para estimar el EDP con la misma confianza".
La decision de usar 10 repeticiones en el dataset final tampoco vino de
un criterio de convergencia: fue "el maximo n evaluado, con el costo
aceptado explicitamente" (misma fuente).

Este script SI usa el dataset final real (9 kernels x 6 niveles, no el
sweep dedicado de 1 sola frecuencia de aquel informe), asi que cubre
ademas la variacion por nivel de frecuencia que aquel nunca cubrio.

Misma convencion que analyze_repetitions_combinatoric.py: CV%% con
desviacion POBLACIONAL (pstdev), orden fijo = el orden real en que
corrieron las repeticiones (rep01..rep10), mas la distribucion completa
sobre C(10,3) para no depender de la suerte de cual trio se mira.
"""
from __future__ import annotations

import csv
import itertools
import statistics
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "repetitions" / "arc174_edp_per_rep.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "repetitions"


def cv_pct(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / abs(mean) * 100.0


def load_edp_by_cell() -> dict[tuple[str, str], list[tuple[int, float]]]:
    cells: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with DATA_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            edp = float(row["elapsed_s"]) * float(row["energy_j"])
            cells[(row["kernel_ref"], row["freq_level_id"])].append(
                (int(row["repetition"]), edp)
            )
    return cells


def main() -> int:
    cells = load_edp_by_cell()
    print(f"celdas (kernel, nivel) con n=10: {sum(1 for v in cells.values() if len(v) == 10)} de {len(cells)}")
    print()

    rows = []
    for (kernel, level), entries in sorted(cells.items()):
        entries.sort()
        edp_values = [e[1] for e in entries]
        n = len(edp_values)
        if n < 3:
            continue

        # Convergencia por orden fijo, n=3..10 acumulado.
        fixed_order_cv = {k: cv_pct(edp_values[:k]) for k in range(3, n + 1)}

        # Distribucion combinatoria a n=3 (todas las C(n,3) posibles).
        combo_cvs = sorted(cv_pct(list(c)) for c in itertools.combinations(edp_values, 3))
        p95_idx = min(len(combo_cvs) - 1, round(0.95 * (len(combo_cvs) - 1)))

        rows.append({
            "kernel_ref": kernel,
            "freq_level_id": level,
            "cv3_fixed": fixed_order_cv[3],
            "cv3_median": statistics.median(combo_cvs),
            "cv3_p95": combo_cvs[p95_idx],
            "cv3_max": combo_cvs[-1],
            "cv6_fixed": fixed_order_cv.get(6, float("nan")),
            "cv10_all": cv_pct(edp_values),
        })

    rows.sort(key=lambda r: -r["cv3_p95"])

    header = f"{'kernel':<30}{'nivel':<6}{'n=3 fijo':>10}{'n=3 p95':>10}{'n=3 max':>10}{'n=6 fijo':>10}{'n=10':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['kernel_ref']:<30}{r['freq_level_id']:<6}"
              f"{r['cv3_fixed']:>9.2f}%{r['cv3_p95']:>9.2f}%{r['cv3_max']:>9.2f}%"
              f"{r['cv6_fixed']:>9.2f}%{r['cv10_all']:>9.2f}%")

    print()
    worst_p95 = rows[0]
    print(f"Peor caso (n=3, p95 sobre todos los trios posibles): "
          f"{worst_p95['kernel_ref']}@{worst_p95['freq_level_id']} = {worst_p95['cv3_p95']:.2f}%")

    # Cuenta cuantas celdas ya estan "resueltas" (CV% < 5) a distintos n.
    for threshold in (2.0, 5.0, 10.0):
        for k in (3, 6, 10):
            key = f"cv{k}_fixed" if k != 10 else "cv10_all"
            ok = sum(1 for r in rows if r[key] < threshold)
            print(f"n={k:2d}: {ok}/{len(rows)} celdas con CV% de EDP < {threshold:.0f}%")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "table_edp_convergence_arc174.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"tabla completa: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
