"""¿Cuánto margen le queda a un modelo de CPU por encima de la mejor constante?

Es el mismo análisis que `gpu_policy_headroom.py`, aplicado al eje CPU
sobre datos que YA EXISTEN y son válidos (pacca_cpu_final_attempt03_
20260820_arc174, 424/540 corridas, uncore funcionaba cuando se corrió,
antes de la regresión CAP_PERFMON de ARC-184). No necesita CAP_PERFMON:
solo lee windows.csv/summary.txt ya en disco.

METRICA. Energía de paquete + DRAM vía RAPL (pkg+dram), que en pacca SÍ
es legible sin ningún permiso especial -- a diferencia de uncore_imc.

Se comparan las mismas tres políticas que en GPU:
  1. siempre F0 (máxima frecuencia)
  2. mejor constante (una sola frecuencia fija para todo el conjunto)
  3. oráculo por kernel (el mejor nivel de cada kernel, conocido a posteriori)

El margen aprovechable por un modelo es oráculo - mejor_constante.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

BASE = Path.home() / "hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174"
CID = "pacca_cpu_final_attempt03_20260820"

KERNELS = [
    "npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu",
    "dgemm_n2048", "rodinia_lavamd_omp", "rajaperf_polybench_3mm_omp",
]
LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]
BASELINE = "F0"
REPS = range(1, 11)  # hasta 10 repeticiones por combinación en esta campaña


def read_run(run_dir: Path) -> dict[str, float] | None:
    """Suma energía (RAPL pkg+dram) y tiempo directo de windows.csv.

    Esta campaña quedó reprocesada (ARC-174): no tiene summary.txt, solo
    windows.csv + reprocess_provenance.json. energy_valid marca por fila
    si la lectura RAPL de esa ventana es de fiar; se suma solo esas.
    """
    windows_path = run_dir / "windows.csv"
    if not windows_path.exists():
        return None

    total_energy_uj = 0.0
    n_energy_rows = 0
    t_min: float | None = None
    t_max: float | None = None
    with windows_path.open() as handle:
        for row in csv.DictReader(handle):
            t_start = row.get("t_start_ns")
            t_end = row.get("t_end_ns")
            if t_start not in (None, "") and t_end not in (None, ""):
                t0, t1 = float(t_start), float(t_end)
                t_min = t0 if t_min is None else min(t_min, t0)
                t_max = t1 if t_max is None else max(t_max, t1)
            if row.get("energy_valid") != "1":
                continue
            pkg = row.get("pkg_delta_uj")
            dram = row.get("dram_delta_uj")
            if pkg in (None, "") or dram in (None, ""):
                continue
            total_energy_uj += float(pkg) + float(dram)
            n_energy_rows += 1

    if n_energy_rows == 0 or t_min is None or t_max is None:
        return None
    elapsed_s = (t_max - t_min) / 1e9
    energy_j = total_energy_uj / 1e6
    if elapsed_s <= 0 or energy_j <= 0:
        return None
    return {"elapsed_s": elapsed_s, "energy_j": energy_j}


def collect(base: Path, cid: str) -> dict[tuple[str, str], dict[str, float]]:
    aggregated: dict[tuple[str, str], dict[str, float]] = {}
    for kernel in KERNELS:
        for level in LEVELS:
            runs = []
            for rep in REPS:
                run_dir = base / f"{cid}__{kernel}__{level}__rep{rep:02d}"
                record = read_run(run_dir)
                if record is not None:
                    runs.append(record)
            if not runs:
                continue
            mean_e = sum(r["energy_j"] for r in runs) / len(runs)
            mean_t = sum(r["elapsed_s"] for r in runs) / len(runs)
            aggregated[(kernel, level)] = {
                "energy_j": mean_e, "elapsed_s": mean_t, "n_reps": float(len(runs)),
            }
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--campaign-id", default=CID)
    parser.add_argument("--max-slowdown-pct", type=float, action="append")
    args = parser.parse_args()
    budgets = args.max_slowdown_pct or [4.0, 10.0, 15.0, 1e9]

    runs = collect(args.base, args.campaign_id)
    if not runs:
        print(f"ERROR: no se leyó ninguna corrida bajo {args.base}")
        return 1

    print("=" * 78)
    print("TABLA 1 -- Energía y tiempo por nivel (RAPL pkg+dram, media de repeticiones)")
    print("=" * 78)
    header = f"{'kernel':<28} {'nivel':>5} {'t(s)':>9} {'E(J)':>9} {'n':>3}"
    print(header)
    print("-" * len(header))
    for kernel in KERNELS:
        for level in LEVELS:
            rec = runs.get((kernel, level))
            if rec is None:
                continue
            print(f"{kernel:<28} {level:>5} {rec['elapsed_s']:>9.3f} {rec['energy_j']:>9.1f} {int(rec['n_reps']):>3}")
    print()

    def eligible(kernel: str, level: str, budget: float) -> bool:
        ref = runs.get((kernel, BASELINE))
        rec = runs.get((kernel, level))
        if ref is None or rec is None:
            return False
        return rec["elapsed_s"] <= ref["elapsed_s"] * (1.0 + budget / 100.0)

    def saving(kernel: str, level: str) -> float:
        ref = runs[(kernel, BASELINE)]
        rec = runs[(kernel, level)]
        return 100.0 * (ref["energy_j"] - rec["energy_j"]) / ref["energy_j"]

    all_kernels = [k for k in KERNELS if (k, BASELINE) in runs]

    for budget in budgets:
        tag = "sin limite" if budget > 1e8 else f"<= {budget:.0f}% mas lento"
        print("=" * 78)
        print(f"PRESUPUESTO DE DEGRADACION: {tag}")
        print("=" * 78)

        best_const, best_const_saving = None, float("-inf")
        for level in LEVELS:
            if not all(eligible(k, level, budget) for k in all_kernels):
                continue
            mean_saving = sum(saving(k, level) for k in all_kernels) / len(all_kernels)
            if mean_saving > best_const_saving:
                best_const, best_const_saving = level, mean_saving

        oracle_total, per_kernel = 0.0, {}
        for k in all_kernels:
            options = [lv for lv in LEVELS if eligible(k, lv, budget)]
            if not options:
                options = [BASELINE]
            best_lv = max(options, key=lambda lv: saving(k, lv))
            per_kernel[k] = (best_lv, saving(k, best_lv))
            oracle_total += saving(k, best_lv)
        oracle_mean = oracle_total / len(all_kernels)

        print(f"   siempre {BASELINE:<4}          ahorro medio = {0.0:6.2f}%")
        if best_const is None:
            print("   mejor constante        ninguna elegible en todos los kernels")
            const_val = 0.0
        else:
            print(f"   mejor constante ({best_const:<3})  ahorro medio = {best_const_saving:6.2f}%")
            const_val = best_const_saving
        print(f"   oraculo por kernel     ahorro medio = {oracle_mean:6.2f}%")
        print(f"   >> MARGEN PARA EL MODELO         = {oracle_mean - const_val:6.2f} puntos")
        print("   optimo por kernel:")
        for k in all_kernels:
            lv, sv = per_kernel[k]
            mark = "" if best_const is None or lv == best_const else "   <- difiere de la constante"
            print(f"      {k:<28} {lv:<4} {sv:+7.2f}%{mark}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
