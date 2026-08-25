"""V2 (Estrategia_GPU_Fase2.md §10): ¿la rejilla fina (G1-G4, 300MHz ->
75MHz de paso entre F0 y F1) abre margen bajo presupuesto estricto, o el
4% sigue siendo inalcanzable?

Reusa gpu_oracle_headroom.read_run() apuntando a
pacca_gpu_fine_grid_dataset_20260823 (job 6471, 210/210 aceptadas,
verificado V1) con los 10 niveles reales de esa campaña (REF, F0, G1-G4,
F1-F4) en vez de los 6 originales -- mismo patron de
gpu_policy_headroom.py (mejor constante vs. oraculo vs. siempre F0), pero
con la grilla completa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from classifier.analysis import gpu_oracle_headroom as core

BASE = Path.home() / "hyperion-results/campaigns/pacca_gpu_fine_grid_dataset_20260823"
CID = "pacca_gpu_fine_grid_dataset_20260823"
KERNELS = [
    "rodinia_gaussian", "gpu_dgemm_n4096", "rodinia_heartwall", "rodinia_lavamd",
    "rodinia_myocyte", "rodinia_backprop", "rodinia_dwt2d",
]
LEVELS = ["REF", "F0", "G1", "G2", "G3", "G4", "F1", "F2", "F3", "F4"]
BASELINE = "F0"


def load(base: Path, cid: str, kernels: list[str], reps: int) -> dict[tuple[str, str], dict[str, float]]:
    core.BASE = base
    core.CID = cid
    core.KERNELS = list(kernels)
    core.CPU_LEVELS = ["REF"]
    core.GPU_LEVELS = list(LEVELS)
    core.FIXED_GPU_LEVELS = [lv for lv in LEVELS if lv != "REF"]
    core.REPS = list(range(1, reps + 1))
    data = core.collect()
    return {(k, lv): v for (k, _c, lv), v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-slowdown-pct", type=float, action="append")
    parser.add_argument("--min-energy-j", type=float, default=50.0)
    args = parser.parse_args()
    budgets = args.max_slowdown_pct or [4.0, 10.0, 15.0, 1e9]

    runs = load(BASE, CID, KERNELS, args.reps)
    all_kernels = sorted({k for (k, _lv) in runs})
    if not all_kernels:
        print(f"ERROR: no se cargo ninguna corrida bajo {BASE}")
        return 1

    negligible = {
        k for k in all_kernels
        if (k, BASELINE) in runs and runs[(k, BASELINE)]["gpu_j"] < args.min_energy_j
    }
    if negligible:
        print(f"Kernels de energia despreciable (< {args.min_energy_j:.0f} J en F0): {sorted(negligible)}")
        for k in sorted(negligible):
            print(f"   {k}: E_gpu(F0) = {runs[(k, BASELINE)]['gpu_j']:.1f} J")
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
        return 100.0 * (ref["gpu_j"] - rec["gpu_j"]) / ref["gpu_j"]

    group = [k for k in all_kernels if k not in negligible]

    for budget in budgets:
        tag = "sin limite" if budget > 1e8 else f"<= {budget:.0f}% mas lento"
        print("=" * 84)
        print(f"PRESUPUESTO DE DEGRADACION: {tag} -- REJILLA FINA (10 niveles, G1-G4 incluidos)")
        print("=" * 84)

        best_const, best_const_saving = None, float("-inf")
        for level in LEVELS:
            if not all(eligible(k, level, budget) for k in group):
                continue
            mean_saving = sum(saving(k, level) for k in group) / len(group)
            if mean_saving > best_const_saving:
                best_const, best_const_saving = level, mean_saving

        oracle_total, per_kernel = 0.0, {}
        for k in group:
            options = [lv for lv in LEVELS if eligible(k, lv, budget)]
            if not options:
                options = [BASELINE]
            best_lv = max(options, key=lambda lv: saving(k, lv))
            per_kernel[k] = (best_lv, saving(k, best_lv))
            oracle_total += saving(k, best_lv)
        oracle_mean = oracle_total / len(group)

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
        for k in group:
            lv, sv = per_kernel[k]
            mark = "" if best_const is None or lv == best_const else "   <- difiere de la constante"
            print(f"      {k:<22} {lv:<4} {sv:+7.2f}%{mark}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
