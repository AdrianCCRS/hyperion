"""Diversidad de carga (Anexo N / job 6472): ¿el margen de DVFS de
`rodinia_dwt2d` depende del tamaño del problema, o es una propiedad fija
del kernel?

Compara alpha, energia y EDP entre 5 tamanos de dwt2d (192 a 8192 px, mas
el original ~512) sobre los 6 niveles estandar. Responde dos preguntas
distintas de la de V2 (que mira resolucion de rejilla a tamano fijo):
si el catalogo necesitara declarar el mismo kernel en varios tamanos como
cargas DISTINTAS para el modelo, o si un solo tamano ya representa bien
el comportamiento DVFS del kernel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from classifier.analysis import gpu_oracle_headroom as core

BASE = Path.home() / "hyperion-results/campaigns/pacca_gpu_dwt2d_size_sweep_20260823"
CID = "pacca_gpu_dwt2d_size_sweep_20260823"
KERNELS = [
    "rodinia_dwt2d_s192", "rodinia_dwt2d_s2048", "rodinia_dwt2d_s4096",
    "rodinia_dwt2d_s8192", "rodinia_dwt2d",
]
LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]
FIXED_LEVELS = ["F0", "F1", "F2", "F3", "F4"]
BASELINE = "F0"


def load(reps: int) -> dict[tuple[str, str], dict[str, float]]:
    core.BASE = BASE
    core.CID = CID
    core.KERNELS = list(KERNELS)
    core.CPU_LEVELS = ["REF"]
    core.GPU_LEVELS = list(LEVELS)
    core.FIXED_GPU_LEVELS = list(FIXED_LEVELS)
    core.REPS = list(range(1, reps + 1))
    data = core.collect()
    return {(k, lv): v for (k, _c, lv), v in data.items()}


def fit_alpha(runs: dict[tuple[str, str], dict[str, float]], kernel: str) -> tuple[float, float] | None:
    durations = {}
    for lv in FIXED_LEVELS:
        rec = runs.get((kernel, lv))
        if rec is None:
            continue
        mhz = rec.get("gpu_mhz")
        if mhz:
            durations[mhz] = rec["elapsed_s"]
    f0 = runs.get((kernel, BASELINE), {}).get("gpu_mhz")
    if not f0 or f0 not in durations or len(durations) < 2:
        return None
    from classifier.features.align import fit_alpha as _fit
    try:
        return _fit(durations, f0)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    runs = load(args.reps)
    print(f"{'kernel':<24}{'OI declarada':>14}{'alpha':>10}{'r2':>8}{'E_gpu(F0) J':>14}{'mejor nivel':>13}{'EDP gain':>10}")
    for kernel in KERNELS:
        f0 = runs.get((kernel, BASELINE))
        if f0 is None:
            print(f"{kernel:<24} SIN DATOS")
            continue
        alpha_fit = fit_alpha(runs, kernel)
        alpha_str = f"{alpha_fit[0]:.3f}" if alpha_fit else "n/a"
        r2_str = f"{alpha_fit[1]:.3f}" if alpha_fit else "n/a"

        best_level, best_edp_gain = BASELINE, 0.0
        f0_edp = f0["gpu_j"] * f0["elapsed_s"]
        for lv in LEVELS:
            rec = runs.get((kernel, lv))
            if rec is None:
                continue
            edp = rec["gpu_j"] * rec["elapsed_s"]
            gain = 100.0 * (f0_edp - edp) / f0_edp if f0_edp else 0.0
            if gain > best_edp_gain:
                best_level, best_edp_gain = lv, gain

        print(f"{kernel:<24}{'--':>14}{alpha_str:>10}{r2_str:>8}{f0['gpu_j']:>14.1f}"
              f"{best_level:>13}{best_edp_gain:>+9.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
