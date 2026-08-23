"""¿Cuánto margen le queda a un MODELO por encima de la mejor constante?

Esta es la pregunta que decide si vale la pena entrenar algo, y es
distinta de la del Anexo M. Allí se comparó el oráculo contra "siempre
F0". Pero "siempre F0" no es el rival serio: el rival serio es **la mejor
frecuencia fija única**, elegida offline mirando todo el conjunto. Si esa
constante ya captura casi todo el oráculo, un modelo por kernel no aporta
nada defendible.

Se comparan tres políticas, todas sobre energía de GPU (NVML), que es la
métrica del campo tras la corrección del Anexo M:

  1. `siempre F0`        -- el baseline trivial "todo a máxima velocidad".
  2. `mejor constante`   -- la frecuencia fija única que minimiza la
                            energía media del conjunto. Es el rival real.
  3. `oráculo por kernel`-- el mejor nivel para cada kernel, conocido a
                            posteriori. Es el techo de cualquier modelo.

El margen aprovechable por un modelo es `oráculo - mejor constante`.

Todo bajo un presupuesto de degradación configurable: un nivel solo es
elegible si no alarga la corrida más de `--max-slowdown-pct` respecto de
F0. Sin esa restricción, "el óptimo" puede ser un nivel inaceptable.

KERNELS DE ENERGIA DESPRECIABLE. `rodinia_backprop` mueve entre 8 y 49 J
de GPU en toda la corrida -- dos órdenes de magnitud menos que el resto
(600-1000 J). Sus porcentajes de ahorro son enormes y sin sentido físico
(ruido dividido por casi cero). Se reporta el resumen con y sin él.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from classifier.analysis import gpu_oracle_headroom as core

LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]
BASELINE = "F0"


def load(base: Path, cid: str, kernels: list[str], cpu_level: str, reps: int):
    core.BASE = base
    core.CID = cid
    core.KERNELS = list(kernels)
    core.CPU_LEVELS = [cpu_level]
    core.REPS = list(range(1, reps + 1))
    data = core.collect()
    return {(k, lv): v for (k, _c, lv), v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-level", default="REF")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--max-slowdown-pct",
        type=float,
        action="append",
        help="Puede repetirse para evaluar varios presupuestos.",
    )
    parser.add_argument(
        "--min-energy-j",
        type=float,
        default=50.0,
        help="Kernels cuya energia de GPU en F0 este por debajo de esto se "
             "marcan como despreciables y se reportan aparte.",
    )
    args = parser.parse_args()
    budgets = args.max_slowdown_pct or [4.0, 10.0, 15.0, 1e9]

    campaigns = [
        (
            Path.home() / "hyperion-results/campaigns/pacca_gpu_nucleo_activo_20260823",
            "pacca_gpu_nucleo_activo_20260823",
            ["rodinia_gaussian", "gpu_dgemm_n4096", "rodinia_heartwall", "rodinia_lavamd"],
        ),
        (
            Path.home() / "hyperion-results/campaigns/pacca_gpu_alpha_screening_20260823",
            "pacca_gpu_alpha_screening_20260823",
            ["rodinia_myocyte", "rodinia_backprop", "rodinia_dwt2d"],
        ),
    ]

    runs: dict[tuple[str, str], dict[str, float]] = {}
    for base, cid, kernels in campaigns:
        runs.update(load(base, cid, kernels, args.cpu_level, args.reps))

    all_kernels = sorted({k for (k, _lv) in runs})
    if not all_kernels:
        print("ERROR: no se cargo ninguna corrida")
        return 1

    negligible = {
        k for k in all_kernels
        if (k, BASELINE) in runs and runs[(k, BASELINE)]["gpu_j"] < args.min_energy_j
    }
    if negligible:
        print(f"Kernels de energia despreciable (< {args.min_energy_j:.0f} J en F0), "
              f"reportados aparte: {sorted(negligible)}")
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

    for budget in budgets:
        tag = "sin limite" if budget > 1e8 else f"<= {budget:.0f}% mas lento"
        print("=" * 84)
        print(f"PRESUPUESTO DE DEGRADACION: {tag}")
        print("=" * 84)

        for group_name, group in (
            ("TODOS", all_kernels),
            ("SIN despreciables", [k for k in all_kernels if k not in negligible]),
        ):
            if not group:
                continue
            if group_name == "SIN despreciables" and not negligible:
                continue

            # Mejor constante: el nivel fijo unico que maximiza el ahorro medio,
            # exigiendo que sea elegible en TODOS los kernels del grupo.
            best_const, best_const_saving = None, float("-inf")
            for level in LEVELS:
                if not all(eligible(k, level, budget) for k in group):
                    continue
                mean_saving = sum(saving(k, level) for k in group) / len(group)
                if mean_saving > best_const_saving:
                    best_const, best_const_saving = level, mean_saving

            # Oraculo por kernel.
            oracle_total, per_kernel = 0.0, {}
            for k in group:
                options = [lv for lv in LEVELS if eligible(k, lv, budget)]
                if not options:
                    options = [BASELINE]
                best_lv = max(options, key=lambda lv: saving(k, lv))
                per_kernel[k] = (best_lv, saving(k, best_lv))
                oracle_total += saving(k, best_lv)
            oracle_mean = oracle_total / len(group)

            print(f"\n-- grupo {group_name} ({len(group)} kernels) --")
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
