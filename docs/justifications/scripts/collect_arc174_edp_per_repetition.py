#!/usr/bin/env python3
"""Vuelca energia/tiempo POR REPETICION (no promediados) de la campana CPU
final ya corregida (arc174, ver commit del fix de `repetition_index` en
reprocess_frequency_quality_v2.py) -- insumo para el analisis de
convergencia de EDP en `analyze_edp_repetitions_arc174.py`.

Reusa `cpu_policy_headroom.read_run()` (misma logica ya validada: suma
RAPL pkg+dram solo sobre ventanas con energy_valid=='1', tiempo por rango
de timestamps) en vez de reimplementar el parseo de windows.csv.

Corre EN PACCA (lee ~/hyperion-results/campaigns/..., 546 directorios de
corrida) -- es lectura liviana (un windows.csv por corrida, no el CSV
gigante concatenado), pensado para correr via srun en la particion normal,
no en el nodo de login (misma precaucion que el reprocesamiento completo,
que si tumbo pacca dos veces cuando se corrio pesado en el login node).

Uso:
    python3 docs/justifications/scripts/collect_arc174_edp_per_repetition.py
        > docs/justifications/data/repetitions/arc174_edp_per_rep.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "hyperion"))
from classifier.analysis.cpu_policy_headroom import KERNELS, LEVELS, read_run  # noqa: E402

BASE = Path.home() / "hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174"
CID = "pacca_cpu_final_attempt03_20260820"
REPS = range(1, 11)


def main() -> int:
    writer = csv.writer(sys.stdout)
    writer.writerow(["kernel_ref", "freq_level_id", "repetition", "elapsed_s", "energy_j"])
    n_missing = 0
    n_written = 0
    for kernel in KERNELS:
        for level in LEVELS:
            for rep in REPS:
                run_dir = BASE / f"{CID}__{kernel}__{level}__rep{rep:02d}"
                record = read_run(run_dir)
                if record is None:
                    n_missing += 1
                    continue
                writer.writerow([kernel, level, rep, record["elapsed_s"], record["energy_j"]])
                n_written += 1
    print(f"# escritas={n_written} faltantes={n_missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
