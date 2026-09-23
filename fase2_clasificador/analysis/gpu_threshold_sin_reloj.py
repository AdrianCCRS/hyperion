"""Umbral de abstencion del candidato de GPU vigente (random forest sin reloj ni potencia), con el mismo criterio que
el del libro para el candidato anterior (`gpu_quality_report.py::stage_threshold`: exactitud balanceada por celda
sobre lo decidido, cobertura media por celda >= 0.60, LOFO por familia sobre las corridas ya medidas). No reentrena
ningun modelo ni mide nada nuevo. Correr en pacca; nunca en local."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fase2_clasificador.analysis import gpu_quality_report as gq

CLOCK, POWER = "gpu_sm_clock_mhz_median", "gpu_power_mw_median"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frame = gq.load(a.source, split_rajaperf_cuda=True, extra_features=["gpu_mem_util_pct_std"])
    gq.FEATURES[:] = [f for f in gq.FEATURES if f not in (CLOCK, POWER)]
    gq.CANDIDATE = "random_forest"
    print(f"variables: {gq.FEATURES}  filas={len(frame)}", flush=True)
    families = sorted(frame[gq.FAMILY_COL].unique())
    fam_codes = pd.Categorical(frame[gq.FAMILY_COL], categories=families).codes
    gq.stage_threshold(frame, families, fam_codes, a, a.out)


if __name__ == "__main__":
    main()
