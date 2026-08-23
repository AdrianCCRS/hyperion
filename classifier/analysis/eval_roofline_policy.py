#!/usr/bin/env python3
"""Evaluacion offline de la politica Roofline-DVFS (ARC-183).

LA POLITICA PROPUESTA

    f*(t) = min { f : OI_t <= I_ridge(f) }

es decir: la frecuencia MAS BAJA en la que la ventana sigue estando del lado
memory-bound del Roofline, con el argumento de que bajar mas convertiria al
computo en el nuevo cuello de botella. Si la ventana ya es compute-bound al
maximo reloj, la politica devuelve F0.

QUE SE COMPRUEBA AQUI. La politica es una hipotesis falsable y este
proyecto tiene los datos para falsarla sin medir nada nuevo: hay corridas
completas de los nueve kernels en los cinco niveles fijos, con energia RAPL
de paquete y duracion. Para cada kernel se compara el EDP que habria
obtenido la politica contra tres referencias:

  - ORACULO: el mejor nivel realmente observado. Es la cota superior de lo
    que cualquier politica puede lograr con esta rejilla.
  - SIEMPRE F0: la frecuencia maxima. Es lo que hace `performance`, y es lo
    que la evidencia actual del proyecto dice que es optimo en 9 de 9.
  - REF: el gobernador nativo de Linux.

LA DUDA CONCRETA QUE MOTIVA LA PRUEBA. La etiqueta Roofline dice que techo
esta mas cerca, NO como responde el tiempo de ejecucion al reloj. Son cosas
distintas y este dataset las separa: npb_cg tiene b medio 0.839 --
firmemente memory-bound-- y sin embargo alpha = 0.757, o sea que su tiempo
escala casi proporcionalmente con la frecuencia. Una politica que baje el
reloj por estar del lado memory-bound le haria perder mas tiempo del que
gana en potencia.

El EDP se calcula con energia y duracion SUMADAS sobre ventanas crudas
(quality_status ok, sin el filtro de frecuencia), que T0.2b valido contra
la duracion que el kernel reporta por stdout.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.analysis.gates_c1_c2_c3 import USECOLS, discover_runs  # noqa: E402

LEVEL_ORDER = ["F4", "F3", "F2", "F1", "F0"]  # de menor a mayor frecuencia
NOMINAL_MHZ = {"F0": 3200.0, "F1": 2600.0, "F2": 2000.0, "F3": 1400.0, "F4": 800.0}


def load_raw(index: pd.DataFrame, kernel: str) -> pd.DataFrame:
    frames = []
    cols = list(dict.fromkeys(USECOLS + ["pkg_delta_uj"]))
    for row in index[index["kernel_ref"] == kernel].itertuples():
        frame = pd.read_csv(row.windows_path, usecols=lambda c: c in cols, low_memory=False)
        frame["rep_idx"] = row.rep_idx
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return df.loc[df["quality_status"] == "ok"].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = discover_runs(Path(args.campaign_dir))
    print(f"nodo de análisis: {platform.node()}\n", flush=True)

    # I_ridge por nivel: ya viene calibrado por nivel en el dataset
    # (ARC-78), una columna por fila. Se toma la mediana por nivel.
    all_df = []
    per_kernel = {}
    for kernel in sorted(index["kernel_ref"].unique()):
        df = load_raw(index, kernel)
        if df.empty:
            continue
        per_kernel[kernel] = df
        all_df.append(df[["freq_level_id", "i_ridge_used"]])
    ridge_by_level = (pd.concat(all_df)
                      .groupby("freq_level_id")["i_ridge_used"].median().to_dict())
    print("I_ridge calibrado por nivel (FLOP/byte):", flush=True)
    for lvl in LEVEL_ORDER:
        if lvl in ridge_by_level:
            print(f"   {lvl}  {NOMINAL_MHZ[lvl]:6.0f} MHz   {ridge_by_level[lvl]:.4f}", flush=True)

    rows = []
    for kernel, df in per_kernel.items():
        # EDP real por nivel: energia x tiempo, promediado sobre repeticiones.
        edp = {}
        for lvl in [*LEVEL_ORDER, "REF"]:
            sub = df[df["freq_level_id"] == lvl]
            if sub.empty:
                continue
            per_rep = sub.groupby("rep_idx").agg(
                e_uj=("pkg_delta_uj", "sum"), t_ns=("delta_t_ns", "sum"))
            per_rep = per_rep[(per_rep["e_uj"] > 0) & (per_rep["t_ns"] > 0)]
            if per_rep.empty:
                continue
            edp[lvl] = float((per_rep["e_uj"] * 1e-6 * per_rep["t_ns"] * 1e-9).mean())
        fixed = {k: v for k, v in edp.items() if k in LEVEL_ORDER}
        if "F0" not in fixed or len(fixed) < 3:
            continue

        # DECISION DE LA POLITICA. Se evalua sobre las ventanas observadas en
        # F0, que es el estado desde el que un demonio arrancaria. Para cada
        # ventana se busca el nivel MAS BAJO cuyo ridge siga por encima de su
        # OI; el nivel elegido para el kernel es el modo sobre sus ventanas.
        obs = df[(df["freq_level_id"] == "F0")].copy()
        oi = pd.to_numeric(obs["operational_intensity_uncore_real"], errors="coerce")
        oi = oi[np.isfinite(oi) & (oi > 0)]
        if oi.empty:
            continue
        choices = []
        for value in oi.to_numpy():
            pick = "F0"
            for lvl in LEVEL_ORDER:          # de menor a mayor frecuencia
                if lvl in ridge_by_level and value <= ridge_by_level[lvl]:
                    pick = lvl
                    break
            choices.append(pick)
        chosen = pd.Series(choices).mode().iloc[0]
        frac_chosen = float((pd.Series(choices) == chosen).mean())

        oracle = min(fixed, key=fixed.get)
        rows.append({
            "kernel": kernel,
            "OI_mediana": round(float(oi.median()), 3),
            "politica_elige": chosen,
            "frac_ventanas_acuerdo": round(frac_chosen, 3),
            "oraculo": oracle,
            "edp_politica": round(fixed.get(chosen, float("nan")), 4),
            "edp_oraculo": round(fixed[oracle], 4),
            "edp_F0": round(fixed["F0"], 4),
            "edp_REF": round(edp["REF"], 4) if "REF" in edp else None,
            "politica_vs_oraculo": round(fixed.get(chosen, np.nan) / fixed[oracle], 3),
            "politica_vs_F0": round(fixed.get(chosen, np.nan) / fixed["F0"], 3),
        })

    table = pd.DataFrame(rows)
    print("\n== política Roofline-DVFS contra las referencias ==", flush=True)
    print(table.to_string(index=False), flush=True)

    worse = table[table["politica_vs_F0"] > 1.0]
    print(f"\nla política empeora el EDP frente a F0 en {len(worse)} de {len(table)} kernels", flush=True)
    print(f"penalización mediana frente al oráculo: "
          f"{(table['politica_vs_oraculo'].median() - 1) * 100:+.1f} %", flush=True)
    print(f"penalización PEOR frente al oráculo:    "
          f"{(table['politica_vs_oraculo'].max() - 1) * 100:+.1f} %  "
          f"({table.loc[table['politica_vs_oraculo'].idxmax(), 'kernel']})", flush=True)
    print(f"veces que la política acierta el óptimo: "
          f"{int((table['politica_elige'] == table['oraculo']).sum())} de {len(table)}", flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(),
        "ridge_by_level": ridge_by_level,
        "per_kernel": rows,
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
