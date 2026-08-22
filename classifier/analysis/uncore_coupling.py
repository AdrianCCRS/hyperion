#!/usr/bin/env python3
"""¿El DVFS de núcleo arrastra al uncore en este nodo? (ARC-179)

MOTIVO. El ajuste de alpha por tramo (C2, ARC-178) devolvió alpha > 1 en
cuatro de nueve kernels (dgemm 1.159, npb_ft hasta 1.327, lavamd 1.092,
3mm 1.056), con r2 de 0.96-0.999. Bajo la ley clásica

    T(f)/T(f_ref) = (1 - alpha) + alpha * (f_ref/f)

alpha es una FRACCIÓN del tiempo y no puede pasar de 1: el tiempo no puede
escalar más que proporcionalmente al inverso del reloj. Que el ajuste sea
excelente y aun así dé alpha > 1 no es ruido, es el modelo mal
especificado.

HIPÓTESIS. La ley supone que la parte NO sensible al reloj (la espera a
memoria) es constante en frecuencia. Eso solo vale si el uncore --la malla,
el controlador de memoria, la L3-- corre a frecuencia independiente. Si al
bajar `scaling_max_freq` de los núcleos el uncore también baja, entonces la
memoria TAMBIÉN se vuelve más lenta, el término "constante" crece, y el
ajuste lo absorbe inflando alpha por encima de 1.

POR QUÉ IMPORTA MÁS QUE UN DETALLE DE AJUSTE. Si es cierto, explica el
hallazgo central del proyecto --que ningún kernel baja del umbral de
viabilidad alpha <= 0.226-- como una propiedad DEL NODO y no como un
defecto de selección de cargas. Son dos tesis muy distintas:

  (a) "los nueve kernels resultaron sensibles a frecuencia"  -> parece mala
      elección de benchmarks, y la respuesta sería agregar más cargas.
  (b) "en esta plataforma el DVFS de núcleo también frena el uncore, así
      que la fracción insensible a la frecuencia no llega a existir para
      NINGUNA carga" -> es un resultado medido sobre la plataforma, y
      agregar cargas no lo cambiaría.

PRUEBA. El ancho de banda ALCANZADO por una carga limitada por memoria es
la magnitud que distingue los dos casos. Si el uncore fuera independiente,
una carga que ya satura la memoria mantendría su ancho de banda al bajar el
reloj del núcleo: está esperando a la DRAM, no calculando. Si el ancho de
banda cae junto con la frecuencia de núcleo, el uncore está acoplado.

Se reporta el ancho de banda relativo a F0 por nivel. Un kernel
memory-bound con uncore independiente debería quedarse cerca de 1.0; una
caída que siga a f/f_ref es la firma del acoplamiento.
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

from classifier.analysis.gates_c1_c2_c3 import discover_runs, load_kernel  # noqa: E402

USECOLS_EXTRA = ["bytes_moved_uncore_real", "operational_intensity_uncore_real"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = discover_runs(Path(args.campaign_dir))
    kernels = sorted(index["kernel_ref"].unique())
    print(f"nodo de análisis: {platform.node()}", flush=True)

    rows = []
    for kernel in kernels:
        df = load_kernel(index, kernel)
        if df.empty:
            continue
        df = df[df["freq_level_id"] != "REF"]
        work = pd.DataFrame({
            "level": df["freq_level_id"],
            "bytes": pd.to_numeric(df["bytes_moved_uncore_real"], errors="coerce"),
            "dt_ns": pd.to_numeric(df["delta_t_ns"], errors="coerce"),
            "mhz": pd.to_numeric(df["freq_khz_observed"], errors="coerce") / 1000.0,
        }).dropna()
        work = work[(work["dt_ns"] > 0) & (work["bytes"] > 0)]
        if work.empty:
            continue

        # Ancho de banda alcanzado, GB/s. Se agrega por nivel con la MEDIANA
        # para que una ventana atípica no mueva el resultado.
        work["bw_gbs"] = work["bytes"] / work["dt_ns"]  # bytes/ns == GB/s
        by_level = work.groupby("level").agg(
            bw_gbs=("bw_gbs", "median"),
            mhz=("mhz", "median"),
            n=("bw_gbs", "size"),
        ).reset_index()

        ref = by_level[by_level["level"] == "F0"]
        if ref.empty:
            continue
        bw_ref = float(ref["bw_gbs"].iloc[0])
        mhz_ref = float(ref["mhz"].iloc[0])

        for _, r in by_level.iterrows():
            rows.append({
                "kernel": kernel,
                "level": r["level"],
                "mhz": round(float(r["mhz"]), 1),
                "freq_rel": round(float(r["mhz"]) / mhz_ref, 4),
                "bw_gbs": round(float(r["bw_gbs"]), 4),
                "bw_rel_to_f0": round(float(r["bw_gbs"]) / bw_ref, 4),
                "n_windows": int(r["n"]),
            })

    table = pd.DataFrame(rows).sort_values(["kernel", "mhz"], ascending=[True, False])
    print(table.to_string(index=False), flush=True)

    # Firma cuantitativa: pendiente de bw_rel contra freq_rel. Cerca de 0
    # significa uncore independiente (el ancho de banda no sigue al reloj);
    # cerca de 1 significa acoplamiento proporcional.
    print("\npendiente d(bw_rel)/d(freq_rel) por kernel "
          "[~0 = uncore independiente, ~1 = acoplado]", flush=True)
    slopes = {}
    for kernel, group in table.groupby("kernel"):
        x = group["freq_rel"].to_numpy(dtype=float)
        y = group["bw_rel_to_f0"].to_numpy(dtype=float)
        if x.size < 2:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        slopes[kernel] = round(slope, 4)
        print(f"  {kernel:32s} {slope:+.4f}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"analysis_node": platform.node(), "rows": rows, "slopes": slopes}, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
