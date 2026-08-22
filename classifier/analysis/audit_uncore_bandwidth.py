#!/usr/bin/env python3
"""Auditoría del conteo de bytes de uncore contra STREAM (ARC-180).

CONTEXTO. Un análisis previo (ARC-179) calculó el ancho de banda como
`bytes_moved_uncore_real / delta_t_ns` fila por fila y obtuvo 997 GB/s para
STREAM contra un pico de nodo de 59.5 GB/s, y concluyó que el conteo de
bytes estaba inflado ~17x. ESA CONCLUSIÓN ERA UN ERROR DEL ANÁLISIS, no un
defecto de los datos.

`_apply_uncore_intervals` (orchestrator/postprocess.py) difunde los bytes de
UN intervalo de uncore a TODAS las ventanas de CPU que ese intervalo cubre.
El intervalo de uncore es de ~10 ms (piso de `perf stat -I`) y la ventana de
CPU de ~1 ms, así que cada intervalo cubre del orden de diez ventanas y cada
una lleva escrito el total del intervalo, no su parte. Dividir ese total
entre el `delta_t_ns` de una sola ventana infla el resultado por el número
de ventanas que comparten el intervalo.

El OI sí está bien calculado: `operational_intensity_uncore_real` divide la
SUMA de flops de todas las ventanas cubiertas entre esos mismos bytes, así
que numerador y denominador viven en la misma escala temporal.

QUÉ HACE ESTE SCRIPT. Recalcula el ancho de banda a la granularidad
correcta --la del intervalo de uncore-- y lo contrasta con el pico del
nodo. Los intervalos se reconstruyen agrupando filas CONSECUTIVAS que
comparten el mismo par (cas_read, cas_write), que es la firma de haber
recibido la misma difusión.

Comprueba tres cosas:

  A. STREAM debe dar un ancho de banda cercano al pico del nodo
     (59.5 GB/s). Es el kernel de calibración de ancho de banda: si esto no
     cuadra, el conteo de bytes sí tiene un problema real.
  B. El número de ventanas por intervalo, que es el factor de inflación del
     cálculo erróneo. Debería rondar 10-17.
  C. Si ese factor depende de la FRECUENCIA. Importa porque el análisis de
     acoplamiento de uncore (ARC-179) usa cocientes de ancho de banda entre
     niveles: un factor constante se cancela y la conclusión sobrevive, uno
     que varíe con la frecuencia la invalidaría.
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

from classifier.analysis.gates_c1_c2_c3 import discover_runs  # noqa: E402

COLS = [
    "run_id", "kernel_ref", "freq_level_id", "freq_khz_observed",
    "delta_t_ns", "bytes_moved_uncore_real",
    "uncore_cas_count_read_interval", "uncore_cas_count_write_interval",
]


def intervals_from_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye los intervalos de uncore agrupando filas consecutivas con
    el mismo (cas_read, cas_write). Cada grupo es una difusión."""
    work = df.dropna(subset=["uncore_cas_count_read_interval",
                             "uncore_cas_count_write_interval",
                             "bytes_moved_uncore_real", "delta_t_ns"]).copy()
    if work.empty:
        return pd.DataFrame()
    key = (
        work["uncore_cas_count_read_interval"].astype("int64").astype(str)
        + "_" + work["uncore_cas_count_write_interval"].astype("int64").astype(str)
    )
    # Un cambio de clave abre un intervalo nuevo. Se compara contra la fila
    # anterior y no se agrupa por valor: dos intervalos distintos pueden
    # repetir el mismo conteo por casualidad y fundirlos sesgaría el
    # resultado hacia intervalos artificialmente largos.
    group_id = (key != key.shift()).cumsum()
    grouped = work.groupby(group_id).agg(
        bytes_interval=("bytes_moved_uncore_real", "first"),
        duration_ns=("delta_t_ns", "sum"),
        n_windows=("delta_t_ns", "size"),
    ).reset_index(drop=True)
    grouped = grouped[(grouped["duration_ns"] > 0) & (grouped["bytes_interval"] > 0)]
    grouped["bw_gbs"] = grouped["bytes_interval"] / grouped["duration_ns"]
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--bw-peak-gbs", type=float, default=59.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    print(f"nodo de análisis: {platform.node()}", flush=True)

    # -------- A: STREAM y ERT (corridas de calibración, rep00) -----------
    print("\n== A. calibración: ancho de banda a granularidad de intervalo ==", flush=True)
    calib_rows = []
    for pattern in ("*stream_official*", "*ert_probe*"):
        for path in sorted(campaign_dir.glob(f"{pattern}/windows.csv")):
            df = pd.read_csv(path, usecols=COLS, low_memory=False)
            iv = intervals_from_windows(df)
            if iv.empty:
                continue
            name = path.parent.name
            level = name.split("__")[-2]
            calib_rows.append({
                "run": name.split("__", 1)[1],
                "kernel": "stream" if "stream" in name else "ert",
                "level": level,
                "bw_gbs_interval": round(float(iv["bw_gbs"].median()), 2),
                "bw_gbs_naive_per_window": round(
                    float((df["bytes_moved_uncore_real"] / df["delta_t_ns"]).median()), 2),
                "windows_per_interval": round(float(iv["n_windows"].median()), 2),
                "n_intervals": int(len(iv)),
            })
    calib = pd.DataFrame(calib_rows)
    if not calib.empty:
        print(calib.to_string(index=False), flush=True)
        stream = calib[calib["kernel"] == "stream"]
        if not stream.empty:
            best = float(stream["bw_gbs_interval"].max())
            print(f"\nSTREAM máximo a granularidad de intervalo: {best:.2f} GB/s", flush=True)
            print(f"pico del nodo declarado:                    {args.bw_peak_gbs:.2f} GB/s", flush=True)
            print(f"cociente: {best / args.bw_peak_gbs:.3f}  "
                  f"[~1.0 => el conteo de bytes es correcto]", flush=True)

    # -------- B/C: factor de inflación y su dependencia de la frecuencia --
    print("\n== B/C. ventanas por intervalo, por nivel de frecuencia ==", flush=True)
    index = discover_runs(campaign_dir)
    rows = []
    for kernel in sorted(index["kernel_ref"].unique()):
        for row in index[index["kernel_ref"] == kernel].itertuples():
            df = pd.read_csv(row.windows_path, usecols=COLS, low_memory=False)
            iv = intervals_from_windows(df)
            if iv.empty:
                continue
            rows.append({
                "kernel": kernel,
                "level": row.freq_level_id,
                "windows_per_interval": float(iv["n_windows"].median()),
                "bw_gbs_interval": float(iv["bw_gbs"].median()),
            })
    per_run = pd.DataFrame(rows)
    by_level = per_run.groupby("level").agg(
        wpi_median=("windows_per_interval", "median"),
        wpi_p05=("windows_per_interval", lambda s: float(np.percentile(s, 5))),
        wpi_p95=("windows_per_interval", lambda s: float(np.percentile(s, 95))),
        n_runs=("windows_per_interval", "size"),
    ).round(3).reset_index()
    print(by_level.to_string(index=False), flush=True)

    spread = float(by_level["wpi_median"].max() / by_level["wpi_median"].min())
    print(f"\nrazón max/min de ventanas-por-intervalo entre niveles: {spread:.3f}", flush=True)
    print("[~1.0 => el factor NO depende de la frecuencia, así que los", flush=True)
    print(" cocientes de ancho de banda de ARC-179 siguen siendo válidos]", flush=True)

    # -------- Ancho de banda corregido por kernel y nivel ----------------
    print("\n== ancho de banda CORREGIDO (GB/s), por kernel y nivel ==", flush=True)
    table = per_run.pivot_table(index="kernel", columns="level",
                                values="bw_gbs_interval", aggfunc="median").round(2)
    print(table.to_string(), flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(),
        "bw_peak_gbs_declared": args.bw_peak_gbs,
        "calibration": calib_rows,
        "windows_per_interval_by_level": by_level.to_dict(orient="records"),
        "wpi_spread_across_levels": spread,
        "bandwidth_corrected": json.loads(table.to_json()),
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
