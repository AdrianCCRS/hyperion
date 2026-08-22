#!/usr/bin/env python3
"""T0.2c: rehacer C2 (alpha por TRAMO) con duraciones sin filtrar (ARC-182).

C2 ajusto alpha por celda (kernel x repeticion x centil de avance) sumando
`delta_t_ns` sobre ventanas que HABIAN PASADO el filtro de calidad de
frecuencia. T0.2 mostro que ese filtro rechaza mas ventanas a alta
frecuencia que a baja en los nueve kernels, asi que las duraciones por celda
arrastran el mismo sesgo que las duraciones por corrida y todos los alpha
por tramo estan inflados.

T0.2b valido la alternativa: a nivel de corrida, las duraciones de ventanas
CRUDAS coinciden con la duracion que el kernel reporta por stdout con menos
de 0.008 de diferencia en alpha. El stdout no se puede desagregar por tramo
--es un solo numero por corrida-- pero esa concordancia justifica usar las
ventanas crudas, que si se pueden desagregar.

QUE DECIDE. El numero central del diagnostico: cuantas celdas caen por
debajo del umbral alpha <= 0.226. Con duraciones filtradas la respuesta era
CERO de 9000, con un minimo de 0.242. Si al corregir aparecen celdas por
debajo, el catalogo actual SI toca el regimen viable en algunos tramos y
eso cambia el plan.
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
from classifier.features import align  # noqa: E402

NOMINAL_MHZ = {"F0": 3200.0, "F1": 2600.0, "F2": 2000.0, "F3": 1400.0, "F4": 800.0}
BREAK_EVEN = 0.226


def load_kernel_raw(index: pd.DataFrame, kernel: str) -> pd.DataFrame:
    """Como load_kernel pero SIN el filtro de calidad de frecuencia.

    Se conserva `quality_status == "ok"`: eso descarta ventanas cuya
    telemetria es invalida, que es otra cosa que descartar ventanas cuya
    FRECUENCIA se salio de tolerancia. Solo la segunda introduce el sesgo
    dependiente de la frecuencia.
    """
    frames = []
    for row in index[index["kernel_ref"] == kernel].itertuples():
        frame = pd.read_csv(row.windows_path, usecols=USECOLS, low_memory=False)
        frame["rep_idx"] = row.rep_idx
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.loc[df["quality_status"] == "ok"].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = discover_runs(Path(args.campaign_dir))
    print(f"nodo de análisis: {platform.node()}", flush=True)

    rows = []
    for kernel in sorted(index["kernel_ref"].unique()):
        df = load_kernel_raw(index, kernel)
        if df.empty:
            continue
        df["repetition"] = df["rep_idx"]
        work = align.add_instruction_progress(df)
        work = align.assign_progress_bins(work, n_bins=args.n_bins)
        cells = align.aggregate_cells(work, feature_cols=["ipc"])

        alphas = []
        for _, group in cells.groupby(["kernel_ref", "repetition", "progress_bin"], observed=True):
            durations = {}
            for _, cell in group.iterrows():
                mhz = NOMINAL_MHZ.get(cell["freq_level_id"])
                if mhz:
                    durations[mhz] = float(cell["duration_ns"])
            if 3200.0 not in durations or len(durations) < 3:
                continue
            try:
                alpha, _ = align.fit_alpha(durations, 3200.0)
            except ValueError:
                continue
            if np.isfinite(alpha):
                alphas.append(alpha)

        if not alphas:
            continue
        arr = np.array(alphas, dtype=float)
        rows.append({
            "kernel": kernel,
            "n_celdas": int(arr.size),
            "alpha_medio": round(float(arr.mean()), 4),
            "alpha_sd": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else None,
            "alpha_p05": round(float(np.percentile(arr, 5)), 4),
            "alpha_min": round(float(arr.min()), 4),
            "pct_celdas_bajo_umbral": round(100.0 * float((arr <= BREAK_EVEN).mean()), 2),
        })

    table = pd.DataFrame(rows).sort_values("alpha_medio")
    print("\n== alpha por TRAMO con duraciones sin filtrar ==", flush=True)
    print(table.to_string(index=False), flush=True)

    total_cells = int(table["n_celdas"].sum())
    below = sum(r["n_celdas"] * r["pct_celdas_bajo_umbral"] / 100.0 for r in rows)
    print(f"\numbral: alpha <= {BREAK_EVEN}", flush=True)
    print(f"celdas por debajo: {below:.0f} de {total_cells} "
          f"({100.0 * below / total_cells:.2f} %)", flush=True)
    print(f"alpha mínimo sobre todas las celdas: {table['alpha_min'].min():.4f}", flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(),
        "break_even": BREAK_EVEN,
        "per_kernel": rows,
        "total_celdas": total_cells,
        "celdas_bajo_umbral": below,
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
