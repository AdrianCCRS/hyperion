"""Contraste agente frente a F1 fijo en E-A (revisión C1) y latencia de decisión por fase (revisión C5).

Entradas:
  docs/libro/datos/fase4_20260926/fase4_EA.csv              confirmatorio E-A bajo performance (job 7696)
  docs/libro/datos/fase4_20260930/fase4_EA_powersave.csv    confirmatorio E-A bajo powersave (job 7789)
  docs/libro/datos/fase4_20260930/f1_fijo_celdas/           muestras nvidia-smi (utilización, reloj SM; cada 250 ms) de
                                                            las celdas F1 fijo, y fases/decisiones de las celdas activas

Salidas en stdout: (1) ubicación de las muestras con el reloj fuera de 1230-1290 MHz en el brazo F1 fijo, (2) razones
por bloque con media geométrica e IC95 (t sobre el logaritmo) de F1 fijo frente a la base y del agente frente a F1
fijo, (3) latencia de la primera decisión y reposo de GPU al inicio de cada fase en las celdas activas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path(__file__).resolve().parents[1] / "datos"
CELDAS = D / "fase4_20260930" / "f1_fijo_celdas"
PERIODO_S = 0.25


def muestras_fuera_de_rango() -> None:
    for f in sorted(CELDAS.glob("*fijo_gpu_f1_*__fixed_gpu_clock.csv")):
        filas = []
        for linea in f.read_text().splitlines():
            partes = linea.split(",")
            try:
                filas.append((int(partes[0]), int(partes[1])))
            except (ValueError, IndexError):
                continue
        fuera = [i for i, (u, c) in enumerate(filas) if u >= 10 and not 1230 <= c <= 1290]
        fases = [json.loads(x) for x in f.with_name(f.name.replace("fixed_gpu_clock.csv", "phases.jsonl")).read_text().splitlines()]
        t0 = fases[0]["begin_ns"]
        limites = [((p["begin_ns"] - t0) / 1e9, (p["end_ns"] - t0) / 1e9, p["kernel_id"]) for p in fases]
        # Atribución aproximada por índice de muestra (el muestreador arranca junto con la aplicación)
        por_fase: dict[str, int] = {}
        for i in fuera:
            t = i * PERIODO_S
            k = next((n for a, b, n in limites if a <= t <= b + 1.5), "entre fases")
            por_fase[k] = por_fase.get(k, 0) + 1
        relojes = sorted({filas[i][1] for i in fuera})
        print(f"{f.name[:60]}: {len(fuera)} de {len(filas)} fuera de rango; por fase {por_fase}; relojes {relojes}")


def gm_ic(x: pd.Series) -> str:
    lg = np.log(x.to_numpy())
    h = stats.t.ppf(0.975, len(lg) - 1) * lg.std(ddof=1) / np.sqrt(len(lg))
    return (f"{np.exp(lg.mean()):.4f} (IC95 {np.exp(lg.mean() - h):.3f} a {np.exp(lg.mean() + h):.3f}; "
            f"favorables {int((lg < 0).sum())}/{len(lg)})")


def contraste() -> None:
    for nombre, f in (("performance", D / "fase4_20260926" / "fase4_EA.csv"),
                      ("powersave", D / "fase4_20260930" / "fase4_EA_powersave.csv")):
        d = pd.read_csv(f)
        d["edp"] = (d.e_cpu_j + d.e_gpu_j) * d.wall_s
        p = d.pivot_table(index="rep", columns="arm", values=["edp", "e_gpu_j", "wall_s"])
        print(f"== {nombre}")
        for m in ("edp", "e_gpu_j", "wall_s"):
            print(f"  {m:8s} F1 fijo/base {gm_ic(p[m]['fijo_gpu_f1'] / p[m]['base'])}")
            print(f"  {m:8s} agente/F1 fijo {gm_ic(p[m]['activo_gpu'] / p[m]['fijo_gpu_f1'])}")


def latencia_decision() -> None:
    for f in sorted(CELDAS.glob("*activo_gpu_*__phases.jsonl")):
        fases = [json.loads(x) for x in f.read_text().splitlines()]
        decisiones = [json.loads(x) for x in f.with_name(f.name.replace("phases.jsonl", "gpu_decisions.jsonl")).read_text().splitlines()]
        partes = []
        for p in fases:
            ds = [x for x in decisiones if p["begin_ns"] <= x["ts_ns"] <= p["end_ns"]]
            if not ds:
                partes.append(f"{p['kernel_id'][:14]} sin decisión")
                continue
            lat = (ds[0]["ts_ns"] - p["begin_ns"]) / 1e9
            reposo = lat - ds[0]["features"]["phase_active_for_s"]
            dur = (p["end_ns"] - p["begin_ns"]) / 1e9
            partes.append(f"{p['kernel_id'][:14]} decisión a {lat:.1f} s (reposo {reposo:.1f} s, fase {dur:.1f} s)")
        print(f"{f.name[:45]}: " + " | ".join(partes))


if __name__ == "__main__":
    muestras_fuera_de_rango()
    contraste()
    latencia_decision()
