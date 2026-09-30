"""Figura del escenario E-C y del contraste agente frente a F1 fijo (revisión C1 del director).

Paneles: (a) E-A, EDP del nodo de agente y F1 fijo frente a la REF del mismo bloque, bajo performance (job 7696) y
powersave (job 7789); (b) E-C, lo mismo con los seis bloques de los jobs 7815 (performance) y 7818 (powersave); (c)
duración de la fase de rodinia_heartwall en E-C por brazo (puntos llenos: performance; vacíos: powersave). Barras:
mediana; puntos: cada bloque. En E-A el brazo F1 fijo se valida con el
criterio restringido a las fases de memoria (ver analisis_f1_fijo_20260930.py); en E-C cumple el criterio completo.

Entradas:
  docs/libro/datos/fase4_20260926/fase4_EA.csv, docs/libro/datos/fase4_20260930/fase4_EA_powersave.csv
  docs/libro/datos/fase4_20260930/{EC_7815, EC_7818_powersave}/EC_confirmatorio/{results.csv, cells/*/phases.jsonl}
Reproduce: python3 docs/libro/scripts/generar_figura_EC_20260930.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generar_figuras_fase4_20260930 as g30  # noqa: E402  (rcParams, colores, barra y leyenda de gobernador)

L, F = g30.L, g30.F
D26, D30 = L / "datos" / "fase4_20260926", L / "datos" / "fase4_20260930"
EC = {"performance": D30 / "EC_7815" / "EC_confirmatorio", "powersave": D30 / "EC_7818_powersave" / "EC_confirmatorio"}
FIJO = "#6b46c1"  # violeta: el naranja ya significa memory_bound en la tesis (validado contra el verde del agente)
ARMS = (("base", "REF", g30.COL["base"]), ("activo_gpu", "Agente", g30.COL["activo_gpu"]), ("fijo_gpu_f1", "F1 fijo", FIJO))


def por_bloque(path):
    """{brazo: [EDP del nodo / EDP de la REF del mismo bloque]} (sin filtrar por state_ok: F1 fijo de E-A se valida aparte)."""
    rows = list(csv.DictReader(open(path)))
    edp = {(r["arm"], r["rep"]): (float(r["e_cpu_j"]) + float(r["e_gpu_j"])) * float(r["wall_s"]) for r in rows}
    reps = sorted({rep for _, rep in edp}, key=int)
    return {a: [edp[(a, rep)] / edp[("base", rep)] for rep in reps] for a, _, _ in ARMS}


def duracion_heartwall(ec):
    out = {}
    for a, _, _ in ARMS:
        vals = []
        for c in sorted(ec.glob(f"cells/gpumem_sensitive_{a}_*")):
            for linea in open(c / "phases.jsonl"):
                p = json.loads(linea)
                if p["kernel_id"] == "rodinia_heartwall":
                    vals.append((p["end_ns"] - p["begin_ns"]) / 1e9)
        out[a] = vals
    return out


def main():
    EA = {"performance": por_bloque(D26 / "fase4_EA.csv"), "powersave": por_bloque(D30 / "fase4_EA_powersave.csv")}
    ECb = {gov: por_bloque(EC[gov] / "results.csv") for gov in g30.GOVS}
    hw = {gov: duracion_heartwall(EC[gov]) for gov in g30.GOVS}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": (1.3, 1.3, 1)})
    w = 0.38
    ax = axes[0]
    for j, gov in enumerate(g30.GOVS):
        for i, (a, _, c) in enumerate(ARMS):  # la REF vale 1 por construcción: sin puntos ni etiqueta
            g30.barra(ax, i + (j - 0.5) * w * 1.08, EA[gov][a], c, w, gov, fs=5.8, off=0.003, puntos=a != "base", etiqueta=a != "base")
    ax.set_title("E-A: sin fase de cómputo\nsensible al reloj", fontsize=8.5)
    ax = axes[1]
    for j, gov in enumerate(g30.GOVS):
        for i, (a, _, c) in enumerate(ARMS):
            g30.barra(ax, i + (j - 0.5) * w * 1.08, ECb[gov][a], c, w, gov, fs=5.8, off=0.003, puntos=a != "base", etiqueta=a != "base")
    ax.set_title("E-C: con heartwall\n(cómputo sensible al reloj)", fontsize=8.5)
    for ax in axes[:2]:
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_ylim(0.93, 1.05)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([n for _, n, _ in ARMS], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("EDP del nodo relativo a la REF del bloque", fontsize=8)
    axes[1].set_yticklabels([])
    ax = axes[2]
    for i, (a, _, c) in enumerate(ARMS):  # puntos y mediana (no barras): el eje no parte de cero
        for j, gov in enumerate(g30.GOVS):
            vals = hw[gov][a]; med = float(np.median(vals)); x0 = i + (j - 0.5) * 0.4
            lleno = gov == "performance"
            ax.plot(x0 + np.linspace(-0.1, 0.1, len(vals)), sorted(vals), "o", ms=3.8, color=c if lleno else "white",
                    markeredgecolor="white" if lleno else c, markeredgewidth=0.6 if lleno else 1.0)
            ax.plot([x0 - 0.16, x0 + 0.16], [med, med], color="#1a202c", lw=1.2)
        med = float(np.median(hw["performance"][a] + hw["powersave"][a]))
        ax.text(i, med + 0.35, f"{med:.1f} s", ha="center", va="bottom", fontsize=6.3)
    ax.set_ylim(29, 34.5); ax.set_xlim(-0.6, 2.6)
    ax.set_title("E-C: duración de la fase\nheartwall", fontsize=8.5)
    ax.set_ylabel("Segundos", fontsize=8)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([n for _, n, _ in ARMS], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=7.5)
    g30.leyenda_gob(fig, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(F / "fig_fase4_EC_20260930.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
