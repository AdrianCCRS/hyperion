"""Contraste de gobernador nativo para el Escenario E-B (familias inéditas de memoria): powersave con EPP default frente a
performance (la línea base usada en el resto de la Fase 4). Complementa, no reemplaza, generar_figuras_fase4_20260926.py.

Entradas:
  docs/libro/datos/fase4_20260929/fase4_EB_powersave.csv   job 7779 (2026-09-28/29), governor powersave/EPP default fijado
                                                            en los 12 núcleos delegados (0-5, 16-21), verificado antes de medir
                                                            y restaurado al terminar. Mismos cinco brazos, kernels inéditos
                                                            (BabelStream, lavamd, myocyte) y 8 bloques aleatorizados que la
                                                            E-B ya medida bajo performance.
  docs/libro/datos/fase4_20260926/{fase4_E,fase4_EB,fase4_EBdiag}.csv  la misma E-B bajo el gobernador performance (turbo
                                                            activo), agrupada igual que en generar_figuras_fase4_20260926.py
                                                            (eb_agrupado): 21 repeticiones de base y sombra, 11 de activo.
Solo se usan celdas con state_ok = 1.
Reproduce: python3 docs/libro/scripts/generar_figuras_fase4_20260929.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fase4_evaluacion.analyze_matrix import load_results  # noqa: E402

L = ROOT / "docs" / "libro"
D26, D29, F = L / "datos" / "fase4_20260926", L / "datos" / "fase4_20260929", L / "figuras"
ARMS = {"base": "REF", "sombra": "Sombra", "activo_gpu": "Activo"}
COLOR = {"performance": "#a0aec0", "powersave": "#2f855a"}
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})


def _valid(path):
    return [r for r in load_results(path) if r["state_ok"] == "1"]


def cargar():
    """Devuelve {governor: {arm: [filas]}} para los tres brazos comunes (base, sombra, activo_gpu) de E-B inédito."""
    g = {"performance": {}, "powersave": {}}
    perf = []
    for name, arms in (("fase4_E", ("base", "sombra", "activo_gpu")), ("fase4_EB", ("base", "sombra", "activo_gpu")),
                        ("fase4_EBdiag", ("base", "sombra"))):
        perf += [r for r in _valid(D26 / f"{name}.csv") if r["set"] == "unseen" and r["scope"] == "gpumem" and r["arm"] in arms]
    for r in perf:
        g["performance"].setdefault(r["arm"], []).append(r)
    ps = [r for r in _valid(D29 / "fase4_EB_powersave.csv") if r["set"] == "unseen" and r["scope"] == "gpumem" and r["arm"] in ARMS]
    for r in ps:
        g["powersave"].setdefault(r["arm"], []).append(r)
    return g


def figura(g):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=False)
    arms = list(ARMS)
    refs = {gov: np.median([r["edp"] for r in g[gov]["base"]]) for gov in g}
    for ax, key, tit in zip(axes, ("edp", "e_gpu_j"), ("EDP del nodo", "Energía de GPU")):
        refs_k = {gov: np.median([r[key] for r in g[gov]["base"]]) for gov in g}
        w = 0.34
        for j, gov in enumerate(("performance", "powersave")):
            xs = np.arange(len(arms)) + (j - 0.5) * w
            vals = [np.median([r[key] for r in g[gov][a]]) / refs_k[gov] for a in arms]
            ax.bar(xs, vals, w, color=COLOR[gov], edgecolor="none", linewidth=0.6,
                   label=gov if key == "edp" else None)
            for x, v, a in zip(xs, vals, arms):
                n = len(g[gov][a])
                ax.text(x, v + 0.006, f"{v:.3f}\n(n={n})", ha="center", fontsize=6.3, linespacing=1.2)
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARMS[a] for a in arms])
        ax.set_title(f"{tit}\nrelativo a REF de su propio gobernador", fontsize=8.6)
        ax.set_ylim(0.93, 1.20)
    axes[0].set_ylabel("Razón frente a REF (mediana)")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02),
               title="Gobernador de CPU (delegado, EPP default)", title_fontsize=8)
    fig.tight_layout(rect=(0.01, 0.1, 0.99, 1))
    fig.savefig(F / "fig_fase4_EB_powersave_20260929.png")
    plt.close(fig)


def tabla(g):
    rows = []
    for gov in ("performance", "powersave"):
        refb = {k: np.median([r[k] for r in g[gov]["base"]]) for k in ("wall_s", "e_cpu_j", "e_gpu_j", "edp")}
        for a in ("base", "sombra", "activo_gpu"):
            rs = g[gov][a]
            med = {k: np.median([r[k] for r in rs]) for k in ("wall_s", "e_cpu_j", "e_gpu_j", "edp")}
            rows.append(f"{gov} & {ARMS[a]} & {len(rs)} & {med['wall_s']:.1f} & {med['e_cpu_j']:.0f} & {med['e_gpu_j']:.0f} & "
                        f"{med['wall_s'] / refb['wall_s']:.3f} & {med['edp'] / refb['edp']:.3f} \\\\")
        rows.append("\\midrule")
    body = "\n".join(rows[:-1])
    tex = ("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
           "\\textbf{Gobernador} & \\textbf{Brazo} & \\textbf{$n$} & \\textbf{$T$ (s)} & \\textbf{$E_{\\mathrm{CPU}}$ (J)} & "
           "\\textbf{$E_{\\mathrm{GPU}}$ (J)} & \\textbf{$T$/REF} & \\textbf{EDP/REF} \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n")
    (D29 / "tabla_EB_powersave.tex").write_text(tex)

    # REF bajo powersave frente a REF bajo performance: efecto puro del gobernador, sin agente.
    rb_perf = {k: np.median([r[k] for r in g["performance"]["base"]]) for k in ("wall_s", "e_cpu_j", "edp")}
    rb_ps = {k: np.median([r[k] for r in g["powersave"]["base"]]) for k in ("wall_s", "e_cpu_j", "edp")}
    (D29 / "cruce_governor_ref.tex").write_text(
        f"T={rb_ps['wall_s'] / rb_perf['wall_s']:.3f} & $E_{{\\mathrm{{CPU}}}}$={rb_ps['e_cpu_j'] / rb_perf['e_cpu_j']:.3f} & "
        f"EDP={rb_ps['edp'] / rb_perf['edp']:.3f}\n")


if __name__ == "__main__":
    g = cargar()
    figura(g)
    tabla(g)
    print("ok", {gov: {a: len(rs) for a, rs in arms.items()} for gov, arms in g.items()})
