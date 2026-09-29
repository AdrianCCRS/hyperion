"""Contraste de gobernador nativo para el Escenario E-B (familias inéditas de memoria): powersave con EPP default frente a
performance (la línea base usada en el resto de la Fase 4). Genera la tabla y el cruce de REF; la figura va en la columna B de
fig_fase4_E_20260929.png (escenario_e en generar_figuras_fase4_20260926.py).

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

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fase4_evaluacion.analyze_matrix import load_results  # noqa: E402

L = ROOT / "docs" / "libro"
D26, D29 = L / "datos" / "fase4_20260926", L / "datos" / "fase4_20260929"
ARMS = {"base": "REF", "sombra": "Sombra", "activo_gpu": "Activo"}


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
    tabla(g)
    print("ok", {gov: {a: len(rs) for a, rs in arms.items()} for gov, arms in g.items()})
