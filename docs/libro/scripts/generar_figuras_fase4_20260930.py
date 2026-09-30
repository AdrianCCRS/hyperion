"""Figuras de la Fase 4 con los dos gobernadores nativos de intel_pstate lado a lado: performance (barra lisa, jobs 7696/7692)
y powersave con EPP default (barra rayada, job 7789; en E-B también job 7779). Cada brazo se expresa frente a la REF de su
propio gobernador.

Entradas:
  docs/libro/datos/fase4_20260926/{fase4_A,fase4_E,fase4_EB,fase4_EBdiag,fase4_D}.csv  performance
  docs/libro/datos/fase4_20260924/cloverleaf_confirm_7692.csv                          performance
  docs/libro/datos/fase4_20260930/fase4_{A,E,D,cloverleaf}_powersave.csv                powersave (job 7789)
  docs/libro/datos/fase4_20260929/fase4_EB_powersave.csv                               powersave E-B (job 7779)
  docs/libro/datos/gpu_calidad_20260922/politica/policy_by_family.json                  ganancia por nivel de GPU (Fase 2)
E-B se agrupa por la bimodalidad de myocyte: performance con las tres corridas de eb_agrupado() y powersave con las
repeticiones de 7779 y de la etapa E de 7789. Solo se usan celdas con state_ok = 1.
Reproduce: python3 docs/libro/scripts/generar_figuras_fase4_20260930.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generar_figuras_fase4_20260926 as g26  # noqa: E402  (fija rcParams, colores y cargadores de performance)
from fase4_evaluacion.analyze_matrix import load_results, summarize  # noqa: E402

L, F = g26.L, g26.F
D24, D26, D29, D30 = L / "datos" / "fase4_20260924", g26.D, g26.D29, L / "datos" / "fase4_20260930"
GOVS = ("performance", "powersave")
HATCH = {"performance": None, "powersave": "////"}
COL = {"base": g26.REF, "base_noturbo": "#cbd5e0", "sombra": "#718096", "activo": "#2f855a", "activo_gpu": "#2f855a",
       "activo_f0": "#68d391", "activo_nofloor": "#9ae6b4", "activo_gpu_cpuobs": "#9ae6b4"}
NOM = {"base": "REF", "base_noturbo": "REF sin turbo", "sombra": "Sombra", "activo": "Activo", "activo_gpu": "Activo",
       "activo_f0": "Activo F0", "activo_nofloor": "Activo sin piso", "activo_gpu_cpuobs": "Activo + agente de CPU"}


def valid(path):
    return [r for r in load_results(path) if r["state_ok"] == "1"]


def por_grupo(rows):
    g = {}
    for r in rows:
        g.setdefault((r["scope"], r["set"], r["arm"]), []).append(r)
    return g


MATRIZ = {"performance": lambda n: valid(D26 / f"fase4_{n}.csv"), "powersave": lambda n: valid(D30 / f"fase4_{n}_powersave.csv")}


def eb(gov):
    """E-B agrupado por gobernador: {brazo: [filas]}."""
    if gov == "performance":
        return g26.eb_agrupado()
    rows = [r for r in valid(D29 / "fase4_EB_powersave.csv") + valid(D30 / "fase4_E_powersave.csv")
            if r["scope"] == "gpumem" and r["set"] == "unseen"]
    out = {}
    for r in rows:
        out.setdefault(r["arm"], []).append(r)
    return out


def leyenda_gob(fig, **kw):
    fig.legend(handles=[Patch(facecolor="#a0aec0", label="performance"),
                        Patch(facecolor="#a0aec0", hatch=HATCH["powersave"], edgecolor="white", label="powersave (EPP default)")],
               frameon=False, fontsize=7.5, ncol=2, title="Gobernador nativo", title_fontsize=7.5, **kw)


def barra(ax, x, vals, color, w, gov, fs=6.3, puntos=True, etiqueta=True, off=0.006):
    med = float(np.median(vals))
    ax.bar(x, med, w, color=color, hatch=HATCH[gov], edgecolor="white" if HATCH[gov] else None, linewidth=0)
    if puntos:
        jit = np.linspace(-w * 0.27, w * 0.27, len(vals)) if len(vals) > 3 else np.zeros(len(vals))
        ax.plot(x + jit, sorted(vals), "o", color="#1a202c", ms=2.1 if len(vals) > 3 else 2.6)
    if etiqueta:
        ax.text(x, (max(vals) if puntos else med) + off, f"{med:.3f}", ha="center", va="bottom", fontsize=fs, rotation=90)


def edp_alcance():
    """Matriz inicial: EDP del nodo por alcance, kernels y brazo, con REF implícita en la paridad."""
    s = {gov: {(x["scope"], x["set"], x["arm"]): x for x in summarize(MATRIZ[gov]("A"))} for gov in GOVS}
    brazos = {"cpu": ("sombra", "activo", "activo_f0"), "gpu": ("sombra", "activo"), "joint": ("sombra", "activo", "activo_nofloor")}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True, gridspec_kw={"width_ratios": (3, 2, 3)})
    for ax, sc in zip(axes, ("cpu", "gpu", "joint")):
        arms = brazos[sc]
        n = 2 * len(arms)
        w = 0.84 / n
        for j, a in enumerate(arms):
            for k, gov in enumerate(GOVS):
                off = (2 * j + k - (n - 1) / 2) * w
                for i, ks in enumerate(("known", "unseen")):
                    v = s[gov][(sc, ks, a)]["ratio_EDP"]
                    ax.bar(i + off, v, w, color=COL[a], hatch=HATCH[gov], edgecolor="white" if HATCH[gov] else None, linewidth=0)
                    ax.text(i + off, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_xticks(range(2)); ax.set_xticklabels(["A", "B"])
        ax.set_title(g26.NOM[sc], fontsize=10)
        ax.set_ylim(0.95, 1.22)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("EDP del nodo relativo a REF")
    arms_all = ("sombra", "activo", "activo_f0", "activo_nofloor")
    fig.legend(handles=[Patch(color=COL[a], label=NOM[a]) for a in arms_all], loc="lower left", bbox_to_anchor=(0.01, 0.0),
               ncol=4, frameon=False, fontsize=7.5)
    leyenda_gob(fig, loc="lower right", bbox_to_anchor=(0.99, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(F / "fig_fase4_edp_alcance_20260930.png")
    plt.close(fig)


def gpu_potencial():
    pol = json.load(open(L / "datos" / "gpu_calidad_20260922" / "politica" / "policy_by_family.json"))["memory_bound"]["levels"]
    lv = [k for k in pol if int(k[1:]) <= 6]
    g = np.array([pol[k]["gain"] for k in lv]) * 100
    lo = np.array([pol[k]["gain_ci95"][0] for k in lv]) * 100
    hi = np.array([pol[k]["gain_ci95"][1] for k in lv]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.errorbar(range(len(lv)), g, yerr=[g - lo, hi - g], fmt="o", color=g26.MEMORY, capsize=3)
    ax.axhline(0, color="#4a5568", lw=0.8, ls=":")
    ax.set_xticks(range(len(lv))); ax.set_xticklabels(lv)
    ax.set_xlabel("Nivel de reloj de GPU (F1 = 1260 MHz)")
    ax.set_ylabel("Ganancia de EDP en memory_bound (%)")
    ax.set_title("Fase 2: ganancia por nivel", fontsize=9.5)
    share = {"known": 18.0 / 49.5, "unseen": 8.2 / 16.9}
    ax2 = axes[1]
    w = 0.17
    x = np.arange(2)
    exp = [share[k] * g[1] for k in ("known", "unseen")]
    ax2.bar(x - 2 * w, exp, w, color="#cbd5e0")
    for xi, v in zip(x - 2 * w, exp):
        ax2.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    for k, gov in enumerate(GOVS):
        s = {(r["scope"], r["set"], r["arm"]): r for r in summarize(MATRIZ[gov]("A"))}
        for m, (clave, color) in enumerate((("ratio_EDPgpu", "#2f855a"), ("ratio_EDP", "#4a5568"))):
            vals = [100 * (1 - s[("gpu", ks, "activo")][clave]) for ks in ("known", "unseen")]
            xs = x + (-1 + 2 * m + k) * w
            ax2.bar(xs, vals, w, color=color, hatch=HATCH[gov], edgecolor="white" if HATCH[gov] else None, linewidth=0)
            for xi, v in zip(xs, vals):
                ax2.text(xi, max(v, 0) + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    ax2.axhline(0, color="#4a5568", lw=0.8, ls=":")
    ax2.set_xticks(x); ax2.set_xticklabels(["A", "B"])
    ax2.set_ylabel("Ganancia de EDP (%)")
    ax2.set_title("Fase 4, alcance GPU", fontsize=9.5)
    ax2.set_ylim(-1.5, 7)
    ax2.legend(handles=[Patch(color="#cbd5e0", label="Esperada (EDP de GPU)"), Patch(color="#2f855a", label="Medida, EDP de GPU"),
                        Patch(color="#4a5568", label="Medida, EDP del nodo")], frameon=False, fontsize=7, loc="upper left")
    leyenda_gob(fig, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(F / "fig_fase4_gpu_potencial_20260930.png")
    plt.close(fig)


def escenario_e():
    A = {gov: por_grupo(MATRIZ[gov]("E")) for gov in GOVS}
    B = {gov: eb(gov) for gov in GOVS}
    armsA = ["base", "base_noturbo", "sombra", "activo_gpu", "activo_gpu_cpuobs"]
    labA = ["REF", "REF sin\nturbo", "Sombra", "Activo", "Activo +\nag. CPU"]
    armsB = ["base", "sombra", "activo_gpu"]
    labB = [f"{n}\n(n={len(B['performance'][a])} | {len(B['powersave'][a])})" for n, a in zip(("REF", "Sombra", "Activo"), armsB)]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), gridspec_kw={"width_ratios": (1.55, 1)})
    w = 0.38
    for row, (key, tit) in enumerate((("edp", "EDP del nodo"), ("e_gpu_j", "Energía de GPU"))):
        for col, (arms, labs, datos, nombre) in enumerate(((armsA, labA, lambda gov, a: A[gov][("gpumem", "known", a)], "A (vistos)"),
                                                          (armsB, labB, lambda gov, a: B[gov][a], "B (inéditos)"))):
            ax = axes[row, col]
            for j, gov in enumerate(GOVS):
                ref = np.median([r[key] for r in datos(gov, "base")])
                for i, a in enumerate(arms):
                    barra(ax, i + (j - 0.5) * w * 1.08, [r[key] / ref for r in datos(gov, a)], COL[a], w, gov, fs=5.8, off=0.018)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(labs if row == 1 else [], fontsize=7)
            ax.set_title(f"{tit}, {nombre}", fontsize=9)
            ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
            ax.set_ylim(0.8, 1.32)
            ax.grid(axis="x", visible=False)
    leyenda_gob(fig, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.03, 0.06, 1, 1))
    fig.text(0.012, 0.53, "Relativo a la REF de su gobernador (mediana y cada repetición)", rotation=90, va="center", fontsize=8.5)
    fig.savefig(F / "fig_fase4_E_20260930.png")
    plt.close(fig)


def escenario_d():
    G = {gov: por_grupo(MATRIZ[gov]("D")) for gov in GOVS}
    entradas = ["chain", "eam", "lj", "rhodo"]
    arms = ["sombra", "activo_gpu", "activo_gpu_cpuobs"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    n = 2 * len(arms)
    w = 0.84 / n
    for e_i, e in enumerate(entradas):
        for j, a in enumerate(arms):
            for k, gov in enumerate(GOVS):
                ref = np.median([r["edp"] for r in G[gov][("lammps", e, "base")]])
                barra(ax, e_i + (2 * j + k - (n - 1) / 2) * w, [r["edp"] / ref for r in G[gov][("lammps", e, a)]], COL[a], w, gov, fs=5.6)
    ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
    ax.set_xticks(range(len(entradas))); ax.set_xticklabels(entradas)
    ax.set_ylabel("EDP del nodo relativo a REF")
    ax.set_ylim(0.96, 1.075)
    ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(color=COL[a], label=l) for a, l in zip(arms, ("Sombra", "Activo GPU", "Activo GPU + agente de CPU"))],
               loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, frameon=False, fontsize=7.5)
    leyenda_gob(fig, loc="lower right", bbox_to_anchor=(0.99, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(F / "fig_fase4_D_20260930.png")
    plt.close(fig)


def _cloverleaf(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["block"] = int(r["block"])
        for k in ("wall_s", "e_cpu_j", "e_gpu_j"):
            r[k] = float(r[k])
        r["edp"] = (r["e_cpu_j"] + r["e_gpu_j"]) * r["wall_s"]
    ref = {r["block"]: r for r in rows if r["arm"] == "ref"}
    return {a: {k: [r[k] / ref[r["block"]][k] for r in rows if r["arm"] == a] for k in ("wall_s", "e_cpu_j", "e_gpu_j", "edp")}
            for a in ("ref", "sombra", "activo")}


def cloverleaf():
    C = {"performance": _cloverleaf(D24 / "cloverleaf_confirm_7692.csv"), "powersave": _cloverleaf(D30 / "fase4_cloverleaf_powersave.csv")}
    arms = [("ref", "base"), ("sombra", "sombra"), ("activo", "activo")]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.5), sharey=True)
    w = 0.4
    for ax, (key, tit) in zip(axes, (("wall_s", "Duración"), ("e_cpu_j", "Energía de CPU"), ("e_gpu_j", "Energía de GPU"), ("edp", "EDP del nodo"))):
        for j, gov in enumerate(GOVS):
            for i, (a, c) in enumerate(arms):
                barra(ax, i + (j - 0.5) * w * 1.08, C[gov][a][key], COL[c], w, gov, fs=5.8, off=0.004)
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_title(tit, fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(["REF", "Sombra", "Activo"], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0.9, 1.06)
    axes[0].set_ylabel("Relativo a la REF del bloque", fontsize=8.5)
    leyenda_gob(fig, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(F / "fig_fase4_cloverleaf_20260930.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    edp_alcance(); gpu_potencial(); escenario_e(); escenario_d(); cloverleaf()
    print("ok")
