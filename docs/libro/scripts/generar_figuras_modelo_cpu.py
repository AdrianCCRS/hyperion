"""Figuras del clasificador CPU (sección de resultados).

Lee docs/libro/datos/cpu_calidad_30fam y escribe docs/libro/figuras/fig_cpu_*_20260919.png.
Reproduce: python3 docs/libro/scripts/generar_figuras_modelo_cpu.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[1]
D = BASE / "datos" / "cpu_calidad_30fam"
F = BASE / "figuras"
COMPUTE, MEMORY, NEUTRO, REF = "#2b6cb0", "#dd6b20", "#4a5568", "#a0aec0"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})

NOMBRES = {
    "majority": "Regla mayoritaria", "stump_base": "Árbol de profundidad 1",
    "logistic_base": "Regresión logística (6 var.)", "logistic_inter": "Regresión logística (12 var.)",
    "xgb_base": "XGBoost (6 var.)", "xgb_inter": "XGBoost (12 var.)",
    "rf_inter": "Random Forest (12 var.)", "et_inter": "Extra Trees (12 var.)",
}
ORDEN = ["majority", "stump_base", "logistic_base", "logistic_inter", "et_inter", "rf_inter", "xgb_base", "xgb_inter"]


def comparacion_modelos() -> None:
    m = json.load(open(D / "matrix.json"))["summary"]
    val = [m[k]["full"]["mean_over_seeds"]["cell_balanced_acc"] for k in ORDEN]
    lo = [m[k]["full"]["ci95_family_bootstrap"]["cell_balanced_acc"][0] for k in ORDEN]
    hi = [m[k]["full"]["ci95_family_bootstrap"]["cell_balanced_acc"][1] for k in ORDEN]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(ORDEN))[::-1]
    color = [REF if k in ("majority", "stump_base") else COMPUTE for k in ORDEN]
    ax.errorbar(val, y, xerr=[np.array(val) - lo, np.array(hi) - val], fmt="none", ecolor="#b9cbe2", elinewidth=2.2, capsize=3)
    ax.scatter(val, y, s=42, c=color, zorder=3)
    for v, yy in zip(val, y):
        ax.annotate(f"{v:.3f}", (v, yy), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#333")
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([NOMBRES[k] for k in ORDEN])
    ax.set_xlim(0.42, 0.88)
    ax.set_xlabel("Exactitud balanceada por celda familia-clase")
    fig.tight_layout(); fig.savefig(F / "fig_cpu_comparacion_modelos_20260919.png"); plt.close(fig)


def matriz_confusion() -> None:
    d = json.load(open(D / "final_model.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for ax, clave, titulo in ((axes[0], "full_coverage", "Todas las decisiones"),
                              (axes[1], "selective", "Solo confianza ≥ 0.85")):
        c = d[clave]["confusion"]
        mat = np.array([[c["compute_as_compute"], c["compute_as_memory"]],
                        [c["memory_as_compute"], c["memory_as_memory"]]])
        pct = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]/1000:,.0f}k\n{pct[i, j]*100:.1f} %", ha="center", va="center",
                        color="white" if pct[i, j] > 0.55 else "#1a202c", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["compute", "memory"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["compute", "memory"])
        ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
        ax.set_title(titulo, fontsize=10); ax.grid(False)
    fig.tight_layout(); fig.savefig(F / "fig_cpu_matriz_confusion_20260919.png"); plt.close(fig)


def por_familia() -> None:
    f = pd.read_csv(D / "final_model_by_family.csv").sort_values("accuracy")
    inv = pd.read_csv(D / "inventory_by_family.csv").set_index("family")
    fig, ax = plt.subplots(figsize=(7.4, 6.9))
    y = np.arange(len(f))
    col = [COMPUTE if inv.loc[k, "mixed_10_90"] else MEMORY for k in f["family"]]
    ax.barh(y - 0.19, f["accuracy"], height=0.38, color=col)
    ax.barh(y + 0.19, f["coverage"], height=0.38, color="#cbd5e0")
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(f["family"].str.replace("_", r"\_", regex=False).str.replace(r"\_", "_", regex=False))
    ax.set_xlabel("Proporción")
    ax.set_xlim(0, 1.02)
    ax.legend(handles=[Patch(color=COMPUTE, label="Exactitud, familia mixta"),
                       Patch(color=MEMORY, label="Exactitud, familia de una sola clase"),
                       Patch(color="#cbd5e0", label="Cobertura (confianza ≥ 0.85)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(F / "fig_cpu_resultado_por_familia_20260919.png"); plt.close(fig)


def umbral_y_calibracion() -> None:
    rc = pd.read_csv(D / "risk_coverage.csv"); cal = pd.read_csv(D / "calibration_bins.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.3))
    a.plot([0.5, 1], [0.5, 1], color="#9a9a9a", ls="--", lw=1, label="Calibración perfecta")
    a.plot(cal["mean_conf"], cal["accuracy"], "o-", color=COMPUTE, lw=2, ms=5, label="Observado")
    a.set_xlabel("Confianza declarada"); a.set_ylabel("Exactitud observada")
    a.set_title("Calibración fuera de familia", fontsize=10); a.legend(frameon=False, fontsize=8.5)
    b.plot(rc["coverage_pooled"], rc["cell_balanced_acc"], "o-", color=COMPUTE, lw=2, ms=5)
    for _, r in rc.iterrows():
        if r["threshold"] in (0.5, 0.85, 0.99):
            b.annotate(f"τ={r['threshold']:.2f}", (r["coverage_pooled"], r["cell_balanced_acc"]),
                       textcoords="offset points", xytext=(6, -11), fontsize=8.5, color="#333")
    b.axhline(rc.loc[rc["threshold"] == 0.5, "cell_balanced_acc"].iloc[0], color=MEMORY, ls="--", lw=1)
    b.set_xlabel("Cobertura"); b.set_ylabel("Exactitud balanceada"); b.invert_xaxis()
    b.set_title("Abstención: qué se gana y qué se deja de decidir", fontsize=10)
    fig.tight_layout(); fig.savefig(F / "fig_cpu_umbral_calibracion_20260919.png"); plt.close(fig)


def curva_aprendizaje() -> None:
    lc = pd.read_csv(D / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.errorbar(lc["k"], lc["mean"], yerr=lc["ci95_half"], fmt="o-", color=COMPUTE, lw=2, ms=5, capsize=3)
    ax.set_xlabel("Familias algorítmicas en el ajuste")
    ax.set_ylabel("Exactitud balanceada\nen la familia retirada")
    ax.set_xticks(lc["k"])
    fig.tight_layout(); fig.savefig(F / "fig_cpu_curva_aprendizaje_20260919.png"); plt.close(fig)


def margen_ridge() -> None:
    e = pd.read_csv(D / "error_by_ridge_margin.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    col = [MEMORY if s > 0.5 else COMPUTE for s in e["share_memory"]]
    ax.bar(range(len(e)), e["accuracy"], color=col, width=0.72)
    for i, (a_, n) in enumerate(zip(e["accuracy"], e["share_of_all"])):
        ax.annotate(f"{n*100:.0f} %", (i, a_), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="#555")
    ax.axhline(0.5, color="#9a9a9a", ls="--", lw=1)
    etiquetas = e["margin_bin"].astype(str).str.replace("inf", "∞").str.replace(".0", "", regex=False)
    ax.set_xticks(range(len(e))); ax.set_xticklabels(etiquetas, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Exactitud"); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\log_2(\mathrm{OI}/\mathrm{ridge})$ del intervalo")
    ax.legend(handles=[Patch(color=MEMORY, label="intervalos memory-bound"),
                       Patch(color=COMPUTE, label="intervalos compute-bound")], loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / "fig_cpu_margen_ridge_20260919.png"); plt.close(fig)


def celdas() -> None:
    """Las celdas familia-clase: acierto de cada una y su tamano."""
    c = pd.read_csv(D / "metrics_cells.csv")
    orden = (c.groupby("familia")["acierto"].mean().sort_values().index.tolist())
    y = {f: i for i, f in enumerate(orden)}
    fig, ax = plt.subplots(figsize=(7.4, 8.6))
    for _, r in c.iterrows():
        col = MEMORY if r["clase_memory"] else COMPUTE
        size = 18 + 26 * np.log10(max(r["n_intervalos"], 1))
        ax.scatter(r["acierto"], y[r["familia"]], s=size, color=col, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=1)
    ax.set_yticks(range(len(orden))); ax.set_yticklabels(orden, fontsize=8.5)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel("Fracción de intervalos de la celda clasificados correctamente")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=COMPUTE, markersize=8, label="celda compute-bound"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=MEMORY, markersize=8, label="celda memory-bound")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=8.5,
              title="El tamaño del punto indica el número de intervalos de la celda (escala logarítmica)", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(F / "fig_cpu_celdas_20260919.png"); plt.close(fig)


if __name__ == "__main__":
    comparacion_modelos(); matriz_confusion(); por_familia()
    umbral_y_calibracion(); curva_aprendizaje(); margen_ridge(); celdas()
    print("figuras escritas en", F)
