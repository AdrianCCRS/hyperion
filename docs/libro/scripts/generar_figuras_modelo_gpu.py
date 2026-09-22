"""Figuras del clasificador GPU historico (seccion de resultados, ronda 2026-09-22).

Lee tmp/gpu_final_rigor_20260922/ (generado por
fase2_clasificador/analysis/gpu_quality_report.py y
fase2_clasificador/analysis/gpu_final_rigor_audit.py, corridos en pacca) y
escribe docs/libro/figuras/fig_gpu_*_20260922.png.
Reproduce: python3 docs/libro/scripts/generar_figuras_modelo_gpu.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parents[2].parent
D = BASE / "tmp" / "gpu_final_rigor_20260922"
F = Path(__file__).resolve().parents[1] / "figuras"
COMPUTE, MEMORY, REF = "#2b6cb0", "#dd6b20", "#a0aec0"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})

NOMBRES = {
    "mayoritaria": "Regla mayoritaria", "arbol_prof1": "Árbol de profundidad 1",
    "regresion_log": "Regresión logística", "arbol_prof6": "Árbol de profundidad 6",
    "random_forest": "Random Forest", "extra_trees": "Extra Trees", "xgboost": "XGBoost",
}
ORDEN = ["mayoritaria", "arbol_prof1", "arbol_prof6", "xgboost", "extra_trees", "random_forest", "regresion_log"]


def comparacion_modelos() -> None:
    m = json.load(open(D / "matrix.json"))["summary"]
    val = [m[k]["mean_over_seeds"]["cell_balanced_acc"] for k in ORDEN]
    lo = [m[k]["ci95_family_bootstrap"]["cell_balanced_acc"][0] for k in ORDEN]
    hi = [m[k]["ci95_family_bootstrap"]["cell_balanced_acc"][1] for k in ORDEN]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = np.arange(len(ORDEN))[::-1]
    color = [REF if k in ("mayoritaria", "arbol_prof1") else (COMPUTE if k == "regresion_log" else "#718096") for k in ORDEN]
    ax.errorbar(val, y, xerr=[np.array(val) - lo, np.array(hi) - val], fmt="none", ecolor="#b9cbe2", elinewidth=2.2, capsize=3)
    ax.scatter(val, y, s=48, c=color, zorder=3)
    for v, yy in zip(val, y):
        ax.annotate(f"{v:.3f}", (v, yy), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#333")
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([NOMBRES[k] for k in ORDEN])
    ax.set_xlim(0.30, 1.02)
    ax.set_xlabel("Exactitud balanceada por celda familia-clase (IC95 por bootstrap de familias)")
    fig.tight_layout(); fig.savefig(F / "fig_gpu_comparacion_modelos_20260922.png"); plt.close(fig)


def matriz_confusion() -> None:
    audit = json.load(open(D / "final_rigor_audit.json"))
    c = audit["confusion_matrix_and_ece"]["confusion_matrix"]
    ext = audit["external_sealed_evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    pct = mat / mat.sum(axis=1, keepdims=True)
    ax = axes[0]
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}\n{pct[i, j]*100:.1f} %", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else "#1a202c", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["compute", "memory"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["compute", "memory"])
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    ax.set_title("LOFO, 483 corridas (candidato final)", fontsize=9.5); ax.grid(False)

    ax = axes[1]
    fams = list(ext["per_family_accuracy"].keys())
    accs = [ext["per_family_accuracy"][k] for k in fams]
    y = np.arange(len(fams))
    ax.barh(y, accs, color=[MEMORY if a < 0.6 else COMPUTE for a in accs])
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f.replace("rodinia_", "") for f in fams])
    ax.set_xlim(0, 1.02); ax.set_xlabel("Exactitud")
    ax.set_title(f"Validación externa sellada\n(n={ext['n_runs']}, exact. global={ext['accuracy']:.3f})", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(F / "fig_gpu_matriz_confusion_20260922.png"); plt.close(fig)


def por_familia() -> None:
    f = pd.read_csv(D / "regresion_log_by_family.csv").sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(f))
    # memory_share es la fraccion de corridas memory_bound de la familia;
    # >= 0.5 -> la familia es predominantemente memory_bound.
    col = [MEMORY if s >= 0.5 else COMPUTE for s in f["memory_share"]]
    ax.barh(y, f["accuracy"], color=col)
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=1)
    labels = [f"{fam} (n={n})" for fam, n in zip(f["family"], f["n"])]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Exactitud (LOFO, candidato final)")
    ax.legend(handles=[Patch(color=COMPUTE, label="Familia predominantemente compute_bound"),
                       Patch(color=MEMORY, label="Familia predominantemente memory_bound")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(F / "fig_gpu_resultado_por_familia_20260922.png"); plt.close(fig)


def calibracion_y_umbral() -> None:
    audit = json.load(open(D / "final_rigor_audit.json"))
    bins = pd.DataFrame(audit["confusion_matrix_and_ece"]["bins"])
    rc = pd.read_csv(D / "threshold_grid.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.3))
    a.plot([0.5, 1], [0.5, 1], color="#9a9a9a", ls="--", lw=1, label="Calibración perfecta")
    a.plot(bins["confianza_media"], bins["exactitud_media"], "o-", color=COMPUTE, lw=2, ms=5, label="Observado (LOFO)")
    a.set_xlabel("Confianza declarada"); a.set_ylabel("Exactitud observada")
    a.set_title(f"Calibración (ECE={audit['confusion_matrix_and_ece']['ece']:.3f})", fontsize=10)
    a.legend(frameon=False, fontsize=8.5)

    b.plot(rc["cell_coverage"], rc["cell_balanced_acc"], "o-", color=COMPUTE, lw=2, ms=5)
    for _, r in rc.iterrows():
        if r["threshold"] in (0.5, 0.7, 0.98):
            off = {0.5: (8, -14), 0.7: (8, 6), 0.98: (-30, -16)}[r["threshold"]]
            b.annotate(f"τ={r['threshold']:.2f}", (r["cell_coverage"], r["cell_balanced_acc"]),
                       textcoords="offset points", xytext=off, fontsize=8.5, color="#333")
    b.axhline(rc.loc[rc["threshold"] == 0.5, "cell_balanced_acc"].iloc[0], color=MEMORY, ls="--", lw=1)
    b.set_xlabel("Cobertura"); b.set_ylabel("Exactitud balanceada por celda"); b.invert_xaxis()
    b.set_title("Abstención: ganancia frente a cobertura", fontsize=10)
    fig.tight_layout(); fig.savefig(F / "fig_gpu_calibracion_umbral_20260922.png"); plt.close(fig)


if __name__ == "__main__":
    comparacion_modelos()
    matriz_confusion()
    por_familia()
    calibracion_y_umbral()
    print("figuras escritas en", F)
