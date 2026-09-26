"""Figuras del clasificador GPU final (random forest de tres entradas invariantes).

Lee docs/libro/datos/gpu_calidad_20260924/rf_operativo (probabilidades LOFO promediadas sobre
cinco semillas, calibración y resumen de umbrales) y escribe docs/libro/figuras/fig_gpu_rf_*.png.
Reproduce: python3 docs/libro/scripts/generar_figuras_gpu_rf.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parents[1]
D = BASE / "datos" / "gpu_calidad_20260924" / "rf_operativo"
F = BASE / "figuras"
COMPUTE, MEMORY, NEUTRO = "#2b6cb0", "#dd6b20", "#4a5568"
UMBRAL = 0.90
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})


def cargar() -> pd.DataFrame:
    d = pd.read_csv(D / "lofo_probabilities.csv")
    d["mem"] = d["truth"].eq("memory_bound")
    d["pred"] = d["p_memory"] > 0.5
    d["ok"] = d["mem"] == d["pred"]
    return d


def por_familia() -> None:
    d = cargar()
    f = (d.groupby("family")
          .agg(acc=("ok", "mean"), cov=("confidence", lambda c: (c >= UMBRAL).mean()), mem=("mem", "mean"))
          .sort_values(["acc", "cov"]))
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True,
                                  gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.06})
    y = np.arange(len(f))
    mixta = ((f["mem"] >= 0.1) & (f["mem"] <= 0.9)).to_numpy()
    for a_, values, xlabel in ((ax, f["acc"].to_numpy(), "Exactitud"),
                               (axc, f["cov"].to_numpy(), f"Cobertura con $\\tau$ = {UMBRAL:.2f}")):
        for yy in y[::2]:
            a_.axhspan(yy - 0.5, yy + 0.5, color="#f1f1f1", zorder=0)
        a_.hlines(y, 0, values, color="#c9c9c9", lw=0.7, zorder=1)
        a_.scatter(values[mixta], y[mixta], s=20, color=NEUTRO, zorder=3)
        a_.scatter(values[~mixta], y[~mixta], s=20, facecolor="white", edgecolor=NEUTRO, linewidth=1.2, zorder=3)
        a_.set_xlim(-0.02, 1.03); a_.set_ylim(-0.6, len(f) - 0.4)
        a_.set_xticks([0, 0.5, 1.0]); a_.set_xticklabels(["0", "0.5", "1"])
        a_.set_xlabel(xlabel); a_.grid(axis="y", visible=False)
    ax.axvline(0.5, color="#9a9a9a", ls="--", lw=0.9)
    ax.text(0.5, len(f) - 0.3, "azar", ha="center", va="bottom", fontsize=7, color="#777")
    ax.set_yticks(y); ax.set_yticklabels(f.index, fontsize=6.6)
    axc.tick_params(axis="y", left=False)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=NEUTRO, markeredgecolor=NEUTRO, label="Familia mixta"),
                       Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=NEUTRO,
                              markeredgewidth=1.2, label="Familia homogénea")],
              loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=True, facecolor="white", edgecolor="#dcdcdc", fontsize=7)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.96, bottom=0.14)
    fig.savefig(F / "fig_gpu_rf_resultado_por_familia.png"); plt.close(fig)


def matriz_confusion() -> None:
    d = cargar()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for ax, sub, titulo in ((axes[0], d, "Todas las decisiones"),
                            (axes[1], d[d["confidence"] >= UMBRAL], f"Solo confianza $\\geq$ {UMBRAL:.2f}")):
        c = pd.crosstab(sub["truth"], sub["pred"]).reindex(index=["compute_bound", "memory_bound"],
                                                            columns=[False, True], fill_value=0)
        mat = c.to_numpy()
        pct = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}\n{pct[i, j]*100:.1f} %", ha="center", va="center",
                        color="white" if pct[i, j] > 0.55 else "#1a202c", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["compute", "memory"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["compute", "memory"])
        ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
        ax.set_title(titulo, fontsize=10); ax.grid(False)
    fig.tight_layout(); fig.savefig(F / "fig_gpu_rf_matriz_confusion.png"); plt.close(fig)


def calibracion_y_umbral() -> None:
    cal = pd.read_csv(D / "calibration_bins.csv")
    rc = pd.read_csv(D / "threshold_summary.csv").sort_values("threshold")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.3))
    a.plot([0.5, 1], [0.5, 1], color="#9a9a9a", ls="--", lw=1)
    a.plot(cal["confianza_media"], cal["exactitud_media"], "-", color=COMPUTE, lw=1.2, zorder=2)
    a.scatter(cal["confianza_media"], cal["exactitud_media"], s=12 + 260 * cal["n"] / cal["n"].max(), color=COMPUTE, zorder=3)
    a.set_xlabel("Confianza declarada"); a.set_ylabel("Exactitud observada")
    a.set_title("Calibración fuera de familia", fontsize=10); a.legend(handles=[Line2D([], [], color="#9a9a9a", ls="--", lw=1, label="Calibración perfecta"),
                       Line2D([], [], color=COMPUTE, marker="o", ms=5, lw=1.2, label="Observado")],
             frameon=False, fontsize=8.5, loc="lower left")
    b.plot(rc["cell_coverage"], rc["cell_balanced_accuracy_decided"], "o-", color=COMPUTE, lw=2, ms=5)
    for _, r in rc.iterrows():
        if r["threshold"] in (0.5, 0.7, 0.9):
            off = {0.5: (2, -16), 0.7: (4, -15), 0.9: (-44, -4)}[r["threshold"]]
            b.annotate(f"τ={r['threshold']:.2f}", (r["cell_coverage"], r["cell_balanced_accuracy_decided"]),
                       textcoords="offset points", xytext=off, fontsize=8.5, color="#333")
    base = rc.loc[rc["threshold"] == 0.5, "cell_balanced_accuracy_decided"].iloc[0]
    b.axhline(base, color=MEMORY, ls="--", lw=1, label="Sin abstención (decide todo)")
    b.legend(frameon=False, fontsize=8.5, loc="upper left")
    b.set_ylim(0.804, 0.906)
    b.set_xlabel("Cobertura por celda"); b.set_ylabel("Exactitud balanceada"); b.invert_xaxis()
    b.set_title("Abstención: ganancia frente a cobertura", fontsize=10)
    fig.tight_layout(); fig.savefig(F / "fig_gpu_rf_calibracion_umbral.png"); plt.close(fig)


if __name__ == "__main__":
    por_familia(); matriz_confusion(); calibracion_y_umbral()
