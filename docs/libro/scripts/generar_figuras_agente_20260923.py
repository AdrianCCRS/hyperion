"""Figuras de la verificacion del agente y del candidato operativo de GPU (ronda 2026-09-23).

Valores tomados de las mediciones registradas en Plan_Fase3_Daemon.md:
  - candado de reloj de GPU: jobs 7599/7600 (rendimiento relativo a 1410 MHz);
  - invarianza frente al reloj: job 7601 (LOFO, 483 corridas, 16 familias).
Reproduce: python3 docs/libro/scripts/generar_figuras_agente_20260923.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F = Path(__file__).resolve().parents[1] / "figuras"
COMPUTE, MEMORY, REF = "#2b6cb0", "#dd6b20", "#a0aec0"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})


def candado() -> None:
    clk = [210, 510, 810, 1110, 1260, 1410]
    comp = [0.15, 0.37, 0.58, 0.79, 0.89, 1.00]
    mem = [0.72, 0.98, 1.01, 1.01, 1.01, 1.00]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(clk, [c / 1410 for c in clk], color=REF, ls="--", lw=1, label="Proporcional al reloj")
    ax.plot(clk, comp, "o-", color=COMPUTE, label="Carga compute_bound (GFLOP/s)")
    ax.plot(clk, mem, "s-", color=MEMORY, label="Carga memory_bound (GB/s)")
    ax.axvline(1260, color="#4a5568", lw=0.8, ls=":")
    ax.text(1250, 1.07, "F1 = 1260 MHz", ha="right", va="center", fontsize=8.5, color="#4a5568")
    ax.set_xlabel("Reloj del multiprocesador fijado (MHz)")
    ax.set_ylabel("Rendimiento relativo a 1410 MHz")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(F / "fig_gpu_candado_rendimiento_20260923.png")
    plt.close(fig)


def invarianza() -> None:
    cond = ["Cinco variables", "Sin reloj", "Sin reloj\nni potencia"]
    lr = [0.792, 0.654, np.nan]
    rf = [0.807, 0.812, 0.819]
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    b1 = ax.bar(x - w / 2, lr, w, color=COMPUTE, label="Regresión logística")
    b2 = ax.bar(x + w / 2, rf, w, color=MEMORY, label="Random forest")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", fontsize=8.5)
    ax.text(2 - w / 2, 0.52, "no aplica", ha="center", fontsize=8, color="#718096")
    ax.set_xticks(x)
    ax.set_xticklabels(cond)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Exactitud balanceada por celda (LOFO)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(F / "fig_gpu_invarianza_reloj_20260923.png")
    plt.close(fig)


if __name__ == "__main__":
    candado()
    invarianza()
