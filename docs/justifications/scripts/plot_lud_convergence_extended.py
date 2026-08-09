#!/usr/bin/env python3
"""Grafico de convergencia extendida de rodinia_lud (5..200 lanzamientos)."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_CSV = Path(__file__).parent.parent / "data" / "ncu_launch_count" / "rodinia_lud_convergence_extended.csv"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_PCT = 1.0


def main():
    rows = list(csv.DictReader(open(DATA_CSV)))
    launch_counts = [int(r["launch_count"]) for r in rows]
    oi = [float(r["operational_intensity"]) for r in rows]

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(launch_counts, oi, marker="o", color=color_for(0), linewidth=2)
    ax.axvline(150, color=color_for(3), linestyle="--", linewidth=1.5,
               label="Convergencia (Δ<1% entre puntos, criterio pre-declarado)")
    ax.set_xlabel("Número de lanzamientos perfilados (--launch-count)")
    ax.set_ylabel("Intensidad operacional (FLOP/byte)")
    ax.set_title("rodinia_lud: convergencia extendida hasta 200 lanzamientos")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lud_ncu_convergence_extended.png")
    plt.close(fig)
    print(f"Escrito: {PLOTS_DIR / 'lud_ncu_convergence_extended.png'}")


if __name__ == "__main__":
    main()
