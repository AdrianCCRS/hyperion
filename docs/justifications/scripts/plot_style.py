"""Estilo compartido de graficos para docs/justifications (PDF/LaTeX, modo claro).

Paleta categorica en orden fijo (misma paleta de referencia validada del
skill de dataviz: blue, orange, aqua, yellow, magenta, green, violet, red),
nunca ciclada mas alla de 8 series -- si un grafico necesitara mas de 8
series, se divide en pequenos multiplos en vez de generar un color nuevo.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#dedcd5"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })


def color_for(index: int) -> str:
    return CATEGORICAL[index % len(CATEGORICAL)]
