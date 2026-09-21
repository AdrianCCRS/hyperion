"""Figuras de energía y potencia de la política de frecuencia CPU.

Lee docs/libro/datos/cpu_calidad_30fam/politica (tabla de política, sin las cargas de duración fija)
y docs/libro/datos/cpu_calidad_20260918/{politica,potencia} y escribe
docs/libro/figuras/fig_cpu_energia_*_20260920.png.
Reproduce: python3 docs/libro/scripts/generar_figuras_energia_cpu.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

BASE = Path(__file__).resolve().parents[1]
D = BASE / "datos" / "cpu_calidad_20260918"
F = BASE / "figuras"
COMPUTE, MEMORY, NEUTRO = "#2b6cb0", "#dd6b20", "#4a5568"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})
NIVELES = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
GHZ = dict(zip(NIVELES, [3.2, 2.9, 2.6, 2.3, 2.0, 1.7, 1.4, 1.1, 0.8]))
CALIBRACION = ("stream_official", "ert_probe")


def relativos_a_referencia() -> None:
    p = pd.read_csv(BASE / "datos" / "cpu_calidad_30fam" / "politica" / "policy_by_level.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), sharey=True)
    for ax, cls, titulo in ((axes[0], "compute_bound", "Kernels compute-bound"),
                            (axes[1], "memory_bound", "Kernels memory-bound")):
        s = p[p["cls"] == cls].set_index("level").loc[NIVELES]
        x = [GHZ[n] for n in NIVELES]
        ax.plot(x, s["time_ratio"], "s-", color=NEUTRO, lw=1.8, ms=4, label="Tiempo")
        ax.plot(x, s["energy_ratio"], "o-", color=COMPUTE, lw=2, ms=5, label="Energía")
        ax.plot(x, s["median_ratio"], "^-", color=MEMORY, lw=1.8, ms=5, label="Producto energía-retardo")
        ax.axhline(1.0, color="#9a9a9a", ls="--", lw=1)
        ax.set_title(f"{titulo} ({int(s['n_kernels'].iloc[0])})", fontsize=10)
        ax.set_xlabel("Frecuencia fija (GHz)")
        ax.invert_xaxis()
    axes[0].set_ylabel("Relativo a la referencia (mediana)")
    axes[0].set_yscale("log"); axes[0].set_ylim(0.9, 9.5)
    axes[0].set_yticks([1, 2, 4, 8]); axes[0].set_yticklabels(["1", "2", "4", "8"])
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(F / "fig_cpu_energia_relativa_20260920.png"); plt.close(fig)


def potencia_y_piso() -> None:
    r = pd.concat([pd.read_csv(D / "politica" / "runs_energy.csv"),
                   pd.read_csv(D / "politica" / "runs_energy_sweep_20260919.csv")], ignore_index=True)
    r = r[r["accepted"].astype(bool) & r["level"].isin(NIVELES) & ~r["kernel_ref"].isin(CALIBRACION)].copy()
    r["w"] = (r["pkg_uj"] + r["dram_uj"]) / (r["elapsed_ns"] / 1e3)   # uJ/us = W
    pk = r.groupby(["kernel_ref", "level"])["w"].median().unstack()[NIVELES]
    m = pd.read_csv(D / "potencia" / "power_model_by_kernel.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    x = [GHZ[n] for n in NIVELES]
    for _, fila in pk.iterrows():
        a.plot(x, fila.to_numpy(), color="#cbd5e0", lw=0.8, zorder=1)
    a.plot(x, pk.median().to_numpy(), "o-", color=COMPUTE, lw=2.2, ms=5, zorder=3, label="Mediana entre kernels")
    p0 = float(m["p0_w"].median())
    a.axhline(p0, color=MEMORY, ls="--", lw=1.4, zorder=2, label=f"Piso estático ({p0:.0f} W)")
    a.set_ylim(0, pk.max().max() * 1.05); a.invert_xaxis()
    a.set_xlabel("Frecuencia fija (GHz)"); a.set_ylabel("Potencia de paquete y DRAM (W)")
    a.legend(frameon=False, fontsize=8.5, loc="lower left")
    b.hist(m["static_frac_f0"], bins=np.linspace(0.6, 1.0, 17), color=COMPUTE, edgecolor="white")
    b.axvline(float(m["static_frac_f0"].median()), color=MEMORY, ls="--", lw=1.4)
    b.set_xlabel("Fracción estática a 3.2 GHz"); b.set_ylabel("Kernels")
    fig.tight_layout(); fig.savefig(F / "fig_cpu_energia_piso_20260920.png"); plt.close(fig)
    print("mediana P a 3.2 y 0.8 GHz:", round(pk["F0"].median(), 1), round(pk["F8"].median(), 1),
          "| static_frac mediana", round(m["static_frac_f0"].median(), 3), "| kernels", len(pk))


if __name__ == "__main__":
    relativos_a_referencia(); potencia_y_piso()
