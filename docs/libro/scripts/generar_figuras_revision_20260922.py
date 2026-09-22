"""Figuras nuevas de la revisión crítica de CPU (2026-09-22).

Lee docs/libro/datos/cpu_calidad_30fam y escribe docs/libro/figuras/fig_cpu_*_20260922.png.
Reproduce: python3 docs/libro/scripts/generar_figuras_revision_20260922.py

Genera cuatro figuras:
  1. Perfiles de las celdas confusables (perfiles gemelos), coordenadas paralelas.
  2. Forest plot de la ganancia del producto energia-retardo por nivel y clase.
  3. Exactitud balanceada por celda y cobertura frente al nivel de frecuencia (doble eje).
  4. Scatter Roofline (rendimiento vs. intensidad operacional) en REF y F8, con los
     techos calibrados. Usa datos/cpu_calidad_30fam/roofline_scatter/ (submuestreo
     estratificado 20260918, reconstruido desde training_cpu_intervals.csv de la
     campana pacca_cpu_final_20260913 en pacca, que trae FLOPs/bytes por intervalo
     y no sobrevive al contrato de columnas del clasificador).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D = BASE / "datos" / "cpu_calidad_30fam"
F = BASE / "figuras"
COMPUTE, MEMORY, NEUTRO = "#2b6cb0", "#dd6b20", "#4a5568"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})


def perfiles_gemelos() -> None:
    med = pd.read_csv(D / "median_pmu_by_family_class.csv")
    med["clase"] = med["y"].map({True: "memory-bound", False: "compute-bound"})
    pares = [
        ("gap_pr", "memory-bound", "phasic", "compute-bound", "gap_pr (memory) vs. fase compute de phasic"),
        ("rajaperf_basic_mat_mat_shared", "memory-bound", "rajaperf_basic_pi_reduce", "compute-bound",
         "celda rara memory de mat_mat_shared vs. pi_reduce (compute)"),
    ]
    cols = ["ipc", "mpki", "cache_miss_rate", "stall_mem_ratio"]
    etiquetas = ["IPC", "MPKI", "Tasa de\nfallos", "Stall\nmem."]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, (fa, ca, fb, cb, titulo) in zip(axes, pares):
        ra = med[(med["family"] == fa) & (med["clase"] == ca)][cols].iloc[0]
        rb = med[(med["family"] == fb) & (med["clase"] == cb)][cols].iloc[0]
        # normaliza cada variable al rango observado en las 30 familias para que las escalas sean comparables
        allmed = med[cols]
        lo, hi = allmed.min(), allmed.max()
        na = (ra - lo) / (hi - lo)
        nb = (rb - lo) / (hi - lo)
        x = np.arange(len(cols))
        ax.plot(x, na.values, "o-", color=MEMORY if "memory" in ca else COMPUTE, lw=2, ms=6,
                label=f"{fa}\n({ca})")
        ax.plot(x, nb.values, "s--", color=MEMORY if "memory" in cb else COMPUTE, lw=2, ms=6,
                label=f"{fb}\n({cb})")
        ax.set_xticks(x); ax.set_xticklabels(etiquetas, fontsize=8.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(titulo, fontsize=9)
        ax.legend(fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    axes[0].set_ylabel("Valor normalizado (min-max entre las 30 familias)")
    fig.tight_layout()
    fig.savefig(F / "fig_cpu_perfiles_gemelos_20260922.png", bbox_inches="tight")
    plt.close(fig)


def forest_politica() -> None:
    d = json.load(open(D / "politica" / "policy_cpu.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, clase, titulo, color in (
        (axes[0], "cpu-compute_bound", "Compute (19 kernels)", COMPUTE),
        (axes[1], "cpu-memory_bound", "Memory (28 kernels)", MEMORY),
    ):
        levels = d["policy"][clase]["levels_tested"]
        names = list(levels.keys())
        gain = [levels[n]["gain_agg"] for n in names]
        lo = [levels[n]["gain_ci95"][0] for n in names]
        hi = [levels[n]["gain_ci95"][1] for n in names]
        y = np.arange(len(names))[::-1]
        xerr = [np.array(gain) - np.array(lo), np.array(hi) - np.array(gain)]
        ax.errorbar(gain, y, xerr=xerr, fmt="o", color=color, ecolor="#b0b0b0", elinewidth=1.6, capsize=3, ms=5)
        ax.axvline(0, color="#333", lw=1)
        ax.axvline(1.0, color="#999", lw=0.8, ls=":")
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Ganancia del EDP frente a REF (%)")
    # el eje x difiere por orden de magnitud entre niveles bajos y altos: usar escala simetrica en signo
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1.0)
    fig.suptitle("Ganancia del producto energía--retardo por nivel, con IC95 (bootstrap de kernels)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(F / "fig_cpu_politica_forest_20260922.png", bbox_inches="tight")
    plt.close(fig)


def frecuencia_doble_eje() -> None:
    df = pd.read_csv(D / "by_frequency_level.csv")
    order = ["REF", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    df["freq_level"] = pd.Categorical(df["freq_level"], categories=order, ordered=True)
    df = df.sort_values("freq_level")
    cov = pd.read_csv(D / "final_model_by_frequency.csv")
    cov["freq"] = pd.Categorical(cov["freq"], categories=order, ordered=True)
    cov = cov.sort_values("freq")
    fig, ax1 = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(order))
    ax1.plot(x, df["cell_balanced_acc"], "o-", color=NEUTRO, lw=2, ms=5, label="Exactitud balanceada por celda")
    ax1.set_ylabel("Exactitud balanceada por celda", color=NEUTRO)
    ax1.tick_params(axis="y", labelcolor=NEUTRO)
    ax1.set_xticks(x); ax1.set_xticklabels(order)
    ax1.set_ylim(0.55, 0.90)
    ax2 = ax1.twinx()
    ax2.plot(x, cov["coverage"], "s--", color=MEMORY, lw=2, ms=5, label="Cobertura (confianza ≥ 0.85)")
    ax2.set_ylabel("Cobertura de la decisión automática", color=MEMORY)
    ax2.tick_params(axis="y", labelcolor=MEMORY)
    ax2.set_ylim(0.60, 0.90)
    ax2.grid(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8.5, frameon=False)
    ax1.set_xlabel("Nivel de frecuencia (REF nativo, F0 el más alto fijo, F8 el más bajo)")
    fig.tight_layout()
    fig.savefig(F / "fig_cpu_exactitud_cobertura_frecuencia_20260922.png", bbox_inches="tight")
    plt.close(fig)


def roofline_scatter() -> None:
    calib = json.load(open(D / "roofline_scatter" / "calibracion_ref_f8.json"))
    samp = pd.read_csv(D / "roofline_scatter" / "ref_f8_sample.csv")
    colores = {"compute_bound": COMPUTE, "memory_bound": MEMORY}
    etiquetas = {"compute_bound": "compute-bound", "memory_bound": "memory-bound"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, nivel, titulo in ((axes[0], "REF", "REF (nativo)"), (axes[1], "F8", "F8 (0.8 GHz)")):
        c = calib[nivel]
        sub = samp[samp["freq_level_id"] == nivel]
        for cls in ("compute_bound", "memory_bound"):
            g = sub[sub["phase_label_train"] == cls]
            ax.scatter(g["operational_intensity_uncore_real"].clip(lower=1e-3), g["performance_flops_per_s"],
                       s=3, alpha=0.12, color=colores[cls], linewidths=0, label=etiquetas[cls], rasterized=True)
        x = np.logspace(-2.5, 2.2, 200)
        techo_bw = c["bw_pico_bytes_per_s"] * x
        techo_comp = np.full_like(x, c["p_pico_flops_per_s"])
        techo = np.minimum(techo_bw, techo_comp)
        ax.plot(x, techo, color="#222", lw=1.6, zorder=5)
        ax.axvline(c["i_ridge_flops_per_byte"], color="#222", lw=0.9, ls=":", zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(3e-3, 1.5e2)
        ax.set_title(f"{titulo}  ·  ridge = {c['i_ridge_flops_per_byte']:.2f} FLOP/byte", fontsize=9.5)
        ax.set_xlabel("Intensidad operacional (FLOP/byte)")
    axes[0].set_ylabel("Rendimiento (FLOP/s)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COMPUTE, markersize=7, label="compute-bound"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MEMORY, markersize=7, label="memory-bound"),
               plt.Line2D([0], [0], color="#222", lw=1.6, label="techo Roofline calibrado"),
               plt.Line2D([0], [0], color="#222", lw=0.9, ls=":", label="ridge")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(F / "fig_cpu_roofline_scatter_20260922.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    perfiles_gemelos()
    forest_politica()
    frecuencia_doble_eje()
    roofline_scatter()
    print("figuras escritas en", F)
