"""Genera figuras del conjunto CPU pacca_cpu_final_20260913.

Los conteos provienen de los 1290 archivos training_cpu_intervals.csv
reprocesados en pacca el 2026-09-17. Solo incluyen intervalos con estado de
entrenamiento ``ok`` y frecuencia ``valid`` o ``not_applicable_native``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


KERNEL_COUNTS = {
    "cpu_cholmod": (55323, 110991), "cpu_gap_pr": (0, 23542),
    "cpu_hpcg": (15, 32230), "cpu_lulesh": (5, 302135),
    "cpu_rajaperf_basic_daxpy": (95, 1882),
    "cpu_rajaperf_basic_init3": (23, 3992),
    "cpu_rajaperf_lcals_first_sum": (61, 10792),
    "cpu_rajaperf_lcals_tridiag_elim": (48, 1954),
    "cpu_rajaperf_polybench_fdtd_2d": (12, 495),
    "cpu_rajaperf_polybench_jacobi_1d": (109, 8095),
    "cpu_rajaperf_stream_add": (43, 4225),
    "cpu_rajaperf_stream_mul": (67, 9594),
    "cpu_rajaperf_stream_triad": (65, 4230), "dgemm_n2048": (10938, 0),
    "dual_axpy_cpu_N100000": (5219, 21),
    "dual_axpy_cpu_N10000000": (0, 3859),
    "dual_axpy_cpu_N3162278": (0, 3674),
    "dual_cholesky_cpu_N1024": (1875, 3098),
    "dual_cholesky_cpu_N2048": (1103, 1976),
    "dual_cholesky_cpu_N256": (6507, 0), "dual_fft_cpu_N1024": (0, 3836),
    "dual_fft_cpu_N128": (4978, 0), "dual_fft_cpu_N4096": (0, 5115),
    "dual_fft_cpu_N472": (2314, 806), "dual_gemm_cpu_N2048": (5981, 9),
    "dual_spmv_cpu_N100000": (4085, 0), "dual_spmv_cpu_N1000000": (0, 3060),
    "dual_stencil_cpu_N1024": (0, 3967), "dual_stencil_cpu_N1536": (0, 3839),
    "dual_stencil_cpu_N256": (5311, 2), "dual_stencil_cpu_N512": (4629, 197),
    "hpccg_cpu_N46": (8767, 1899), "npb_bt": (68397, 40208),
    "npb_cg": (0, 25770), "npb_ft": (582, 15097), "npb_lu": (21194, 49563),
    "npb_mg": (0, 1778), "npb_sp": (346, 55927),
    "phasic_p010": (45331, 51), "phasic_p100": (25248, 20205),
    "phasic_p1000": (22849, 22531),
    "rajaperf_polybench_3mm_omp": (53277, 2732),
    "rodinia_lavamd_omp": (31644, 15),
}

LEVEL_COUNTS = {
    "REF": (19695, 67008), "F0": (19483, 58541), "F1": (20456, 61387),
    "F2": (23100, 63718), "F3": (26744, 66256), "F4": (35034, 67203),
    "F5": (40338, 75435), "F6": (49515, 86401), "F7": (63978, 100097),
    "F8": (88098, 137346),
}

OUT = Path(__file__).resolve().parents[1] / "figuras"
DATA = Path(__file__).resolve().parents[1] / "datos" / "cpu_modelo_20260918"
OOF = Path(__file__).resolve().parents[3] / "tmp" / "cpu_modelo_documentacion_20260918" / "cpu_selective_oof_predictions.csv"
# Paleta distinguible en proyección y apta para lectores con daltonismo.
# El azul identifica compute-bound y el naranja identifica memory-bound en
# todas las figuras de composición de fase.
COMPUTE = "#0072B2"
MEMORY = "#D55E00"
MID = "#E69F00"
LIGHT = "#D1D5DB"
CORRECT = "#009E73"
ERROR = "#CC79A7"
REVIEW = "#9CA3AF"


def save_kernel_composition() -> None:
    rows = sorted(
        ((name, comp / (comp + mem) * 100, comp + mem)
         for name, (comp, mem) in KERNEL_COUNTS.items()),
        key=lambda row: row[1],
    )
    # La última barra no es un kernel. Resume la composición de la muestra
    # empleada por el clasificador y conserva sus conteos absolutos.
    train_compute, train_memory = 15286, 21024
    train_pct = train_compute / (train_compute + train_memory) * 100
    # En la página rotada, la primera fila queda al extremo final de lectura.
    rows.insert(0, ("muestra_entrenamiento", train_pct, train_compute + train_memory))

    names = [row[0].replace("_cpu_", "_") for row in rows]
    compute = np.array([row[1] for row in rows])
    memory = 100 - compute
    fig, ax = plt.subplots(figsize=(11.2, 13.2))
    y = np.arange(len(rows))
    # memory-bound parte del origen y compute-bound se muestra a continuación.
    ax.barh(y, memory, color=MEMORY, label="memory-bound")
    ax.barh(y, compute, left=memory, color=COMPUTE, label="compute-bound")
    ax.axhline(0.5, color="#374151", linewidth=0.8)
    ax.text(memory[0] / 2, y[0], f"memory\n{train_memory:,}", ha="center",
            va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(memory[0] + compute[0] / 2, y[0], f"compute\n{train_compute:,}",
            ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.set_yticks(y, names, fontsize=10.5, fontfamily="monospace")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje de intervalos elegibles")
    ax.set_ylabel("Kernel")
    ax.set_xticks(np.arange(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncols=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cpu_composicion_clase_por_kernel_20260917.png", dpi=300)
    plt.close(fig)


def save_level_composition() -> None:
    levels = list(LEVEL_COUNTS)
    compute = np.array([LEVEL_COUNTS[level][0] for level in levels])
    totals = np.array([sum(LEVEL_COUNTS[level]) for level in levels])
    compute_pct = compute / totals * 100
    memory_pct = 100 - compute_pct
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(levels))
    ax.bar(x, compute_pct, color=COMPUTE, label="compute-bound")
    ax.bar(x, memory_pct, bottom=compute_pct, color=MEMORY, label="memory-bound")
    ax.set_xticks(x, levels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje de intervalos elegibles")
    ax.set_xlabel("Nivel de frecuencia")
    ax.set_yticks(np.arange(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
    ax.set_axisbelow(True)
    for xpos, value in zip(x, compute_pct):
        ax.text(xpos, value / 2, f"{value:.1f}%", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.legend(loc="lower center", ncols=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cpu_composicion_clase_por_frecuencia_20260917.png", dpi=300)
    plt.close(fig)


def save_sampling_audit() -> None:
    derived = DATA / "cpu_sample_family_counts.csv"
    if OOF.exists():
        source = pd.read_csv(DATA / "family_class_frequency_summary.csv")
        source = source.groupby("family", as_index=False)[["n_compute_bound", "n_memory_bound"]].sum()
        source["elegibles"] = source.n_compute_bound + source.n_memory_bound
        sampled = pd.read_csv(OOF).groupby("kernel_family").size().rename("muestra")
        rows = source.set_index("family").join(sampled).sort_values("elegibles")
        rows.reset_index().to_csv(derived, index=False)
    else:
        rows = pd.read_csv(derived).set_index("family").sort_values("elegibles")
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    y = np.arange(len(rows))
    ax.barh(y - 0.18, rows.elegibles, height=0.34, color=LIGHT, label="Intervalos elegibles")
    ax.barh(y + 0.18, rows.muestra, height=0.34, color=COMPUTE, label="Intervalos seleccionados")
    ax.set_yticks(y, rows.index, fontsize=8.5, fontfamily="monospace")
    ax.set_xscale("log")
    ax.set_xlabel("Número de intervalos (escala logarítmica)")
    ax.set_ylabel("Familia algorítmica")
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cpu_muestreo_familia_20260918.png", dpi=300)
    plt.close(fig)


def save_input_correlation() -> None:
    features = [
        "ipc", "mpki", "cache_miss_rate", "stall_mem_ratio", "ips",
        "freq_khz_observed", "cache_references_per_ki",
        "cache_references_per_cycle", "cache_misses_per_cycle",
        "stalls_mem_per_ki", "stalls_mem_per_cache_miss", "log1p_mpki",
    ]
    labels = [
        "IPC", "MPKI", "Tasa fallos", "Stall mem.", "IPS", "Frecuencia",
        "Ref./kI", "Ref./ciclo", "Fallos/ciclo", "Stall/kI",
        "Stall/fallo", "log(1+MPKI)",
    ]
    derived = DATA / "cpu_final_feature_correlation.csv"
    if OOF.exists():
        corr = pd.read_csv(OOF, usecols=features).corr(method="pearson")
        corr.to_csv(derived)
    else:
        corr = pd.read_csv(derived, index_col=0).loc[features, features]
    fig, ax = plt.subplots(figsize=(8.4, 7.1))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.65 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación de Pearson")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cpu_correlacion_entradas_20260918.png", dpi=300)
    plt.close(fig)



def main() -> None:
    save_kernel_composition()
    save_level_composition()
    save_sampling_audit()
    save_input_correlation()
    print("figuras del conjunto CPU escritas en", OUT)


if __name__ == "__main__":
    main()
