"""Importancia por permutación de los modelos desplegables de CPU y GPU.

La importancia es la caída media de exactitud balanceada en la familia retenida
cuando se permuta una entrada. Conserva el protocolo LOFO y no interpreta el
signo de la predicción ni relaciones causales.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fase2_clasificador.analysis import gpu_quality_report as gq


ROOT = Path(__file__).resolve().parents[2].parent
BOOK = Path(__file__).resolve().parents[1]
FIGURES = BOOK / "figuras"
CPU_DATA = BOOK / "datos" / "cpu_calidad_30fam" / "permutation_importance.csv"
GPU_SOURCE = BOOK / "datos" / "gpu_calidad_20260922" / "historical_gpu_relaxed020_20260922.csv"
GPU_OUT = BOOK / "datos" / "gpu_calidad_20260924" / "rf_operativo" / "permutation_importance.csv"

BLUE = "#2b6cb0"
GRAY = "#a0aec0"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})

CPU_NAMES = {
    "mpki": "MPKI",
    "cache_miss_rate": "Tasa de fallos de caché",
    "freq_khz_observed": "Frecuencia observada",
    "ips": "Instrucciones por segundo",
    "ipc": "IPC",
    "stall_mem_ratio": "Ciclos detenidos por memoria",
}
GPU_NAMES = {
    "gpu_util_pct_median": "Utilización GPU",
    "gpu_mem_util_pct_median": "Utilización de memoria",
    "gpu_mem_util_pct_std": "Variabilidad de utilización de memoria",
}
GPU_FEATURES = list(GPU_NAMES)


def score_by_class(y: np.ndarray, pred: np.ndarray) -> float:
    recalls = [np.mean(pred[y == label] == label) for label in (False, True) if np.any(y == label)]
    return float(np.mean(recalls))


def gpu_importance() -> pd.DataFrame:
    gq.FEATURES[:] = GPU_FEATURES
    frame = gq.load(str(GPU_SOURCE), split_rajaperf_cuda=True)
    families = sorted(frame[gq.FAMILY_COL].unique())
    fam_codes = pd.Categorical(frame[gq.FAMILY_COL], categories=families).codes
    family = frame[gq.FAMILY_COL].to_numpy()
    y = frame["y"].to_numpy()
    x = frame[GPU_FEATURES].to_numpy(dtype=np.float32)
    drops = []

    for i, held_out in enumerate(families):
        train = np.flatnonzero(family != held_out)
        test = np.flatnonzero(family == held_out)
        model = gq.make_model("random_forest", 2000)
        model.fit(x[train], y[train], sample_weight=gq.cell_weights(fam_codes[train], y[train]))
        base = score_by_class(y[test], model.predict(x[test]))
        row = {"family": held_out, "base_score": base}
        for j, name in enumerate(GPU_FEATURES):
            rng = np.random.default_rng(20260925 + 100 * i + j)
            permuted = x[test].copy()
            permuted[:, j] = rng.permutation(permuted[:, j])
            row[name] = base - score_by_class(y[test], model.predict(permuted))
        drops.append(row)

    by_family = pd.DataFrame(drops).set_index("family")
    summary = pd.DataFrame({
        "mean_drop": by_family[GPU_FEATURES].mean(),
        "sd_between_families": by_family[GPU_FEATURES].std(),
        "share_families_positive": (by_family[GPU_FEATURES] > 0).mean(),
    }).sort_values("mean_drop", ascending=False)
    GPU_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(GPU_OUT)
    return summary


def plot_importance(data: pd.DataFrame, names: dict[str, str], title: str, output: Path) -> None:
    data = data.sort_values("mean_drop")
    y = np.arange(len(data))
    colors = [BLUE if value > 0 else GRAY for value in data["mean_drop"]]
    fig, ax = plt.subplots(figsize=(6.7, 2.6 if len(data) <= 3 else 3.35))
    ax.barh(y, data["mean_drop"], color=colors)
    ax.axvline(0, color="#4a5568", linewidth=0.8)
    ax.set_yticks(y, [names[name] for name in data.index])
    ax.set_xlabel("Caída de exactitud balanceada al permutar la entrada")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", visible=False)
    for value, yy in zip(data["mean_drop"], y):
        if value >= 0:
            ax.annotate(f"{value:.3f}", (value, yy), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=8.5)
        else:
            ax.annotate(f"{value:.3f}", (value, yy), xytext=(0, 9),
                        textcoords="offset points", va="bottom", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    cpu = pd.read_csv(CPU_DATA, index_col=0)
    gpu = gpu_importance()
    plot_importance(cpu, CPU_NAMES, "CPU, XGBoost de seis entradas", FIGURES / "fig_cpu_importancia_permutacion_20260925.png")
    plot_importance(gpu, GPU_NAMES, "GPU, random forest de tres entradas invariantes", FIGURES / "fig_gpu_importancia_permutacion_20260925.png")
    print("Importancia GPU")
    print(gpu.round(4).to_string())


if __name__ == "__main__":
    main()
