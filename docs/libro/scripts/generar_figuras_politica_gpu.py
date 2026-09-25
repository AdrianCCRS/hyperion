"""Figuras de la política de frecuencia GPU (sección \\ref{sec:resultados-gpu-politica}).

Lee tmp/gpu_policy_20260922/ (generado por fase2_clasificador/analysis/
gpu_policy_table.py y gpu_policy_by_family.py sobre
tmp/historical_gpu_relaxed020_20260922.csv) y escribe
docs/libro/figuras/fig_gpu_politica_*_20260922.png.
Reproduce:
  python3 fase2_clasificador/analysis/gpu_policy_table.py \
      tmp/historical_gpu_relaxed020_20260922.csv --out tmp/gpu_policy_20260922
  python3 fase2_clasificador/analysis/gpu_policy_by_family.py \
      tmp/historical_gpu_relaxed020_20260922.csv --out tmp/gpu_policy_20260922
  python3 docs/libro/scripts/generar_figuras_politica_gpu.py

Genera tres figuras, espejo de las de CPU (generar_figuras_revision_20260922.py
y generar_figuras_energia_cpu.py) pero con la familia como unidad donde CPU usa
el kernel, porque esa es la unidad real de la Tabla \\ref{tab:gpu-politica-ic}:
  1. Forest plot de la ganancia de EDP por nivel y clase (familia como unidad).
  2. Tiempo, energía y EDP relativos a REF por nivel y clase (kernel como unidad,
     descriptivo; la significancia se reporta a nivel de familia en la tabla).
  3. Potencia de GPU frente al reloj SM, con el piso estático ajustado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import sys
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fase2_clasificador.analysis.gpu_policy_table import LEVELS, kernel_class, per_run  # noqa: E402

D = _ROOT / "tmp" / "gpu_policy_20260922"
DATASET = _ROOT / "tmp" / "historical_gpu_relaxed020_20260922.csv"
F = Path(__file__).resolve().parents[1] / "figuras"
COMPUTE, MEMORY, NEUTRO = "#2b6cb0", "#dd6b20", "#4a5568"
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dcdcdc",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})


def forest_politica() -> None:
    d = json.load(open(D / "policy_by_family.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, clase, titulo, color in (
        (axes[0], "compute_bound", "Compute (5 familias)", COMPUTE),
        (axes[1], "memory_bound", "Memory (8 familias)", MEMORY),
    ):
        levels = d[clase]["levels"]
        names = [lv for lv in LEVELS if lv in levels]
        gain = [levels[n]["gain"] * 100 for n in names]
        lo = [levels[n]["gain_ci95"][0] * 100 for n in names]
        hi = [levels[n]["gain_ci95"][1] * 100 for n in names]
        y = np.arange(len(names))[::-1]
        xerr = [np.array(gain) - np.array(lo), np.array(hi) - np.array(gain)]
        ax.errorbar(gain, y, xerr=xerr, fmt="o", color=color, ecolor="#b0b0b0", elinewidth=1.6, capsize=3, ms=5)
        # F1 determina la política del daemon. Etiquetar el punto evita leer el
        # extremo positivo del IC95 como si fuera la ganancia estimada.
        if "F1" in names:
            f1 = names.index("F1")
            ax.annotate(f"{gain[f1]:+.1f}%",
                        xy=(gain[f1], y[f1]), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=color, fontweight="bold")
        ax.axvline(0, color="#333", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Ganancia del EDP frente a REF (%)")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1.0)
    fig.suptitle("Ganancia estimada del producto energía--retardo GPU por nivel\n"
                 "Punto: estimación; barra gris: IC95 por bootstrap de familias", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(F / "fig_gpu_politica_forest_20260922.png", bbox_inches="tight")
    plt.close(fig)


def relativos_a_referencia() -> None:
    df = pd.read_csv(DATASET, low_memory=False)
    runs = per_run(df)
    kc = kernel_class(runs)
    med = runs.groupby(["kernel_ref", "level"])[["edp", "energy_j", "time_s"]].median().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), sharey=True)
    for ax, cls, titulo in ((axes[0], "compute_bound", "Kernels compute-bound"),
                            (axes[1], "memory_bound", "Kernels memory-bound")):
        ks = kc.index[kc["class"] == cls]
        sub = med[med.kernel_ref.isin(ks)]
        ref = sub[sub.level == "REF"].set_index("kernel_ref")
        n_con_ref = ref.index.nunique()  # kernels con REF, los unicos que aportan a estas curvas
        rows = []
        for lv in LEVELS:
            c = sub[sub.level == lv].set_index("kernel_ref")
            common = sorted(set(ref.index) & set(c.index))
            if not common:
                continue
            r, x = ref.loc[common], c.loc[common]
            rows.append({"level": lv, "time_ratio": (x.time_s / r.time_s).median(),
                        "energy_ratio": (x.energy_j / r.energy_j).median(),
                        "edp_ratio": (x.edp / r.edp).median()})
        s = pd.DataFrame(rows).set_index("level")
        x = [LEVELS.index(lv) for lv in s.index]
        ax.plot(x, s["time_ratio"], "s-", color=NEUTRO, lw=1.8, ms=4, label="Tiempo")
        ax.plot(x, s["energy_ratio"], "o-", color=COMPUTE, lw=2, ms=5, label="Energía")
        ax.plot(x, s["edp_ratio"], "^-", color=MEMORY, lw=1.8, ms=5, label="Producto energía-retardo")
        ax.axhline(1.0, color="#9a9a9a", ls="--", lw=1)
        ax.set_title(f"{titulo} ({n_con_ref})", fontsize=10)
        ax.set_xlabel("Nivel de frecuencia GPU")
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS)
    axes[0].set_ylabel("Relativo a REF (mediana entre kernels)")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(F / "fig_gpu_politica_energia_relativa_20260922.png"); plt.close(fig)


def potencia_y_piso() -> None:
    df = pd.read_csv(DATASET, low_memory=False)
    df = df[df.training_eligible & (df.kernel_ref != "minibude_cuda_bm1")]
    agg = df.groupby(["kernel_ref", "gpu_freq_level_id"]).agg(
        clock=("gpu_sm_clock_mhz_median", "median"), power=("gpu_power_mw_median", "median"),
    ).reset_index()

    def power_model(f, p0, c, k):
        return p0 + c * np.power(f, k)

    rows, excluded = [], []
    for kernel_ref, g in agg.groupby("kernel_ref"):
        g = g.dropna(subset=["clock", "power"])
        # con menos de 7 niveles de reloj (incluida REF) el ajuste de 3 parametros
        # queda subdeterminado: el rango es demasiado angosto para separar el piso
        # estatico del termino dinamico, y el ajuste diverge sin advertirlo por si solo.
        if len(g) < 7:
            excluded.append((kernel_ref, f"solo {len(g)} niveles de reloj"))
            continue
        f = g["clock"].to_numpy(dtype=float); p = g["power"].to_numpy(dtype=float)
        try:
            popt, _ = curve_fit(power_model, f, p, p0=[p.min() * 0.5, 1e-6, 2.0], maxfev=20000)
        except RuntimeError:
            excluded.append((kernel_ref, "sin convergencia"))
            continue
        p0, c, k = popt
        pred = power_model(f, *popt)
        r2 = 1 - np.sum((p - pred) ** 2) / np.sum((p - p.mean()) ** 2)
        if r2 < 0.8:
            excluded.append((kernel_ref, f"R2={r2:.2f}, ajuste no identificable"))
            continue
        static_share = p0 / power_model(f.max(), *popt)
        rows.append({"kernel_ref": kernel_ref, "p0_mw": p0, "static_share_at_fmax": static_share, "r2": r2})
    pw = pd.DataFrame(rows)
    print("excluidos del modelo de potencia:", excluded)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    pk = agg.pivot(index="kernel_ref", columns="gpu_freq_level_id", values="power")
    order = sorted([c for c in pk.columns if c != "REF"], key=lambda c: -agg[agg.gpu_freq_level_id == c].clock.median())
    pk = pk[order]
    for _, fila in pk.iterrows():
        a.plot(range(len(order)), fila.to_numpy(), color="#cbd5e0", lw=0.8, zorder=1)
    a.plot(range(len(order)), pk.median().to_numpy(), "o-", color=COMPUTE, lw=2.2, ms=5, zorder=3, label="Mediana entre kernels")
    p0_med = float(pw["p0_mw"].median()) / 1000.0
    a.axhline(p0_med * 1000, color=MEMORY, ls="--", lw=1.4, zorder=2, label=f"Piso estático ({p0_med:.0f} W)")
    a.set_xticks(range(len(order))); a.set_xticklabels(order)
    a.set_xlabel("Nivel de frecuencia GPU (orden decreciente de reloj SM)"); a.set_ylabel("Potencia GPU (mW)")
    a.legend(frameon=False, fontsize=8.5, loc="upper right")
    b.hist(pw["static_share_at_fmax"], bins=np.linspace(0.2, 0.8, 13), color=COMPUTE, edgecolor="white")
    b.axvline(float(pw["static_share_at_fmax"].median()), color=MEMORY, ls="--", lw=1.4)
    b.set_xlabel("Fracción estática al reloj máximo"); b.set_ylabel("Kernels")
    fig.tight_layout(); fig.savefig(F / "fig_gpu_politica_piso_20260922.png"); plt.close(fig)
    print("P0 mediana (W):", round(p0_med, 1), "| fraccion estatica mediana:", round(pw["static_share_at_fmax"].median(), 3),
          "| kernels:", len(pw))


if __name__ == "__main__":
    forest_politica()
    relativos_a_referencia()
    potencia_y_piso()
    print("figuras escritas en", F)
