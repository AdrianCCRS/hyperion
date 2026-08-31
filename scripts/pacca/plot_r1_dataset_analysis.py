#!/usr/bin/env python3
"""Genera las figuras de la Fase R1 (reanalisis por tamano y amortizacion)
para el capitulo de Resultados del libro, a partir de los CSV reales del
dataset final en pacca (candidate_summary.csv / strategy_c_candidates.csv).

Uso:
    python3 plot_r1_dataset_analysis.py [--cand CAND.csv] [--stratc STRATC.csv] [--out DIR]

Por defecto lee las copias locales limpias en el scratchpad de la sesion y
escribe en docs/libro/figuras/.
"""
import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------------------------------------------
# Estilo de publicacion: sin colores por defecto de matplotlib, fuente legible
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

# Paleta cualitativa (Okabe-Ito, apta para daltonismo), una entrada por operacion
OP_COLOR = {
    "gemm":    "#0072B2",
    "cholesky":"#D55E00",
    "fft":     "#009E73",
    "axpy":    "#CC79A7",
    "spmv":    "#E69F00",
    "stencil": "#56B4E9",
}
OP_LABEL = {
    "gemm": "GEMM",
    "cholesky": "Cholesky",
    "fft": "FFT",
    "axpy": "AXPY",
    "spmv": "SpMV",
    "stencil": "Stencil",
}
OP_ORDER = ["gemm", "cholesky", "fft", "axpy", "spmv", "stencil"]
COMPUTE_BOUND = {"gemm", "cholesky", "fft"}
MEMORY_BOUND = {"axpy", "spmv", "stencil"}

DEVICE_COLOR = {"cpu": "#0072B2", "gpu": "#D55E00"}
REGION_COLOR = {"cold": "#D55E00", "warm": "#0072B2"}


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def to_f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  escrito: {path}")


# ----------------------------------------------------------------------------
# 1. Tiempo cold vs warm por operacion (REF, CPU y GPU)
# ----------------------------------------------------------------------------
def fig_cold_warm_time(rows, outdir):
    # Un tamano representativo por operacion: el mayor disponible.
    by_op_max_size = {}
    for r in rows:
        op = r["operation"]
        sz = to_f(r["size"])
        if op not in by_op_max_size or sz > by_op_max_size[op]:
            by_op_max_size[op] = sz

    data = defaultdict(dict)  # (op, device) -> {cold: (mean,std), warm: (mean,std)}
    for r in rows:
        op = r["operation"]
        if to_f(r["size"]) != by_op_max_size.get(op):
            continue
        dev = r["device"]
        action = r["action_id"]
        if dev == "cpu" and action != "cpu:REF":
            continue
        if dev == "gpu" and action != "gpu:REF:REF":
            continue
        region = r["region"]
        data[(op, dev)][region] = (to_f(r["time_mean"]), to_f(r["time_std"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, dev, title in zip(axes, ["cpu", "gpu"], ["CPU (cpu:REF)", "GPU (gpu:REF:REF)"]):
        ops = [op for op in OP_ORDER if (op, dev) in data]
        x = range(len(ops))
        width = 0.35
        cold_vals = [data[(op, dev)].get("cold", (math.nan, 0))[0] for op in ops]
        cold_err = [data[(op, dev)].get("cold", (math.nan, 0))[1] for op in ops]
        warm_vals = [data[(op, dev)].get("warm", (math.nan, 0))[0] for op in ops]
        warm_err = [data[(op, dev)].get("warm", (math.nan, 0))[1] for op in ops]
        ax.bar([i - width / 2 for i in x], cold_vals, width, yerr=cold_err,
               label="Region fria", color="#D55E00", capsize=3)
        ax.bar([i + width / 2 for i in x], warm_vals, width, yerr=warm_err,
               label="Region caliente", color="#0072B2", capsize=3)
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([OP_LABEL[op] for op in ops], rotation=20)
        ax.set_ylabel("Tiempo por despacho (s, log)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Tiempo por despacho, region fria vs. caliente (tamano mayor por operacion)")
    fig.tight_layout()
    savefig(fig, outdir, "fig_r1_cold_warm_time.pdf")


# ----------------------------------------------------------------------------
# 2. Cuando GPU supera a CPU: log10(EDP_gpu/EDP_cpu) vs log10(tamano), region warm
# ----------------------------------------------------------------------------
def fig_crossover(rows, outdir):
    cpu_edp = {}
    gpu_edp = {}
    for r in rows:
        if r["region"] != "warm":
            continue
        op, sz = r["operation"], to_f(r["size"])
        if r["device"] == "cpu" and r["action_id"] == "cpu:REF":
            cpu_edp[(op, sz)] = to_f(r["edp_mean"])
        elif r["device"] == "gpu" and r["action_id"] == "gpu:REF:REF":
            gpu_edp[(op, sz)] = to_f(r["edp_mean"])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for op in OP_ORDER:
        xs, ys = [], []
        for (o, sz), c in cpu_edp.items():
            if o != op:
                continue
            g = gpu_edp.get((o, sz))
            if g is None or c is None or math.isnan(c) or math.isnan(g) or c <= 0 or g <= 0:
                continue
            xs.append(math.log10(sz))
            ys.append(math.log10(g / c))
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        marker = "o" if op in COMPUTE_BOUND else "s"
        ax.plot(xs, ys, marker=marker, color=OP_COLOR[op], label=OP_LABEL[op],
                linewidth=1.4, markersize=5)

    ax.axhline(0, color="#444444", linewidth=1.0, linestyle="--")
    ax.text(ax.get_xlim()[1], 0.05, "GPU y CPU en empate", ha="right", va="bottom",
            fontsize=8.5, color="#444444")
    ax.set_xlabel(r"$\log_{10}(N)$ (tamano de problema)")
    ax.set_ylabel(r"$\log_{10}(EDP_{GPU}/EDP_{CPU})$ en region caliente")
    ax.set_title("Cuando la GPU supera a la CPU (region caliente, REF)")
    ax.annotate("GPU gana\n(debajo de la linea)", xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8.5, color="#333333")
    ax.annotate("CPU gana\n(encima de la linea)", xy=(0.02, 0.90), xycoords="axes fraction",
                fontsize=8.5, color="#333333")
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    savefig(fig, outdir, "fig_r1_crossover_edp.pdf")


# ----------------------------------------------------------------------------
# 2b. Ancho de banda implicito (bytes logicos / tiempo), region warm GPU REF:REF
# ----------------------------------------------------------------------------
def fig_bandwidth(rows_stratc, outdir):
    pts = defaultdict(list)  # op -> list of (size, GB/s)
    for r in rows_stratc:
        if r["region"] != "warm" or r["device"] != "gpu" or r["action_id"] != "gpu:REF:REF":
            continue
        op = r["operation"]
        sz = to_f(r["size"])
        t = to_f(r["time_mean"])
        b = to_f(r["logical_bytes_per_dispatch"])
        if not b or math.isnan(b) or not t or math.isnan(t) or t <= 0:
            continue
        gbs = (b / t) / 1e9
        pts[op].append((sz, gbs))

    fig, ax = plt.subplots(figsize=(8, 5.0))
    for op in OP_ORDER:
        if op not in pts:
            continue
        data = sorted(pts[op])
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        marker = "o" if op in COMPUTE_BOUND else "s"
        ax.plot(xs, ys, marker=marker, color=OP_COLOR[op], label=OP_LABEL[op],
                linewidth=1.4, markersize=5)
    ax.set_xscale("log")
    ax.axhspan(8, 10, color="#999999", alpha=0.15, label="Banda 8-10 GB/s")
    ax.set_xlabel("Tamano de problema $N$ (log)")
    ax.set_ylabel("Bytes logicos / tiempo (GB/s)")
    ax.set_title("Rendimiento implicito de movimiento de datos, GPU en caliente (REF)")
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    savefig(fig, outdir, "fig_r1_bandwidth_convergencia.pdf")
    return pts


# ----------------------------------------------------------------------------
# 3 y 4. Tamano vs tiempo, tamano vs energia (log-log), warm, CPU y GPU REF
# ----------------------------------------------------------------------------
def fig_size_vs_metric(rows, outdir, metric, ylabel, fname, title):
    data = defaultdict(lambda: defaultdict(list))  # (op,dev) -> size -> [val,...] (1 val, mean already)
    for r in rows:
        if r["region"] != "warm":
            continue
        dev = r["device"]
        if dev == "cpu" and r["action_id"] != "cpu:REF":
            continue
        if dev == "gpu" and r["action_id"] != "gpu:REF:REF":
            continue
        op = r["operation"]
        sz = to_f(r["size"])
        v = to_f(r[metric])
        if math.isnan(v):
            continue
        data[(op, dev)][sz].append(v)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, dev, dtitle in zip(axes, ["cpu", "gpu"], ["CPU (cpu:REF)", "GPU (gpu:REF:REF)"]):
        for op in OP_ORDER:
            key = (op, dev)
            if key not in data:
                continue
            sizes = sorted(data[key])
            ys = [statistics.mean(data[key][s]) for s in sizes]
            marker = "o" if op in COMPUTE_BOUND else "s"
            ax.plot(sizes, ys, marker=marker, color=OP_COLOR[op], label=OP_LABEL[op],
                    linewidth=1.3, markersize=4.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tamano de problema $N$ (log)")
        ax.set_title(dtitle)
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, outdir, fname)


# ----------------------------------------------------------------------------
# 5. CV entre repeticiones: histograma/boxplot por region y dispositivo
# ----------------------------------------------------------------------------
def fig_cv(rows, outdir):
    by_region = defaultdict(list)
    by_device = defaultdict(list)
    for r in rows:
        cv = to_f(r["edp_cv_pct"])
        if math.isnan(cv):
            continue
        by_region[r["region"]].append(cv)
        by_device[r["device"]].append(cv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    labels = ["cold", "warm"]
    box_data = [by_region[l] for l in labels]
    bp = ax.boxplot(box_data, tick_labels=["Region fria", "Region caliente"], showfliers=False,
                     patch_artist=True, widths=0.5)
    for patch, l in zip(bp["boxes"], labels):
        patch.set_facecolor(REGION_COLOR[l])
        patch.set_alpha(0.55)
    for l, xpos in zip(labels, [1, 2]):
        med = statistics.median(by_region[l])
        ax.annotate(f"mediana={med:.2f}%", xy=(xpos, med), xytext=(xpos + 0.12, med),
                    fontsize=8.5)
    ax.set_ylabel("CV del EDP entre repeticiones (%)")
    ax.set_title("Por region temporal")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    ax = axes[1]
    labels = ["cpu", "gpu"]
    box_data = [by_device[l] for l in labels]
    bp = ax.boxplot(box_data, tick_labels=["CPU", "GPU"], showfliers=False,
                     patch_artist=True, widths=0.5)
    for patch, l in zip(bp["boxes"], labels):
        patch.set_facecolor(DEVICE_COLOR[l])
        patch.set_alpha(0.55)
    for l, xpos in zip(labels, [1, 2]):
        med = statistics.median(by_device[l])
        ax.annotate(f"mediana={med:.2f}%", xy=(xpos, med), xytext=(xpos + 0.12, med),
                    fontsize=8.5)
    ax.set_ylabel("CV del EDP entre repeticiones (%)")
    ax.set_title("Por dispositivo")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    fig.suptitle("Piso de ruido de medicion: dispersion del EDP entre repeticiones")
    fig.tight_layout()
    savefig(fig, outdir, "fig_r1_cv_ruido.pdf")

    summary = {
        "global": statistics.median([v for vs in by_region.values() for v in vs]),
        "cpu": statistics.median(by_device["cpu"]),
        "gpu": statistics.median(by_device["gpu"]),
        "cold": statistics.median(by_region["cold"]),
        "warm": statistics.median(by_region["warm"]),
    }
    return summary


# ----------------------------------------------------------------------------
# 6/7/8. Energia, tiempo y EDP vs frecuencia para operaciones representativas
# ----------------------------------------------------------------------------
CPU_FREQ_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "REF"]
GPU_FREQ_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "REF"]


def fig_vs_frequency(rows, outdir, metric, ylabel, fname, title, log=False):
    # Operaciones representativas: gemm y cholesky (compute-bound), axpy y stencil (memory-bound)
    reps = [("gemm", "cpu"), ("cholesky", "cpu"), ("axpy", "cpu"), ("stencil", "cpu"),
            ("gemm", "gpu"), ("cholesky", "gpu"), ("axpy", "gpu"), ("stencil", "gpu")]

    # Tamano representativo = mayor por operacion
    by_op_max_size = {}
    for r in rows:
        op = r["operation"]
        sz = to_f(r["size"])
        if op not in by_op_max_size or sz > by_op_max_size[op]:
            by_op_max_size[op] = sz

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=False)
    ax_cpu, ax_gpu = axes

    for op in ["gemm", "cholesky", "axpy", "stencil"]:
        target_size = by_op_max_size[op]
        # CPU: variar cpu_level con action cpu:F*, region warm
        vals = {}
        for r in rows:
            if r["operation"] != op or r["device"] != "cpu" or r["region"] != "warm":
                continue
            if to_f(r["size"]) != target_size:
                continue
            lvl = r["cpu_level"]
            v = to_f(r[metric])
            if not math.isnan(v):
                vals[lvl] = v
        xs = [l for l in CPU_FREQ_ORDER if l in vals]
        ys = [vals[l] for l in xs]
        marker = "o" if op in COMPUTE_BOUND else "s"
        ax_cpu.plot(range(len(xs)), ys, marker=marker, color=OP_COLOR[op],
                    label=f"{OP_LABEL[op]} (N={int(target_size)})", linewidth=1.4, markersize=5)

        # GPU: variar gpu_level con cpu_level=REF (gpu:REF:F*), region warm
        vals = {}
        for r in rows:
            if r["operation"] != op or r["device"] != "gpu" or r["region"] != "warm":
                continue
            if to_f(r["size"]) != target_size:
                continue
            if r["cpu_level"] != "REF":
                continue
            lvl = r["gpu_level"]
            v = to_f(r[metric])
            if not math.isnan(v):
                vals[lvl] = v
        xs2 = [l for l in GPU_FREQ_ORDER if l in vals]
        ys2 = [vals[l] for l in xs2]
        ax_gpu.plot(range(len(xs2)), ys2, marker=marker, color=OP_COLOR[op],
                    label=f"{OP_LABEL[op]} (N={int(target_size)})", linewidth=1.4, markersize=5)

    for ax, xs_ref, dtitle in [(ax_cpu, CPU_FREQ_ORDER, "CPU (cpu:F0..F6,REF)"),
                                (ax_gpu, GPU_FREQ_ORDER, "GPU (gpu:REF:F0..F6,REF), CPU fija en REF")]:
        ax.set_xticks(range(len(xs_ref)))
        ax.set_xticklabels(xs_ref)
        ax.set_title(dtitle)
        if log:
            ax.set_yscale("log")
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)
    ax_cpu.set_ylabel(ylabel)
    ax_gpu.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title)
    fig.supxlabel("Nivel de frecuencia (F0 = maxima ... F6 = minima; REF = gobernador nativo)",
                   fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, outdir, fname)


def main():
    ap = argparse.ArgumentParser()
    scratch = "/tmp/claude-1000/-home-adrianccrs-Documents-Dev-TG-hyperion/d89c765e-3d2c-4058-8655-9fe31e5aa5cc/scratchpad"
    ap.add_argument("--cand", default=os.path.join(scratch, "cand.csv"))
    ap.add_argument("--stratc", default=os.path.join(scratch, "strategy_c_candidates_clean.csv"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "libro", "figuras"))
    args = ap.parse_args()

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    print(f"Leyendo {args.cand} ...")
    rows = read_csv(args.cand)
    print(f"  {len(rows)} filas")
    print(f"Leyendo {args.stratc} ...")
    rows_stratc = read_csv(args.stratc)
    print(f"  {len(rows_stratc)} filas")

    print("Figura 1: tiempo cold vs warm")
    fig_cold_warm_time(rows, outdir)

    print("Figura 2: cruce EDP GPU/CPU vs tamano")
    fig_crossover(rows, outdir)

    print("Figura 2b: ancho de banda implicito")
    bw = fig_bandwidth(rows_stratc, outdir)

    print("Figura 3: tamano vs tiempo")
    fig_size_vs_metric(rows, outdir, "time_mean", "Tiempo por despacho (s, log)",
                        "fig_r1_size_vs_tiempo.pdf",
                        "Tamano de problema vs. tiempo (region caliente, REF)")

    print("Figura 4: tamano vs energia")
    fig_size_vs_metric(rows, outdir, "energy_mean", "Energia por despacho (J, log)",
                        "fig_r1_size_vs_energia.pdf",
                        "Tamano de problema vs. energia (region caliente, REF)")

    print("Figura 5: CV entre repeticiones")
    cv_summary = fig_cv(rows, outdir)

    print("Figura 6: energia vs frecuencia")
    fig_vs_frequency(rows, outdir, "energy_mean", "Energia por despacho (J)",
                      "fig_r1_energia_vs_frecuencia.pdf",
                      "Energia por despacho vs. nivel de frecuencia (region caliente)")

    print("Figura 7: tiempo vs frecuencia")
    fig_vs_frequency(rows, outdir, "time_mean", "Tiempo por despacho (s)",
                      "fig_r1_tiempo_vs_frecuencia.pdf",
                      "Tiempo por despacho vs. nivel de frecuencia (region caliente)")

    print("Figura 8: EDP vs frecuencia")
    fig_vs_frequency(rows, outdir, "edp_mean", "EDP por despacho (J*s)",
                      "fig_r1_edp_vs_frecuencia.pdf",
                      "EDP por despacho vs. nivel de frecuencia (region caliente)", log=True)

    print("\n--- Resumen numerico (para verificar contra el texto del libro) ---")
    print("CV mediano EDP: ", {k: round(v, 3) for k, v in cv_summary.items()})
    for op, pts in bw.items():
        vals = [v for _, v in pts]
        if vals:
            print(f"Ancho de banda implicito {op}: min={min(vals):.2f} GB/s max={max(vals):.2f} GB/s")


if __name__ == "__main__":
    main()
