#!/usr/bin/env python3
"""Genera figuras del Sweep B (gpu_interval_ns) a partir de
docs/justifications/data/gpu_interval_ns/summary.csv (traido de pacca)."""
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gpu_interval_ns"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "gpu_interval_ns"


def _load_rows():
    # rodinia_heartwall_fix.csv reemplaza las filas originales de este
    # kernel: la corrida original midio un video sintetico truncado a 25
    # cuadros (ARC-89, seccion 3 del reporte) que fallaba de inmediato,
    # no el comportamiento real del kernel.
    with open(DATA_DIR / "summary.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["kernel_ref"] != "rodinia_heartwall"]
    fix_path = DATA_DIR / "rodinia_heartwall_fix.csv"
    if fix_path.exists():
        with open(fix_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_rows()

    by_kernel = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["success"] != "True":
            continue
        gpu_interval_ns = int(r["gpu_interval_ns"])
        kernel = r["kernel_ref"]
        n = r.get("n_gpu_telemetry")
        if n not in (None, ""):
            by_kernel[kernel][gpu_interval_ns].append(int(n))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, kernel in enumerate(sorted(by_kernel)):
        xs = sorted(by_kernel[kernel])
        ys = [statistics.fmean(by_kernel[kernel][x]) for x in xs]
        ax.plot([x / 1e6 for x in xs], ys, marker="o", color=color_for(i), label=kernel)
    ax.axhline(5, color="#52514e", linestyle="--", linewidth=1, label="target_windows_per_repetition=5")
    ax.set_xlabel("gpu_interval_ns (ms)")
    ax.set_ylabel("Muestras 'gpu_telemetry' logradas (media por repetición)")
    ax.set_yscale("log")
    ax.set_title("Muestras NVML útiles logradas vs. cadencia de muestreo GPU")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gpu_telemetry_vs_interval.png")
    plt.close(fig)

    table_path = PLOTS_DIR / "table_summary.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_interval_ns_ms", "kernel_ref", "n_gpu_telemetry_mean"])
        for kernel in sorted(by_kernel):
            for gpu_interval_ns in sorted(by_kernel[kernel]):
                writer.writerow([gpu_interval_ns / 1e6, kernel,
                                  round(statistics.fmean(by_kernel[kernel][gpu_interval_ns]), 1)])
    print(f"Escrito: {PLOTS_DIR}/gpu_telemetry_vs_interval.png, {table_path}")


if __name__ == "__main__":
    main()
