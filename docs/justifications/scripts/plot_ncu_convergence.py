#!/usr/bin/env python3
"""Genera figuras del Sweep F (convergencia de FLOP/byte vs. numero de
lanzamientos perfilados con ncu) a partir de
docs/justifications/data/ncu_launch_count/summary.csv (traido de pacca)."""
import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "ncu_launch_count" / "summary.csv"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "ncu_launch_count"


def main():
    apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    by_kernel = defaultdict(list)
    declared = {}
    for r in rows:
        if r.get("operational_intensity") in (None, ""):
            continue
        kernel = r["kernel_ref"]
        by_kernel[kernel].append((int(r["requested_launch_count"]), float(r["operational_intensity"])))
        declared[kernel] = float(r["catalog_declared_oi"]) if r.get("catalog_declared_oi") not in (None, "") else None

    n_kernels = len(by_kernel)
    fig, axes = plt.subplots(1, n_kernels, figsize=(3.2 * n_kernels, 4), sharey=False)
    if n_kernels == 1:
        axes = [axes]
    for ax, (kernel, points) in zip(axes, sorted(by_kernel.items())):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", color=color_for(0))
        if declared.get(kernel) is not None:
            ax.axhline(declared[kernel], color="#52514e", linestyle="--", linewidth=1)
        ax.set_title(kernel, fontsize=8)
        ax.set_xlabel("launch-count")
    axes[0].set_ylabel("FLOP/byte estimado")
    fig.suptitle("Convergencia de FLOP/byte vs. número de lanzamientos perfilados con ncu")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ncu_convergence.png")
    plt.close(fig)

    table_path = PLOTS_DIR / "table_summary.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel_ref", "launch_count", "operational_intensity", "catalog_declared_oi"])
        for kernel in sorted(by_kernel):
            for lc, oi in sorted(by_kernel[kernel]):
                writer.writerow([kernel, lc, round(oi, 4), declared.get(kernel)])
    print(f"Escrito: {PLOTS_DIR}/ncu_convergence.png, {table_path}")


if __name__ == "__main__":
    main()
