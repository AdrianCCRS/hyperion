#!/usr/bin/env python3
"""Genera figuras del Sweep G (distribucion de calibracion repetida) a
partir de docs/justifications/data/calibration_repeats/summary.csv."""
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "calibration_repeats" / "summary.csv"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "calibration_repeats"

# Referencias declaradas en los manifiestos reales, para contrastar la
# dispersion medida contra la tolerancia D03 (40%) / cruzada GPU (5%).
DATASHEET_CPU = {"stream_official": 58353900000.0, "ert_probe": 72004000000.0}


def main():
    apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    by_kernel = defaultdict(list)
    for r in rows:
        if r["success"] != "True" or r.get("value") in (None, ""):
            continue
        by_kernel[r["kernel_ref"]].append(float(r["value"]))

    fig, axes = plt.subplots(1, len(by_kernel), figsize=(3 * len(by_kernel), 4))
    if len(by_kernel) == 1:
        axes = [axes]
    summary_rows = []
    for ax, (kernel, values) in zip(axes, sorted(by_kernel.items())):
        mean = statistics.fmean(values)
        cv = statistics.pstdev(values) / mean * 100.0 if mean else None
        ax.plot(range(1, len(values) + 1), values, marker="o", color=color_for(0))
        ax.axhline(mean, color="#52514e", linestyle="--", linewidth=1)
        ax.set_title(f"{kernel}\nCV={cv:.3f}%%" if cv is not None else kernel, fontsize=8)
        ax.set_xlabel("Repetición")
        deviation_pct = None
        if kernel in DATASHEET_CPU:
            deviation_pct = abs(mean - DATASHEET_CPU[kernel]) / DATASHEET_CPU[kernel] * 100.0
        summary_rows.append((kernel, mean, cv, deviation_pct))
    axes[0].set_ylabel("Valor medido (unidad nativa)")
    fig.suptitle("Distribución de 10 calibraciones independientes")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "calibration_distribution.png")
    plt.close(fig)

    table_path = PLOTS_DIR / "table_summary.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel_ref", "mean", "cv_pct", "deviation_vs_datasheet_pct"])
        for kernel, mean, cv, dev in summary_rows:
            writer.writerow([kernel, mean, round(cv, 4) if cv is not None else "", round(dev, 4) if dev is not None else ""])
    print(f"Escrito: {PLOTS_DIR}/calibration_distribution.png, {table_path}")


if __name__ == "__main__":
    main()
