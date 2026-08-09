#!/usr/bin/env python3
"""Genera las figuras de sensibilidad del detector de calentamiento
(Sweep E) a partir de docs/justifications/data/sensitivity_warmup_detector.csv.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "sensitivity_warmup_detector.csv"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "warmup_sensitivity"

PARAM_LABELS = {
    "cv_threshold_pct": "Umbral de CV%% (detección de calentamiento)",
    "margin": "Margen de seguridad (×)",
    "min_mean_floor": "Piso de ruido GPU (% de utilización)",
    "plateau_ratio": "Fracción de meseta (respaldo por punto de cambio)",
}
X_LABELS = {
    "cv_threshold_pct": "Umbral de CV%",
    "margin": "Margen ×",
    "min_mean_floor": "Piso de ruido (% util. GPU)",
    "plateau_ratio": "Fracción de meseta",
}


def main():
    apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    by_param_device = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["raw_warmup_s"] in (None, ""):
            continue
        key = (r["varied_param"], r["device"])
        by_param_device[key][r["kernel_ref"]].append((float(r["varied_value"]), float(r["raw_warmup_s"])))

    for (param, device), kernels in by_param_device.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (kernel_ref, points) in enumerate(sorted(kernels.items())):
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", color=color_for(i), label=kernel_ref)
        ax.set_xlabel(X_LABELS[param])
        ax.set_ylabel("Calentamiento detectado (s, crudo)")
        ax.set_title(f"Sensibilidad de {param} -- kernels {device.upper()}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_path = PLOTS_DIR / f"sensitivity_{param}_{device}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Escrito: {out_path}")


if __name__ == "__main__":
    main()
