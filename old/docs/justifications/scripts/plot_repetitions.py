#!/usr/bin/env python3
"""Genera figuras del Sweep C/D (repetitions_per_combination /
target_windows_per_repetition) a partir de
docs/justifications/data/repetitions/summary.csv (traido de pacca).

Convergencia de CV%% en funcion de n (numero de repeticiones acumuladas,
3..10) calculada sobre el MISMO conjunto de 10 corridas por kernel -- no
hace falta una campana nueva por cada n.
"""
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

import re

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "repetitions"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "repetitions"

_RUN_DIR_RE = re.compile(r"sweep_repetitions__([a-zA-Z0-9_]+)__REF__rep(\d+)")


def cv_pct(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / abs(mean) * 100.0


def _load_rows():
    # elapsed_seconds viene directo del harness (RunResult), nunca pasa por
    # el bug de freq_khz_observed=None que sí afecta windows_achieved para
    # kernels de CPU (ver interval_ns) -- solo windows_achieved necesita
    # corregirse aqui, con el mismo criterio que plot_interval_ns.py.
    with open(DATA_DIR / "summary.csv", newline="") as f:
        base_rows = list(csv.DictReader(f))

    corrected = {}
    with open(DATA_DIR / "summary_reprocessed.csv", newline="") as f:
        for r in csv.DictReader(f):
            match = _RUN_DIR_RE.search(r["run_dir"])
            if not match:
                continue
            kernel_ref, repetition = match.group(1), str(int(match.group(2)))
            achieved = r.get("n_windows_ok") if r.get("n_windows_ok") not in (None, "") else r.get("n_gpu_telemetry")
            corrected[(kernel_ref, repetition)] = achieved

    rows = []
    for r in base_rows:
        if r["success"] != "True":
            continue
        key = (r["kernel_ref"], r["repetition"])
        if key in corrected:
            r["windows_achieved"] = corrected[key]
        rows.append(r)

    rows = [r for r in rows if r["kernel_ref"] != "rodinia_heartwall"]

    fix_path = DATA_DIR / "dgemm_n2048_fix.csv"
    if fix_path.exists():
        with open(fix_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    hw_fix_path = DATA_DIR / "rodinia_heartwall_fix.csv"
    if hw_fix_path.exists():
        with open(hw_fix_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_rows()

    by_kernel = defaultdict(list)
    target_by_kernel = {}
    device_by_kernel = {}
    for r in rows:
        if r["success"] != "True":
            continue
        kernel = r["kernel_ref"]
        by_kernel[kernel].append((
            int(r["repetition"]),
            float(r["elapsed_seconds"]),
            int(r["windows_achieved"]) if r.get("windows_achieved") not in (None, "") else None,
        ))
        target_by_kernel[kernel] = int(r["target_windows_per_repetition"])
        device_by_kernel[kernel] = r["device"]

    # Convergencia de CV% de elapsed_seconds en funcion de n repeticiones
    # acumuladas -- separado por dispositivo (nunca mas de 8 series por
    # panel: con los 15 kernels juntos, la paleta categorica de 8 colores
    # se repetiria y dos kernels distintos compartirian color, exactamente
    # el tipo de ambiguedad que el metodo de dataviz prohibe).
    convergence_table = []
    for device in ("cpu", "gpu"):
        kernels_this_device = sorted(k for k in by_kernel if device_by_kernel[k] == device)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for i, kernel in enumerate(kernels_this_device):
            entries = sorted(by_kernel[kernel])
            elapsed_series = [e[1] for e in entries]
            ns = list(range(3, len(elapsed_series) + 1))
            cvs = [cv_pct(elapsed_series[:n]) for n in ns]
            ax.plot(ns, cvs, marker="o", color=color_for(i), label=kernel, alpha=0.85)
            convergence_table.append((kernel, cvs[0], cvs[-1]))
        ax.set_xlabel("Número de repeticiones acumuladas (n)")
        ax.set_ylabel("CV%% de tiempo de ejecución (0..n)")
        ax.set_title(f"Convergencia de CV%% de tiempo de ejecución -- kernels {device.upper()}")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"cv_convergence_vs_repetitions_{device}.png")
        plt.close(fig)

    # Ventanas logradas vs objetivo configurado
    table_path = PLOTS_DIR / "table_windows_achieved.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel_ref", "device", "target_windows", "windows_achieved_mean", "windows_achieved_min", "cv3", "cv10"])
        for kernel, cv3, cv10 in convergence_table:
            achieved = [e[2] for e in by_kernel[kernel] if e[2] is not None]
            writer.writerow([
                kernel, device_by_kernel[kernel], target_by_kernel[kernel],
                round(statistics.fmean(achieved), 1) if achieved else "",
                min(achieved) if achieved else "",
                round(cv3, 3) if cv3 is not None else "",
                round(cv10, 3) if cv10 is not None else "",
            ])
    print(f"Escrito: {PLOTS_DIR}/cv_convergence_vs_repetitions.png, {table_path}")


if __name__ == "__main__":
    main()
