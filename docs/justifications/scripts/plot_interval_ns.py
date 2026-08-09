#!/usr/bin/env python3
"""Genera tablas/figuras del Sweep A (interval_ns) a partir de
docs/justifications/data/interval_ns/summary.csv (traido de pacca)."""
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "interval_ns"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "interval_ns"


import re

_RUN_DIR_RE = re.compile(r"interval_(\d+)/sweep_interval_\d+__([a-zA-Z0-9_]+)__REF__rep(\d+)")


def _load_rows():
    # summary.csv tiene interval_ns/repetition/sampling_interval_cv_pct
    # correctos, pero n_windows_ok en 0 para todas las filas: se llamo a
    # run_postprocess() con freq_khz_observed=None, lo que marca cada
    # ventana "no_freq_reading" (comportamiento correcto de la taxonomia de
    # calidad, error de esta herramienta de barrido, no de postprocess.py).
    # summary_reprocessed.csv reprocesa samples.csv (ya en disco, sin
    # recomputo de GPU) con un valor de relleno para obtener el conteo real
    # de ventanas 'ok'. Se combinan aqui por (kernel_ref, interval_ns,
    # repeticion). dgemm_n2048_fix.csv agrega el unico kernel que fallo en
    # la corrida original por libopenblas.so.0 ausente del LD_LIBRARY_PATH
    # bajo sbatch (rerun_dgemm_n2048.py) -- ya trae ambos valores correctos.
    with open(DATA_DIR / "summary.csv", newline="") as f:
        base_rows = list(csv.DictReader(f))

    corrected_ok = {}
    with open(DATA_DIR / "summary_reprocessed.csv", newline="") as f:
        for r in csv.DictReader(f):
            match = _RUN_DIR_RE.search(r["run_dir"])
            if not match:
                continue
            interval_ns, kernel_ref, repetition = match.group(1), match.group(2), match.group(3)
            corrected_ok[(kernel_ref, interval_ns, str(int(repetition)))] = r.get("n_windows_ok")

    rows = []
    for r in base_rows:
        if r["success"] != "True":
            continue
        key = (r["kernel_ref"], r["interval_ns"], r["repetition"])
        if key in corrected_ok:
            r["n_windows_ok"] = corrected_ok[key]
        rows.append(r)

    fix_path = DATA_DIR / "dgemm_n2048_fix.csv"
    if fix_path.exists():
        with open(fix_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_rows()

    # Ventanas ok logradas y CV% de sampling_interval_cv_pct, por kernel x interval_ns
    by_kernel = defaultdict(lambda: defaultdict(list))
    overhead_by_interval = defaultdict(list)
    for r in rows:
        if r["success"] != "True":
            continue
        interval_ns = int(r["interval_ns"])
        kernel = r["kernel_ref"]
        n_ok = int(r["n_windows_ok"]) if r.get("n_windows_ok") not in (None, "") else None
        if n_ok is not None:
            by_kernel[kernel][interval_ns].append(n_ok)
        cv = r.get("sampling_interval_cv_pct")
        if cv not in (None, ""):
            overhead_by_interval[interval_ns].append(float(cv))

    # Figura 1: ventanas 'ok' logradas vs interval_ns, por kernel
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, kernel in enumerate(sorted(by_kernel)):
        xs = sorted(by_kernel[kernel])
        ys = [statistics.fmean(by_kernel[kernel][x]) for x in xs]
        ax.plot([x / 1e6 for x in xs], ys, marker="o", color=color_for(i), label=kernel)
    ax.axhline(50, color="#52514e", linestyle="--", linewidth=1, label="target_windows_per_repetition=50")
    ax.set_xlabel("interval_ns (ms)")
    ax.set_ylabel("Ventanas 'ok' logradas (media por repetición)")
    ax.set_yscale("log")
    ax.set_title("Ventanas útiles logradas vs. cadencia de muestreo CPU")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "windows_ok_vs_interval.png")
    plt.close(fig)

    # Figura 2: overhead del propio muestreo (sampling_interval_cv_pct) vs interval_ns
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(overhead_by_interval)
    means = [statistics.fmean(overhead_by_interval[x]) for x in xs]
    ax.plot([x / 1e6 for x in xs], means, marker="o", color=color_for(0))
    ax.set_xlabel("interval_ns (ms)")
    ax.set_ylabel("CV%% del intervalo de muestreo real")
    ax.set_title("Fidelidad del muestreo vs. cadencia solicitada")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sampling_cv_vs_interval.png")
    plt.close(fig)

    # Tabla resumen en CSV (para incluir en LaTeX con \csvautotabular o manual)
    table_path = PLOTS_DIR / "table_summary.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["interval_ns_ms", "kernel_ref", "windows_ok_mean", "sampling_cv_pct_mean"])
        for kernel in sorted(by_kernel):
            for interval_ns in sorted(by_kernel[kernel]):
                writer.writerow([
                    interval_ns / 1e6, kernel,
                    round(statistics.fmean(by_kernel[kernel][interval_ns]), 1),
                    round(statistics.fmean(overhead_by_interval[interval_ns]), 3) if overhead_by_interval[interval_ns] else "",
                ])
    print(f"Escrito: {PLOTS_DIR}/windows_ok_vs_interval.png, sampling_cv_vs_interval.png, {table_path}")


if __name__ == "__main__":
    main()
