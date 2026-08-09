#!/usr/bin/env python3
"""Sweep H (docs/justifications): grafico y tabla de smt_policy (npb_cg, 5 reps c/u)."""
import csv
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_style, color_for
import matplotlib.pyplot as plt

DATA_CSV = Path(__file__).parent.parent / "data" / "smt_policy" / "summary.csv"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "one_thread_per_physical_core": "1 hilo / núcleo físico (6 hilos)",
    "all_threads": "SMT completo (12 hilos, 2/núcleo)",
}
ORDER = ["one_thread_per_physical_core", "all_threads"]


def main():
    rows = list(csv.DictReader(open(DATA_CSV)))
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    table_rows = []
    for i, policy in enumerate(ORDER):
        sub = [r for r in rows if r["smt_policy"] == policy]
        ipc_vals = [float(r["ipc_mean"]) for r in sub]
        mean_ipc = statistics.fmean(ipc_vals)
        cv_ipc = statistics.pstdev(ipc_vals) / mean_ipc * 100.0
        elapsed_vals = [float(r["elapsed_seconds"]) for r in sub]
        mpki_vals = [float(r["mpki_mean"]) for r in sub]
        llc_vals = [float(r["llc_miss_rate_mean"]) for r in sub]

        xs = [i] * len(ipc_vals)
        ax.scatter(xs, ipc_vals, color=color_for(i), zorder=3, label=LABELS[policy])
        ax.hlines(mean_ipc, i - 0.15, i + 0.15, color=color_for(i), linewidth=2.5, zorder=4)

        table_rows.append({
            "smt_policy": policy,
            "n_threads": sub[0]["n_threads"],
            "ipc_mean": mean_ipc,
            "ipc_cv_pct": cv_ipc,
            "mpki_mean": statistics.fmean(mpki_vals),
            "llc_miss_rate_mean": statistics.fmean(llc_vals),
            "elapsed_seconds_mean": statistics.fmean(elapsed_vals),
        })

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABELS[p] for p in ORDER], fontsize=8)
    ax.set_ylabel("IPC medio por ventana (npb_cg)")
    ax.set_title("Costo de SMT en IPC: npb_cg, 6 núcleos físicos fijos")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "smt_policy_ipc.png")
    plt.close(fig)

    table_csv = PLOTS_DIR / "smt_policy_table.csv"
    with open(table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    drop_pct = (1 - table_rows[1]["ipc_mean"] / table_rows[0]["ipc_mean"]) * 100.0
    print(f"Grafico: {PLOTS_DIR / 'smt_policy_ipc.png'}")
    print(f"Tabla: {table_csv}")
    print(f"Caida de IPC por SMT: {drop_pct:.1f}%")


if __name__ == "__main__":
    main()
