"""Tablas derivadas de la matriz CPU final que alimentan cifras y figuras del libro.

Salidas en --out: kernel_counts.csv (elegibles por kernel y clase), level_counts.csv
(por nivel de frecuencia y clase), family_counts.csv (elegibles y muestra por
familia), sample_summary.json (totales de la muestra con tope por celda),
feature_correlation.csv (Pearson de las 12 entradas sobre esa muestra) y
family_class_frequency_summary.csv (familia, kernel, nivel, conteos por clase).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis import cpu_quality_report as q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = q.load(a.source)
    f["cls"] = np.where(f["y"], "memory_bound", "compute_bound")
    kc = f.pivot_table(index="kernel_ref", columns="cls", values="y", aggfunc="size", fill_value=0)
    kc.to_csv(out / "kernel_counts.csv")
    lc = f.pivot_table(index="freq_level_id", columns="cls", values="y", aggfunc="size", fill_value=0)
    lc.to_csv(out / "level_counts.csv")
    fc = f.pivot_table(index="family", columns="cls", values="y", aggfunc="size", fill_value=0)
    sample = q.capped_sample(f, a.cap, 1000)
    fc["elegibles"] = fc.sum(axis=1)
    fc["muestra"] = f.iloc[sample].groupby("family").size().reindex(fc.index).fillna(0).astype(int)
    fc.rename(columns={"compute_bound": "n_compute_bound", "memory_bound": "n_memory_bound"}).to_csv(out / "family_counts.csv")
    ys = f.iloc[sample]["y"]
    cells = f.iloc[sample].groupby(["family", "y"]).size()
    json.dump({"n_eligible": int(len(f)), "n_compute": int((~f["y"]).sum()), "n_memory": int(f["y"].sum()),
               "sample_rows": int(len(sample)), "sample_compute": int((~ys).sum()), "sample_memory": int(ys.sum()),
               "n_cells": int(f.groupby(["family", "y"]).ngroups), "n_families": int(f["family"].nunique()),
               "n_kernels": int(f["kernel_ref"].nunique())}, open(out / "sample_summary.json", "w"), indent=1)
    f.iloc[sample][q.INTER].corr(method="pearson").to_csv(out / "feature_correlation.csv")
    g = f.groupby(["family", "kernel_ref", "freq_level_id", "cls"]).size().unstack(fill_value=0).reset_index()
    g["n_rows_usable"] = g.get("compute_bound", 0) + g.get("memory_bound", 0)
    g = g.rename(columns={"compute_bound": "n_compute_bound", "memory_bound": "n_memory_bound"})
    g["memory_fraction"] = g["n_memory_bound"] / g["n_rows_usable"]
    g.to_csv(out / "family_class_frequency_summary.csv", index=False)
    print(json.load(open(out / "sample_summary.json")))


if __name__ == "__main__":
    main()
