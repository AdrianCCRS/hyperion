"""Agrega `power_w` por intervalo uncore a la matriz de entrenamiento CPU.

Cada corrida trae `training_cpu_intervals.csv` (una fila por intervalo uncore)
y `windows.csv` (ventanas de ~1 ms con energía RAPL). La potencia de un
intervalo es la energía válida de sus ventanas (paquete + DRAM) sobre su
duración. Salida: un CSV con las columnas de la matriz más `power_w`, listo
para `cpu_quality_report --stages power_w`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def run_power(windows_csv: Path) -> pd.DataFrame:
    w = pd.read_csv(windows_csv, usecols=["uncore_interval_id", "delta_t_ns", "pkg_delta_uj", "dram_delta_uj", "energy_valid"],
                    low_memory=False)
    w = w[w["energy_valid"].astype(str).str.lower().isin(["true", "1"]) & w["uncore_interval_id"].notna()]
    w = w.assign(e=(w["pkg_delta_uj"].fillna(0) + w["dram_delta_uj"].fillna(0)) / 1e6, t=w["delta_t_ns"] / 1e9)
    g = w.groupby("uncore_interval_id").agg(e=("e", "sum"), t=("t", "sum"))
    g["power_w"] = g["e"] / g["t"]
    return g[["power_w"]].reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    frames, missing = [], 0
    for d in sorted(a.campaign_dir.glob("*__rep0?")):
        t, w = d / "training_cpu_intervals.csv", d / "windows.csv"
        if not (t.exists() and w.exists()):
            missing += 1
            continue
        df = pd.read_csv(t, low_memory=False)
        df = df.merge(run_power(w), left_on="uncore_interval_id", right_on="uncore_interval_id", how="left")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"runs={len(frames)} sin_archivos={missing} filas={len(out)} power_w_nan={out['power_w'].isna().mean():.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
