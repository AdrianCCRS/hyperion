#!/usr/bin/env python3
"""Auditoría fresca de actividad real por kernel GPU (ARC-188).

Recalcula, sobre la campaña pacca_gpu_dvfs_20260820, qué fracción de las
ventanas gpu_telemetry de cada kernel muestra actividad real bajo DOS
criterios: el piso de utilización histórico (gpu_util_pct >= 5%) y --donde
hay línea de reposo-- el criterio de potencia de C.3 (ARC-185). Existe
para no citar de memoria cifras de una conversación anterior sin
reverificarlas contra los datos.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

import pandas as pd

RUN_RE = re.compile(r"^(?P<campaign>.+?)__(?P<kernel>.+?)__(?P<cpu_level>[^_]+)__(?P<gpu_level>gpu[A-Z0-9]+)__rep(?P<rep>\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    print(f"nodo de análisis: {platform.node()}\n", flush=True)

    rows = []
    for path in sorted(campaign_dir.iterdir()):
        if not path.is_dir() or "__baseline" in path.name:
            continue
        m = RUN_RE.match(path.name)
        if not m or m.group("rep") == "00":
            continue
        windows_path = path / "windows.csv"
        if not windows_path.exists():
            continue
        df = pd.read_csv(windows_path, usecols=[
            "quality_status", "gpu_util_pct", "gpu_power_mw", "gpu_freq_level_id",
        ])
        gpu_rows = df[df["quality_status"] == "gpu_telemetry"]
        if gpu_rows.empty:
            continue
        util = pd.to_numeric(gpu_rows["gpu_util_pct"], errors="coerce")
        rows.append({
            "kernel": m.group("kernel"),
            "gpu_level": m.group("gpu_level"),
            "n_gpu_telemetry": int(len(gpu_rows)),
            "frac_util_ge_5pct": float((util >= 5.0).mean()),
            "util_median": float(util.median()),
            "power_mw_median": float(pd.to_numeric(gpu_rows["gpu_power_mw"], errors="coerce").median()),
        })

    table = pd.DataFrame(rows)
    by_kernel = table.groupby("kernel").agg(
        n_runs=("gpu_level", "size"),
        frac_util_ge_5pct_mean=("frac_util_ge_5pct", "mean"),
        frac_util_ge_5pct_min=("frac_util_ge_5pct", "min"),
        frac_util_ge_5pct_max=("frac_util_ge_5pct", "max"),
        util_median=("util_median", "median"),
        power_mw_median=("power_mw_median", "median"),
    ).round(4).reset_index().sort_values("frac_util_ge_5pct_mean", ascending=False)

    print(by_kernel.to_string(index=False), flush=True)

    by_kernel_level = table.groupby(["kernel", "gpu_level"]).agg(
        frac_util_ge_5pct=("frac_util_ge_5pct", "mean"),
    ).round(4).reset_index()
    print("\n== por nivel de reloj GPU (para ver si el piso crece al bajar, F3) ==", flush=True)
    print(by_kernel_level.to_string(index=False), flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(),
        "by_kernel": by_kernel.to_dict(orient="records"),
        "by_kernel_level": by_kernel_level.to_dict(orient="records"),
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
