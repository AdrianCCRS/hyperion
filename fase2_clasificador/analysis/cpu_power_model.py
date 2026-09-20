"""Modelo de potencia de paquete P(f) = P0 + c f^k y umbral de alfa para DVFS.

Ajusta, por kernel, la potencia media de la corrida (RAPL paquete + DRAM sobre
tiempo) contra la frecuencia fija (F0..F8 = 3200..800 MHz), y calcula el
alfa critico: el mayor alfa (T ~ f^-alfa) para el que algun nivel bajo mejora
el EDP respecto de F0 con ese modelo de potencia. Salida: CSV y JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CAL = ("stream_official", "ert_probe")
FMAX = 3200.0


def freq_mhz(level: str) -> float:
    return FMAX - 300.0 * int(level[1])


def model(f, p0, c, k):
    return p0 + c * (f / FMAX) ** k


def fit_kernel(g: pd.DataFrame):
    f, p = g.f.to_numpy(), g.p.to_numpy()
    try:
        (p0, c, k), _ = curve_fit(model, f, p, p0=[p.min(), p.max() - p.min(), 2.0],
                                  bounds=([0, 0, 1.0], [400, 400, 4.0]), maxfev=20000)
    except Exception:
        return None
    res = p - model(f, p0, c, k)
    return p0, c, k, float(np.sqrt(np.mean(res ** 2))), float(p[f.argmax()])


def critical_alpha(p0: float, c: float, k: float) -> float:
    """Mayor alfa en [0,1] con algun f<FMAX que mejora EDP = P(f) * f^(-2 alfa)."""
    fs = np.linspace(800, 3200, 200)
    for a in np.linspace(1.0, 0.0, 1001):
        edp = model(fs, p0, c, k) * (fs / FMAX) ** (-2 * a)
        if edp.min() < edp[-1] * 0.999:
            return float(a)
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(p) for p in a.runs_csv])
    d = d[d.accepted.astype(bool) & d.level.str.match(r"^F[0-8]$") & ~d.kernel_ref.isin(CAL)].copy()
    d["f"] = d.level.map(freq_mhz)
    d["p"] = (d.pkg_uj + d.dram_uj) / 1e6 / (d.elapsed_ns / 1e9)
    m = d.groupby(["kernel_ref", "f"]).p.median().reset_index()
    rows = []
    for k, g in m.groupby("kernel_ref"):
        if len(g) < 6:
            continue
        r = fit_kernel(g)
        if r is None:
            continue
        p0, c, k_, rmse, pmax = r
        rows.append(dict(kernel_ref=k, p0_w=p0, c_w=c, k=k_, rmse_w=rmse, p_f0_w=pmax,
                         static_frac_f0=p0 / (p0 + c), power_drop_f8=1 - model(800, p0, c, k_) / model(3200, p0, c, k_),
                         alpha_crit=critical_alpha(p0, c, k_)))
    t = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / "power_model_by_kernel.csv", index=False)
    s = {"n_kernels": len(t), **{f"median_{c}": float(t[c].median()) for c in
         ("p0_w", "c_w", "k", "rmse_w", "static_frac_f0", "power_drop_f8", "alpha_crit")}}
    (a.out / "power_model_summary.json").write_text(json.dumps(s, indent=1))
    print(t.round(3).to_string(index=False))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
