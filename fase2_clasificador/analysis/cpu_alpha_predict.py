"""Variante V2: predecir alpha (fraccion de tiempo insensible al reloj) por familia desde una sola
variable PMU, validado leave-one-family-out, sin usar EDP medido en la prediccion (Plan V2, ver
docs de la discusion de politica CPU memory-bound, 2026-09-22).

alpha se ajusta con el modelo T(f)/T(REF) = alpha + (1-alpha)*3.2/f sobre los 10 niveles de una
familia (minimos cuadrados). Luego alpha_hat = a + b*x (x = stall_mem_ratio o mpki), ajustada SIN
la familia evaluada. Regla: actuar en F2 si alpha_hat >= 0.80. Metrica: ganancia de EDP realizada
(medida, no predicha) promediada sobre las familias donde actua, con IC95 bootstrap de familias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis.cpu_policy_table import LEVELS, kernel_class, per_run
from fase2_clasificador.eval.protocol import derive_kernel_family as dk

F = {l: 3.2 - 0.3 * i for i, l in enumerate(LEVELS)}


def fit_alpha(time_ratio: pd.Series) -> tuple[float, float]:
    x = np.array([3.2 / F[l] for l in LEVELS if l in time_ratio.index])
    t = time_ratio.reindex([l for l in LEVELS if l in time_ratio.index]).to_numpy()
    A = np.c_[np.ones_like(x), x]
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    r2 = 1 - ((t - A @ [a, b]) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
    return a / (a + b), r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--pmu", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha-cut", type=float, default=0.80)
    ap.add_argument("--level", default="F2")
    a = ap.parse_args()
    runs = per_run(pd.concat([pd.read_csv(f) for f in a.runs], ignore_index=True))
    kc = kernel_class(runs)
    runs["family"] = runs.kernel_ref.map(dk)
    med = runs.groupby(["kernel_ref", "level"])[["time_s", "edp"]].median()

    fam_alpha, fam_gain = {}, {}
    for k in kc.index[kc["class"] == "memory_bound"]:
        g = med.loc[k]
        if "REF" not in g.index or a.level not in g.index or len(g) < 5:
            continue
        alpha, r2 = fit_alpha(g["time_s"] / g.loc["REF", "time_s"])
        gain = 1 - g.loc[a.level, "edp"] / g.loc["REF", "edp"]
        fam = dk(k)
        fam_alpha.setdefault(fam, []).append(alpha)
        fam_gain.setdefault(fam, []).append(gain)
    families = sorted(set(fam_alpha) & set(fam_gain))
    alpha_m = pd.Series({f: np.mean(fam_alpha[f]) for f in families})
    gain_m = pd.Series({f: np.mean(fam_gain[f]) for f in families})

    pmu = pd.read_csv(a.pmu)
    pmu = pmu[pmu.y.astype(str) == "True"].set_index("family")
    predictors = ["stall_mem_ratio", "mpki"]
    fams = sorted(set(families) & set(pmu.index))
    alpha_m, gain_m, pmu = alpha_m.loc[fams], gain_m.loc[fams], pmu.loc[fams]

    results = {}
    for pred in predictors:
        x = pmu[pred].to_numpy()
        y = alpha_m.to_numpy()
        pred_alpha, r2_lofo = [], []
        for i in range(len(fams)):
            mask = np.ones(len(fams), dtype=bool); mask[i] = False
            b, a0 = np.polyfit(x[mask], y[mask], 1)
            ah = a0 + b * x[i]
            pred_alpha.append(ah)
        pred_alpha = np.array(pred_alpha)
        ss_res = ((y - pred_alpha) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
        r2_lofo_full = 1 - ss_res / ss_tot
        acted = pred_alpha >= a.alpha_cut
        g = gain_m.to_numpy()
        realized = np.where(acted, g, 0.0)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(realized), (4000, len(realized)))
        ci = np.percentile(realized[idx].mean(axis=1), [2.5, 97.5])
        results[pred] = {
            "n_familias": len(fams), "r2_lofo": round(float(r2_lofo_full), 4),
            "n_actuan": int(acted.sum()),
            "ganancia_media_realizada_todas": round(float(realized.mean()), 4),
            "ic95_bootstrap": [round(float(c), 4) for c in ci],
            "ganancia_media_donde_actua": round(float(g[acted].mean()), 4) if acted.any() else None,
            "familias_actuan": [f for f, m in zip(fams, acted) if m],
        }
    out = {"alpha_cut": a.alpha_cut, "level": a.level, "por_predictor": results,
           "alpha_medido": alpha_m.round(4).to_dict(), "gain_medido": gain_m.round(4).to_dict()}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "v2_alpha_predict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
