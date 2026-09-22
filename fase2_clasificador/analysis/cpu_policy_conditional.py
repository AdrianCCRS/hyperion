"""Politica de frecuencia CPU para memory-bound condicionada a una variable PMU (una sola regla de umbral).

Regla declarada antes de mirar resultados: actuar en el nivel L (F1..F4) si la variable v (de las 4 tasas PMU por
familia, mediana de sus ventanas memory) cumple v >= t (o v <= t); si no, no actuar (REF). Variable, direccion,
umbral y nivel se eligen SOLO con las demas familias (leave-one-family-out anidado) maximizando la ganancia media
de EDP (media geometrica); si esa ganancia de entrenamiento no llega a min_effect, no se actua. La ganancia
realizada se mide en la familia excluida. Unidad: familia (log-razon media de sus kernels, ver cpu_policy_by_family).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["stall_mem_ratio"]
LEVELS = ["F1", "F2"]


def best_rule(lr: pd.DataFrame, x: pd.DataFrame, min_effect: float):
    best = None
    for v in FEATURES:
        for t in sorted(set(x[v])):
            for d in (">=", "<="):
                act = (x[v] >= t) if d == ">=" else (x[v] <= t)
                if act.sum() < 2:
                    continue
                for lv in LEVELS:
                    g = np.where(act, 1 - np.exp(lr[lv]), 0.0)
                    m = float(g.mean())
                    if best is None or m > best[0]:
                        best = (m, v, d, float(t), lv)
    return best if best and best[0] >= min_effect else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=Path, required=True)
    ap.add_argument("--pmu", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-effect", type=float, default=0.01)
    a = ap.parse_args()
    lr = pd.read_csv(a.ratios)
    lr = lr[lr.cls == "memory_bound"].set_index("family")[LEVELS]
    pmu = pd.read_csv(a.pmu)
    x = pmu[pmu.y.astype(str) == "True"].set_index("family")[FEATURES]
    fams = sorted(set(lr.index) & set(x.index))
    lr, x = lr.loc[fams].dropna(), x.loc[fams]
    lr = lr.loc[lr.index]; x = x.loc[lr.index]
    rows = []
    for f in lr.index:
        rule = best_rule(lr.drop(index=f), x.drop(index=f), a.min_effect)
        if rule is None:
            rows.append(dict(family=f, rule=None, acted=False, realized=0.0)); continue
        _, v, d, t, lv = rule
        acts = x.loc[f, v] >= t if d == ">=" else x.loc[f, v] <= t
        rows.append(dict(family=f, rule=f"{lv} si {v} {d} {t:.4g}", acted=bool(acts),
                         realized=float(1 - np.exp(lr.loc[f, lv])) if acts else 0.0))
    res = pd.DataFrame(rows)
    r = res.realized.to_numpy()
    idx = np.random.default_rng(0).integers(0, len(r), (4000, len(r)))
    ci = np.percentile(r[idx].mean(axis=1), [2.5, 97.5])
    full = best_rule(lr, x, a.min_effect)
    acted = res[res.acted]
    summary = {"n_familias": len(r), "ganancia_media_realizada": round(float(r.mean()), 4),
               "ic95_bootstrap_familias": [round(float(c), 4) for c in ci],
               "familias_que_actuan": int(res.acted.sum()), "familias_que_empeoran": int((r < 0).sum()),
               "ganancia_media_donde_actua": round(float(acted.realized.mean()), 4) if len(acted) else None,
               "peor_familia": round(float(r.min()), 4), "mejor_familia": round(float(r.max()), 4),
               "regla_ajustada_con_todas_las_familias (solo descriptiva, optimista)": None if full is None else
               {"ganancia_entrenamiento": round(full[0], 4), "variable": full[1], "direccion": full[2], "umbral": full[3], "nivel": full[4]}}
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "policy_conditional_lofo.csv", index=False)
    (a.out / "policy_conditional.json").write_text(json.dumps(summary, indent=1))
    print(res.round(4).to_string(index=False)); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
