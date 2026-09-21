"""Auditoria de correlacion y colinealidad del vector final CPU (Plan Detallado, seccion 2.5).

Sobre TODOS los intervalos elegibles (no sobre la muestra con tope): correlacion de Pearson y de
Spearman entre las variables de entrada, pares con |rho| por encima del umbral y factor de inflacion
de la varianza (VIF) de cada variable. El VIF se calcula desde la matriz de correlacion como la
diagonal de su inversa, que equivale a regresar cada columna estandarizada sobre las demas
(VIF_j = 1 / (1 - R_j^2)); se reporta con Pearson y con rangos (Spearman).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis import cpu_quality_report as q

PAIR_THRESHOLD = 0.85
VIF_ALERT = 10.0


def vif_from_corr(corr: pd.DataFrame) -> pd.Series:
    return pd.Series(np.diag(np.linalg.inv(corr.to_numpy())), index=corr.columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = q.load(a.source)
    features = list(q.CAND)
    x = frame[features].astype(float)
    pearson = x.corr(method="pearson")
    spearman = x.rank().corr(method="pearson")  # Spearman = Pearson sobre rangos
    pearson.to_csv(out / "final6_pearson.csv")
    spearman.to_csv(out / "final6_spearman.csv")
    pairs = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            pairs.append({"a": f1, "b": f2, "pearson": pearson.loc[f1, f2], "spearman": spearman.loc[f1, f2],
                          "max_abs": max(abs(pearson.loc[f1, f2]), abs(spearman.loc[f1, f2]))})
    pairs = pd.DataFrame(pairs).sort_values("max_abs", ascending=False)
    pairs["above_threshold"] = pairs["max_abs"] > PAIR_THRESHOLD
    pairs.to_csv(out / "final6_pairs.csv", index=False)
    vif = pd.DataFrame({"vif_pearson": vif_from_corr(pearson), "vif_spearman": vif_from_corr(spearman)})
    vif.to_csv(out / "final6_vif.csv")
    summary = {"n_intervals": int(len(x)), "features": features, "pair_threshold": PAIR_THRESHOLD, "vif_alert": VIF_ALERT,
               "pairs_above_threshold": int(pairs["above_threshold"].sum()),
               "max_abs_pair": {k: pairs.iloc[0][k] if k in ("a", "b") else float(pairs.iloc[0][k])
                                for k in ("a", "b", "pearson", "spearman", "max_abs")},
               "max_vif_pearson": float(vif["vif_pearson"].max()), "max_vif_spearman": float(vif["vif_spearman"].max()),
               "vif_above_alert": int((vif.max(axis=1) > VIF_ALERT).sum())}
    (out / "final6_audit.json").write_text(json.dumps(summary, indent=1))
    print(pairs.round(3).to_string(index=False))
    print(vif.round(2).to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
