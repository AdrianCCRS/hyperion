#!/usr/bin/env python3
"""Audita umbrales del RF GPU desplegado mediante probabilidades LOFO.

No reentrena ni cambia el modelo. Para cada semilla, deja una familia fuera,
entrena el RF de tres señales con el mismo peso por celda y guarda el promedio
de P(memory). La salida describe el costo de abstenerse para umbrales fijados;
no selecciona una politica por EDP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis import gpu_quality_report as gq

FEATURES = ["gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_mem_util_pct_std"]
THRESHOLDS = [0.90, 0.85, 0.80, 0.70, 0.60, 0.50]


def calibration_ece(frame: pd.DataFrame, n_bins: int = 10) -> tuple[float, list[dict[str, float | int]]]:
    """ECE sobre las probabilidades LOFO promedio del RF desplegable.

    La confianza es la probabilidad de la clase predicha, igual que la que
    consume el umbral de abstención. No se calibra sobre el modelo ajustado a
    todas las filas, porque eso mezclaría evaluación y ajuste.
    """
    probability = frame["p_memory"].to_numpy(dtype=float)
    truth = frame["truth"].eq("memory_bound").to_numpy()
    prediction = probability > 0.5
    confidence = np.maximum(probability, 1 - probability)
    correct = prediction == truth
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    bins: list[dict[str, float | int]] = []
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue
        conf = float(confidence[mask].mean())
        acc = float(correct[mask].mean())
        n = int(mask.sum())
        ece += n / len(frame) * abs(acc - conf)
        bins.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "n": n, "confianza_media": conf, "exactitud_media": acc})
    return float(ece), bins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frame = gq.load(a.source, split_rajaperf_cuda=True)
    gq.FEATURES[:] = FEATURES
    fams = sorted(frame[gq.FAMILY_COL].unique())
    fam = frame[gq.FAMILY_COL].to_numpy(); y = frame["y"].to_numpy()
    codes = pd.Categorical(fam, categories=fams).codes
    # El protocolo original promedia métricas de cada semilla, no probabilidades
    # entre semillas antes de aplicar abstención. Conservamos las probabilidades
    # medias solo como diagnóstico, pero la tabla primaria replica ese orden.
    by_seed: list[np.ndarray] = []
    total = np.zeros(len(frame), dtype=float)
    for seed in range(a.seeds):
        for f in fams:
            test = np.flatnonzero(fam == f); train = np.flatnonzero(fam != f)
            total[test] += gq.fit_predict("random_forest", frame, train, test, 2000 + seed, codes)
        by_seed.append(total.copy())
        total.fill(0)
    p = np.mean(np.stack(by_seed), axis=0)
    raw = pd.DataFrame({"family": fam, "truth": np.where(y, "memory_bound", "compute_bound"), "p_memory": p})
    raw["confidence"] = np.maximum(p, 1-p)
    raw.to_csv(a.out / "lofo_probabilities.csv", index=False)
    ece, calibration_bins = calibration_ece(raw)
    pd.DataFrame(calibration_bins).to_csv(a.out / "calibration_bins.csv", index=False)
    rows, family_rows = [], []
    for t in THRESHOLDS:
        seed_rows, seed_cells = [], []
        for si, p_seed in enumerate(by_seed):
            x = raw.assign(p_memory=p_seed, confidence=np.maximum(p_seed, 1-p_seed))
            d = x[x.confidence >= t].copy(); d["prediction"] = np.where(d.p_memory > .5, "memory_bound", "compute_bound")
            correct = d.prediction.eq(d.truth); fp = int(((d.truth == "compute_bound") & (d.prediction == "memory_bound")).sum()); fn = int(((d.truth == "memory_bound") & (d.prediction == "compute_bound")).sum())
            cell, cell_cov = [], []
            for f in fams:
                for c in ("compute_bound", "memory_bound"):
                    allc = x[(x.family == f) & (x.truth == c)]; dec = d[(d.family == f) & (d.truth == c)]
                    if len(allc):
                        acc = float(dec.prediction.eq(c).mean()) if len(dec) else np.nan
                        cov = len(dec)/len(allc); cell.append(acc); cell_cov.append(cov); seed_cells.append({"threshold":t,"seed":si,"family":f,"class":c,"n":len(allc),"decided":len(dec),"coverage":cov,"accuracy_decided":acc})
            seed_rows.append({"n":len(raw),"decided":len(d),"row_coverage":len(d)/len(raw),"cell_coverage":float(np.mean(cell_cov)),"accuracy_decided":float(correct.mean()) if len(d) else np.nan,"cell_balanced_accuracy_decided":float(np.nanmean(cell)),"false_memory_on_compute":fp,"false_compute_on_memory":fn,"memory_precision":float(((d.truth == "memory_bound") & (d.prediction == "memory_bound")).sum()/max((d.prediction == "memory_bound").sum(),1)),"memory_recall_decided":float(((d.truth == "memory_bound") & (d.prediction == "memory_bound")).sum()/max((d.truth == "memory_bound").sum(),1))})
        rows.append({"threshold":t, **pd.DataFrame(seed_rows).mean(numeric_only=True).to_dict()})
        family_rows.extend(seed_cells)
    pd.DataFrame(rows).to_csv(a.out / "threshold_summary.csv", index=False)
    pd.DataFrame(family_rows).groupby(["threshold","family","class","n"], as_index=False).mean(numeric_only=True).to_csv(a.out / "threshold_by_family_cell.csv", index=False)
    (a.out / "metadata.json").write_text(json.dumps({"model":"random_forest historical_20260923_sin_reloj","features":FEATURES,"seeds":a.seeds,"thresholds":THRESHOLDS,"rule":"memory if p_memory > 0.5; confidence=max(p_memory,1-p_memory)","ece":ece,"ece_bins":10,"scope":"retrospective LOFO only; no EDP or policy selection"}, indent=2)+"\n")

if __name__ == "__main__": main()
