"""Contrafactual de `freq_khz_observed` en el clasificador de CPU vigente (xgboost_cpu.joblib). El daemon de CPU
cambia la frecuencia (F0 = 3.2 GHz base, F1 = 2.9 GHz en memory) y esa misma cantidad entra al modelo como variable:
si el modelo dependiera de ella, actuar cambiaria lo que el clasificador ve. La ablacion LOFO (sin frecuencia:
0.709 vs ~0.70 con ella) ya estaba en el informe de calidad; falta medir la sensibilidad del MODELO FINAL.

Para cada fila del dataset de entrenamiento se reemplaza la frecuencia observada por un valor fijo y se cuenta cuantas
predicciones (P(memory_bound), regla de decision 0.5) cambian, en global, por clase real y por familia. Referencia: la
misma sustitucion con el valor real (0 cambios). Correr en pacca (el CSV esta alli); nunca en local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FREQ = "freq_khz_observed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads(a.metadata.read_text())
    features = meta["features"]
    model = joblib.load(a.model)
    df = pd.read_csv(a.source, low_memory=False).dropna(subset=features)
    label_col = "phase_label_train" if "phase_label_train" in df else [c for c in df.columns if "label" in c][0]
    fam_col = "kernel_family" if "kernel_family" in df else [c for c in df.columns if "family" in c][0]
    y = df[label_col].eq("memory_bound").to_numpy()
    fam = df[fam_col].to_numpy()
    X = df[features].to_numpy(dtype=np.float32)
    ci = features.index(FREQ)
    base = model.predict_proba(X)[:, 1]
    base_pred = base >= 0.5
    print(f"filas={len(df)} familias={len(set(fam))} frecuencia observada (kHz): "
          f"{df[FREQ].describe().round(0).to_dict()}")
    report = {"n_rows": int(len(df)), "features": features, "contrafactual": {}}
    for label, khz in (("F0_3200MHz", 3_200_000), ("F1_2900MHz", 2_900_000), ("F4_2000MHz", 2_000_000), ("F8_800MHz", 800_000)):
        Xc = X.copy()
        Xc[:, ci] = khz
        p = model.predict_proba(Xc)[:, 1]
        pred = p >= 0.5
        flipped = pred != base_pred
        per_fam = pd.Series(flipped).groupby(fam).mean().sort_values(ascending=False)
        entry = {
            "flip_rate": round(float(flipped.mean()), 4),
            "flip_si_clase_real_compute": round(float(flipped[~y].mean()), 4),
            "flip_si_clase_real_memory": round(float(flipped[y].mean()), 4),
            "compute_a_memory": int((flipped & ~base_pred).sum()), "memory_a_compute": int((flipped & base_pred).sum()),
            "cambio_medio_absoluto_de_proba": round(float(np.abs(p - base).mean()), 4),
            "familia_mas_sensible": {per_fam.index[0]: round(float(per_fam.iloc[0]), 4)},
        }
        report["contrafactual"][label] = entry
        print(label, json.dumps(entry, ensure_ascii=False))
    # F0 -> F1: el cambio que hace el daemon; se mide solo sobre las filas observadas a ~3.2 GHz (la frecuencia base)
    at_f0 = np.abs(df[FREQ].to_numpy() - 3_200_000) <= 0.05 * 3_200_000
    if at_f0.sum():
        Xf = X[at_f0].copy()
        Xf[:, ci] = 2_900_000
        flipped = (model.predict_proba(Xf)[:, 1] >= 0.5) != base_pred[at_f0]
        report["F0_a_F1_solo_filas_a_3200MHz"] = {"n": int(at_f0.sum()), "flip_rate": round(float(flipped.mean()), 4)}
        print("F0->F1 sobre filas a ~3.2 GHz:", report["F0_a_F1_solo_filas_a_3200MHz"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
