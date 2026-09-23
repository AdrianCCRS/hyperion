"""Auditoria del reloj SM como variable del clasificador GPU (Bloque D,
Plan_Fase3_Daemon.md). El daemon fija el reloj (F1 = 1260 MHz) y esa misma
variable entra al clasificador (`gpu_sm_clock_mhz_median`): hay riesgo de
retroalimentacion. Este script responde con datos, sin suponer la causa:

1. ABLACION (LOFO por familia, 5 semillas, ponderacion por celda, IC95 por
   bootstrap de familias -- el mismo protocolo de `gpu_quality_report.py`):
   variables actuales vs sin reloj vs sin reloj ni potencia (la potencia
   tambien depende del reloj). Se reporta la exactitud balanceada por celda y
   la diferencia PAREADA contra el conjunto actual.
2. CONTRAFACTUAL: con el modelo final ajustado sobre todas las filas, se
   reemplaza el reloj de cada corrida por el nativo (1410) y por el de la
   politica (1260) y se cuenta cuantas predicciones cambian. Si casi ninguna
   cambia, el modelo no depende del reloj; si cambian, si.
3. Pesos estandarizados de la regresion logistica final, para ver cuanto pesa
   el reloj frente al resto.

Correr en pacca (sbatch), nunca en local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis import gpu_quality_report as gq

CLOCK = "gpu_sm_clock_mhz_median"
POWER = "gpu_power_mw_median"
MODELS = ["regresion_log", "random_forest"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # Mismo conjunto del candidato vigente: 4 variables base + gpu_mem_util_pct_std, rajaperf_cuda separado.
    frame = gq.load(a.source, split_rajaperf_cuda=True, extra_features=["gpu_mem_util_pct_std"])
    actual = list(gq.FEATURES)
    print(f"variables actuales: {actual}  filas={len(frame)}", flush=True)
    variants = {
        "actual": actual,
        "sin_reloj": [f for f in actual if f != CLOCK],
        "sin_reloj_ni_potencia": [f for f in actual if f not in (CLOCK, POWER)],
    }
    families = sorted(frame[gq.FAMILY_COL].unique())
    fam_codes = pd.Categorical(frame[gq.FAMILY_COL], categories=families).codes

    # 1) ablacion
    counts = {}
    for vname, feats in variants.items():
        gq.FEATURES[:] = feats  # las funciones de gq leen la lista global
        for m in MODELS:
            per_seed = [gq.lofo(frame, families, fam_codes, m, 2000 + s) for s in range(a.seeds)]
            counts[f"{m}|{vname}"] = np.mean(np.stack(per_seed), axis=0)
            mm = gq.metrics_from_counts(counts[f"{m}|{vname}"])
            print(f"[ablacion] {m:14s} {vname:22s} cell_bal_acc={mm['cell_balanced_acc']:.4f} "
                  f"recall_mem={mm['recall_memory']:.3f} recall_comp={mm['recall_compute']:.3f}", flush=True)
    gq.FEATURES[:] = actual
    boot = gq.bootstrap(counts, a.boot)
    report = {"features": variants, "n_rows": len(frame), "n_families": len(families), "ablacion": {}}
    for m in MODELS:
        base = boot[f"{m}|actual"]["cell_balanced_acc"]
        for vname in variants:
            x = boot[f"{m}|{vname}"]["cell_balanced_acc"]
            entry = {"cell_balanced_acc": round(float(gq.metrics_from_counts(counts[f"{m}|{vname}"])["cell_balanced_acc"]), 4),
                     "ci95": [round(v, 4) for v in gq.ci(x)]}
            if vname != "actual":
                d = x - base
                entry["delta_vs_actual"] = {"mean": round(float(d.mean()), 4), "ci95": [round(v, 4) for v in gq.ci(d)]}
            report["ablacion"][f"{m}|{vname}"] = entry
    print("\n=== ablacion (exactitud balanceada por celda, LOFO) ===")
    for k, v in report["ablacion"].items():
        print(f"  {k:40s} {v['cell_balanced_acc']:.4f}  IC95 {v['ci95']}  " + (f"delta {v['delta_vs_actual']}" if 'delta_vs_actual' in v else ""))

    # 2) contrafactual con el modelo final (todas las filas), regresion logistica como el candidato exportado
    X = frame[actual].to_numpy(dtype=np.float32)
    y = frame["y"].to_numpy()
    model = gq.make_model("regresion_log", 2000)
    w = gq.cell_weights(fam_codes, y)
    model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    base_pred = model.predict_proba(X)[:, 1] >= 0.5
    ci_col = actual.index(CLOCK)
    cf = {}
    for target in (1410.0, 1260.0, 810.0):
        Xc = X.copy(); Xc[:, ci_col] = target
        pred = model.predict_proba(Xc)[:, 1] >= 0.5
        flipped = pred != base_pred
        cf[str(int(target))] = {
            "flip_rate": round(float(flipped.mean()), 4),
            "flip_rate_si_clase_real_compute": round(float(flipped[~y].mean()), 4),
            "flip_rate_si_clase_real_memory": round(float(flipped[y].mean()), 4),
            "compute_a_memory": int((flipped & ~base_pred).sum()), "memory_a_compute": int((flipped & base_pred).sum()),
        }
    report["contrafactual_reloj_forzado"] = cf
    print("\n=== contrafactual: reloj forzado, modelo final, todas las filas ===")
    print(json.dumps(cf, indent=1))
    clocks = frame[CLOCK].describe().round(1).to_dict()
    report["reloj_observado_en_datos"] = clocks
    print("reloj SM observado en los datos:", clocks)

    # 3) pesos estandarizados
    lr = model.steps[-1][1]
    weights = dict(zip(actual, [round(float(c), 3) for c in lr.coef_[0]]))
    report["pesos_estandarizados_regresion_log"] = weights
    print("\npesos estandarizados (positivo = memory_bound):", weights)
    (a.out / "gpu_clock_feature_audit.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
