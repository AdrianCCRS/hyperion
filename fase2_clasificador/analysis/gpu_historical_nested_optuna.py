"""Busqueda anidada de hiperparametros (Optuna/TPE) sobre el dataset GPU
historico por corrida, replicando exactamente el mecanismo de
``train_phase_gpu.py``/CPU (``fase2_clasificador/eval/hyperparam_search.py``,
compartido) pero aplicado a los 5 modelos ajustables sobre las 16 familias
del dataset +A1(relajado)+A2(rajaperf_cuda separado)+A6
(gpu_mem_util_pct_std), tal como quedo cerrado en la ronda de esta sesion
(2026-09-22, ver memoria de proyecto
project-gpu-quality-improvement-plan-20260922).

Una sola semilla (igual que los jobs 7355/7390 de CPU) -- el promedio de 5
semillas es para la matriz de comparacion FIJA de gpu_quality_report.py, no
para esta busqueda, que ya es cara por su anidamiento (pliegue externo x
pliegue interno x prueba de Optuna).

Nunca correr esto en la laptop local -- someter siempre a pacca (ver
memoria feedback-never-run-compute-locally).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis.gpu_quality_report import (
    FAMILY_COL, FEATURES, counts_from, load, metrics_from_counts,
)
from fase2_clasificador.eval import protocol
from fase2_clasificador.training import model_specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--split-rajaperf-cuda", action="store_true")
    ap.add_argument("--extra-features", nargs="*", default=None)
    args = ap.parse_args()

    frame = load(args.source, split_rajaperf_cuda=args.split_rajaperf_cuda, extra_features=args.extra_features)
    print(f"features: {FEATURES}", flush=True)
    families = sorted(frame[FAMILY_COL].unique())
    X_all = frame[FEATURES].to_numpy(dtype=np.float32)
    y_all = frame["y"].to_numpy()
    fam_arr = frame[FAMILY_COL].to_numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    tunable_names = list(model_specs.tunable_specs(args.seed, 1.0).keys())
    counts = {name: np.zeros((len(families), 2, 2), dtype=np.int64) for name in tunable_names}
    best_params_por_familia: dict[str, dict[str, dict]] = {name: {} for name in tunable_names}
    n_search_omitida = 0
    t_start = time.time()

    for i, familia in enumerate(families):
        train = np.flatnonzero(fam_arr != familia)
        test = np.flatnonzero(fam_arr == familia)
        df_train_outer = frame.iloc[train]
        X_train, y_train = X_all[train], y_all[train]
        X_test, y_test = X_all[test], y_all[test]

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        tunable = model_specs.tunable_specs(args.seed, scale_pos_weight)

        for name, (build_fn, space_fn) in tunable.items():
            t0 = time.time()
            can_search = (
                args.n_trials > 0
                and hyperparam_search_n_groups(df_train_outer) >= 2
            )
            if can_search:
                from fase2_clasificador.eval import hyperparam_search
                best_params, inner_score = hyperparam_search.search_best_params(
                    build_fn, space_fn, df_train_outer, X_train, y_train,
                    kernel_col=FAMILY_COL, seed=args.seed, n_trials=args.n_trials,
                    fold_fn=protocol.leave_one_kernel_out,
                )
            else:
                n_search_omitida += 1
                best_params, inner_score = dict(model_specs.FALLBACK_PARAMS.get(name, {})), float("nan")
            best_params_por_familia[name][familia] = {"params": best_params, "inner_score": inner_score}
            model = build_fn(**best_params)
            model.fit(X_train, y_train)
            pred = (model.predict_proba(X_test)[:, 1] >= 0.5)
            counts[name][i] = counts_from(fam_arr[test], y_test, pred, [familia])[0]
            print(f"[{i+1}/{len(families)}] {familia} {name}: {time.time()-t0:.1f}s "
                  f"inner_score={inner_score:.3f} params={best_params}", flush=True)

    summary = {name: metrics_from_counts(counts[name]) for name in tunable_names}
    report = {
        "schema": "gpu_historical_nested_optuna/1",
        "n_trials": args.n_trials, "seed": args.seed, "families": families,
        "n_search_omitida": n_search_omitida,
        "wall_time_seconds": round(time.time() - t_start, 1),
        "metrics_by_model": summary,
        "best_params_by_family": best_params_por_familia,
    }
    (args.out / "nested_optuna.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items()}, indent=1))
    print(f"TOTAL: {time.time()-t_start:.0f}s, omitidas={n_search_omitida}", flush=True)


def hyperparam_search_n_groups(df_train_outer: pd.DataFrame) -> int:
    from fase2_clasificador.eval import hyperparam_search
    return hyperparam_search.n_groups(df_train_outer, kernel_col=FAMILY_COL, fold_fn=protocol.leave_one_kernel_out)


if __name__ == "__main__":
    main()
