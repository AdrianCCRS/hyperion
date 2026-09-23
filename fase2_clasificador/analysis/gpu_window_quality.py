"""Reentrenamiento del clasificador de GPU por VENTANA con el dataset historico SIN CUPTI (Bloque D).

Problema que resuelve: el candidato vigente se entreno con variables de CORRIDA completa (mediana y desviacion sobre
toda la corrida, incluidos arranques e inactividad) y el daemon decide en linea sobre la ventana de una fase estable;
`gpu_mem_util_pct_std`, la variable de mayor peso, deja de significar lo mismo (medido: 0/5 y 1/5 de aciertos con
sondeo de 5 ms, 2/5 y 4/6 con 50 ms, jobs 7613/7614).

Construccion (sin CUPTI, sin nuevas mediciones): para cada una de las corridas del dataset historico
(`historical_gpu_relaxed020_20260922.csv`, etiqueta ncu estatica por corrida) se leen sus muestras NVML de 5 ms
(`samples.csv`, filas GPU) y se agrupan en ventanas de 120 ms sin solape (la ventana real de NVML). Por ventana:
util mediana, mem_util mediana, mem_util desviacion muestral -- las mismas 3 variables, sin reloj ni potencia. Se
conservan solo ventanas ACTIVAS (util mediana > 5%, el mismo umbral del daemon) con >= 10 muestras y cuyo reloj SM
observado esta a +-5% del reloj aplicado de la corrida (para que la ventana corresponda al nivel de frecuencia).
La etiqueta de cada ventana es la de su corrida.

Evaluacion (LOFO por familia, ponderacion por celda familia x clase, IC95 por bootstrap de familias):
  * por ventana: exactitud balanceada por celda.
  * por DECISION, como la toma el daemon: votacion por mayoria sobre las primeras K=25 ventanas activas de la
    corrida (~3 s de actividad sostenida); unidad = corrida.
  * BASE: el modelo actual (entrenado con variables de corrida) aplicado a esas mismas ventanas y votado igual.
    Es la reproduccion offline del fallo observado en el daemon.
Correr en pacca (los samples.csv estan alli); nunca en local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fase2_clasificador.analysis import gpu_quality_report as gq

WINDOW_NS = 120_000_000
MIN_SAMPLES = 10
ACTIVITY_PCT = 5.0
CLOCK_TOL = 0.05
VOTE_K = 25
TRAIN_CAP_PER_RUN = 60
FEATURES3 = ["gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_mem_util_pct_std"]
CAMPAIGN_DIRS = {
    "pacca_gpu_final_20260913": "gpu",
    "pacca_gpu_conv2d_extra_20260916": "gpu_conv2d_extra",
    "pacca_gpu_kmeans_extra_20260916": "gpu_kmeans_extra",
    "pacca_gpu_minibude_extra_20260916": "gpu_minibude_extra",
}


def run_windows(samples_csv: Path, applied_mhz: float | None) -> pd.DataFrame:
    df = pd.read_csv(samples_csv, usecols=["tag", "timestamp_ns", "gpu_util_pct", "gpu_mem_util_pct", "gpu_sm_clock_mhz"],
                     low_memory=False)
    g = df[df["tag"] == "GPU"].dropna(subset=["gpu_util_pct", "gpu_mem_util_pct"]).copy()
    if g.empty:
        return g
    g["w"] = (g["timestamp_ns"] - g["timestamp_ns"].min()) // WINDOW_NS
    agg = g.groupby("w").agg(
        n=("gpu_util_pct", "size"),
        gpu_util_pct_median=("gpu_util_pct", "median"),
        gpu_mem_util_pct_median=("gpu_mem_util_pct", "median"),
        gpu_mem_util_pct_std=("gpu_mem_util_pct", "std"),
        clock=("gpu_sm_clock_mhz", "median"),
    ).reset_index()
    agg = agg[(agg["n"] >= MIN_SAMPLES) & (agg["gpu_util_pct_median"] > ACTIVITY_PCT)].copy()
    agg["gpu_mem_util_pct_std"] = agg["gpu_mem_util_pct_std"].fillna(0.0)
    if applied_mhz and not np.isnan(applied_mhz):
        agg = agg[(agg["clock"] - applied_mhz).abs() <= CLOCK_TOL * applied_mhz]
    return agg.sort_values("w").reset_index(drop=True)


def build_dataset(source: str, campaigns_root: Path) -> pd.DataFrame:
    frame = gq.load(source, split_rajaperf_cuda=True, extra_features=["gpu_mem_util_pct_std"])
    parts, missing = [], 0
    for r in frame.itertuples():
        camp = r.run_id.split("__")[0]
        run_dir = campaigns_root / CAMPAIGN_DIRS[camp] / r.run_id
        if not (run_dir / "samples.csv").exists():
            missing += 1
            continue
        w = run_windows(run_dir / "samples.csv", getattr(r, "gpu_freq_mhz_applied", np.nan))
        if w.empty:
            continue
        w["run_id"] = r.run_id
        w["kernel_family"] = getattr(r, gq.FAMILY_COL)
        w["y"] = bool(r.y)
        w["gpu_freq_level_id"] = r.gpu_freq_level_id
        w["order"] = np.arange(len(w))  # posicion dentro de las ventanas activas de la corrida
        parts.append(w)
    print(f"corridas: {len(frame)}, sin samples.csv: {missing}, con ventanas activas: {len(parts)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def vote(pred_by_run: pd.DataFrame) -> pd.DataFrame:
    """Voto por mayoria sobre las primeras VOTE_K ventanas activas de cada corrida."""
    first = pred_by_run[pred_by_run["order"] < VOTE_K]
    out = first.groupby("run_id").agg(kernel_family=("kernel_family", "first"), y=("y", "first"),
                                      frac_mem=("pred", "mean"), n=("pred", "size")).reset_index()
    out["pred"] = out["frac_mem"] > 0.5
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="historical_gpu_relaxed020_20260922.csv (dataset por corrida)")
    ap.add_argument("--campaigns-root", type=Path, default=Path.home() / "hyperion-results/final/campaigns")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--models", nargs="*", default=["regresion_log", "random_forest", "extra_trees", "xgboost"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    win = build_dataset(a.source, a.campaigns_root)
    win.to_csv(a.out / "gpu_windows_120ms.csv", index=False)
    run_frame = gq.load(a.source, split_rajaperf_cuda=True, extra_features=["gpu_mem_util_pct_std"])
    families = sorted(win["kernel_family"].unique())
    print(f"ventanas activas: {len(win)}, familias: {len(families)}, corridas: {win['run_id'].nunique()}", flush=True)
    print(win.groupby("y").size().rename({True: "memory_bound", False: "compute_bound"}).to_string(), flush=True)

    fam_w = win["kernel_family"].to_numpy()
    y_w = win["y"].to_numpy()
    fam_codes_w = pd.Categorical(win["kernel_family"], categories=families).codes
    X_w = win[FEATURES3].to_numpy(dtype=np.float32)
    # submuestra de entrenamiento: hasta TRAIN_CAP_PER_RUN ventanas espaciadas por corrida (las corridas pesan parecido)
    keep = np.zeros(len(win), dtype=bool)
    for _, idx in win.groupby("run_id").indices.items():
        step = max(1, len(idx) // TRAIN_CAP_PER_RUN)
        keep[idx[::step][:TRAIN_CAP_PER_RUN]] = True

    # variables por corrida para el modelo BASE (el actual)
    run_frame = run_frame[run_frame["run_id"].isin(win["run_id"].unique())].reset_index(drop=True)
    run_fam = run_frame[gq.FAMILY_COL].to_numpy()
    run_y = run_frame["y"].to_numpy()
    run_fam_codes = pd.Categorical(run_frame[gq.FAMILY_COL], categories=families).codes
    X_run = run_frame[FEATURES3].to_numpy(dtype=np.float32)

    counts_window, counts_vote = {}, {}
    for m in a.models + ["BASE_random_forest_por_corrida"]:
        per_w, per_v = [], []
        for s in range(a.seeds):
            cw = np.zeros((len(families), 2, 2), dtype=np.int64)
            cv = np.zeros((len(families), 2, 2), dtype=np.int64)
            for i, f in enumerate(families):
                test = np.flatnonzero(fam_w == f)
                if m.startswith("BASE"):
                    tr = np.flatnonzero(run_fam != f)
                    model = gq.make_model("random_forest", 2000 + s)
                    model.fit(X_run[tr], run_y[tr], sample_weight=gq.cell_weights(run_fam_codes[tr], run_y[tr]))
                else:
                    tr = np.flatnonzero((fam_w != f) & keep)
                    model = gq.make_model(m, 2000 + s)
                    w = gq.cell_weights(fam_codes_w[tr], y_w[tr])
                    if hasattr(model, "steps"):
                        model.fit(X_w[tr], y_w[tr], **{f"{model.steps[-1][0]}__sample_weight": w})
                    else:
                        model.fit(X_w[tr], y_w[tr], sample_weight=w)
                proba = model.predict_proba(X_w[test])[:, 1]
                pred = proba > 0.5
                cw[i] = gq.counts_from(fam_w[test], y_w[test], pred, [f])[0]
                t = win.iloc[test][["run_id", "kernel_family", "y", "order"]].copy()
                t["pred"] = pred
                v = vote(t)
                cv[i] = gq.counts_from(v["kernel_family"].to_numpy(), v["y"].to_numpy(), v["pred"].to_numpy(), [f])[0]
            per_w.append(cw); per_v.append(cv)
            mw = gq.metrics_from_counts(cw)["cell_balanced_acc"]; mv = gq.metrics_from_counts(cv)["cell_balanced_acc"]
            print(f"[{m:32s}] semilla {s}: ventana={mw:.4f} votacion={mv:.4f}", flush=True)
        counts_window[m] = np.mean(np.stack(per_w), axis=0)
        counts_vote[m] = np.mean(np.stack(per_v), axis=0)

    report = {"n_windows": int(len(win)), "n_runs": int(win["run_id"].nunique()), "families": families,
              "params": {"window_ns": WINDOW_NS, "vote_k": VOTE_K, "activity_pct": ACTIVITY_PCT, "clock_tol": CLOCK_TOL,
                         "train_cap_per_run": TRAIN_CAP_PER_RUN, "features": FEATURES3, "seeds": a.seeds},
              "resultados": {}}
    for label, counts in (("ventana", counts_window), ("votacion_por_corrida", counts_vote)):
        boot = gq.bootstrap(counts, a.boot)
        base = boot["BASE_random_forest_por_corrida"]["cell_balanced_acc"]
        print(f"\n=== exactitud balanceada por celda ({label}, LOFO) ===")
        report["resultados"][label] = {}
        for m, c in counts.items():
            met = gq.metrics_from_counts(c)
            x = boot[m]["cell_balanced_acc"]
            e = {"cell_balanced_acc": round(met["cell_balanced_acc"], 4), "ci95": [round(v, 4) for v in gq.ci(x)],
                 "recall_memory": round(met["recall_memory"], 3), "recall_compute": round(met["recall_compute"], 3)}
            if m != "BASE_random_forest_por_corrida":
                d = x - base
                e["delta_vs_base"] = {"mean": round(float(d.mean()), 4), "ci95": [round(v, 4) for v in gq.ci(d)]}
            report["resultados"][label][m] = e
            print(f"  {m:32s} {e['cell_balanced_acc']:.4f} IC95 {e['ci95']} mem={e['recall_memory']} comp={e['recall_compute']} "
                  + (f"delta_vs_base {e['delta_vs_base']}" if "delta_vs_base" in e else ""), flush=True)
    (a.out / "gpu_window_quality.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
