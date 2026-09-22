"""Informe de calidad del clasificador GPU (compute/memory) por corrida,
sobre el dataset histórico fusionado (``merge_historical_gpu_phase_runs.py``,
contrato ``granularity=run``, etiqueta constante por ``ncu`` offline -- NO el
dataset nuevo por ventana CUPTI, que todavía no está postprocesado).

Traslada a GPU la misma "forma" que ``cpu_quality_report.py`` fijó para CPU:

- Métrica primaria: exactitud balanceada sobre celdas familia x clase (media
  del recall de cada celda observada). Bien definida en familias de una sola
  clase y no depende de la prevalencia ni del volumen de cada familia.
- LOFO por familia (``kernel_family``, ya viene resuelta en el CSV fusionado),
  ponderación por celda (nunca ``class_weight="balanced"`` de sklearn, que
  pondera por clase global, no por familia x clase).
- Comparación de los 7 modelos candidatos del plan (mismos nombres que
  ``model_specs.py``/CPU), 5 semillas, IC95 por bootstrap de familias.

Primera pasada (núcleo): sin selección de variables, sin umbral de
abstención, sin tabla de política -- eso queda para una siguiente ronda,
igual que en CPU se hizo por etapas.

Clase positiva: ``memory_bound``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

FEATURES = ["gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median"]
# A6 (estrategia sin cluster): el dataset ya trae estos agregados por señal
# (calculados por gpu_phases.py, nunca usados como feature) -- el mecanismo
# de kmeans (lanzamientos cortos y repetidos que la mediana diluye a "poco
# ocupado") deberia ser visible en la variabilidad, no en el centro.
VARIABILITY_FEATURES = ["gpu_util_pct_std", "gpu_mem_util_pct_std", "gpu_power_mw_std", "gpu_sm_clock_mhz_std",
                          "gpu_util_pct_n_distinct"]
LABEL = "phase_label_train"
FAMILY_COL = "kernel_family"
# Mismo motivo de fuga que train_phase_gpu.py: la etiqueta se deriva de estas.
FORBIDDEN = {"operational_intensity", "i_ridge_used", "phase_label_hint", "roofline_calibration_ref"}
MODEL_NAMES = ["mayoritaria", "arbol_prof1", "regresion_log", "arbol_prof6", "random_forest", "extra_trees", "xgboost"]
CANDIDATE = "xgboost"


# ----------------------------------------------------------------- datos
# A2 del plan: "rajaperf_cuda" (protocol.derive_kernel_family, compartido con
# CPU y documentado como decision deliberada -- no se toca) agrupa 3 kernels
# CUDA distintos (gemm/heat_3d/jacobi_2d) en una sola familia LOFO. Este
# override es LOCAL a este script de diagnostico GPU, no cambia el protocolo
# compartido ni ningun dataset de produccion.
_RAJAPERF_CUDA_SPLIT = {
    "gpu_rajaperf_gemm": "rajaperf_cuda_gemm",
    "gpu_rajaperf_heat_3d": "rajaperf_cuda_heat_3d",
    "gpu_rajaperf_jacobi_2d": "rajaperf_cuda_jacobi_2d",
}


def load(path: str, split_rajaperf_cuda: bool = False, catalog_path: str | None = None,
         add_variability: bool = False, extra_features: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    leak = set(FEATURES) & FORBIDDEN
    assert not leak, leak
    frame = frame[frame[LABEL].isin(["compute_bound", "memory_bound"])].copy()
    frame["y"] = frame[LABEL].eq("memory_bound").to_numpy()
    if split_rajaperf_cuda:
        override = frame["kernel_ref"].map(_RAJAPERF_CUDA_SPLIT)
        frame[FAMILY_COL] = override.fillna(frame[FAMILY_COL])
    if add_variability:
        for f in VARIABILITY_FEATURES:
            if f not in FEATURES:
                FEATURES.append(f)
    if extra_features:
        for f in extra_features:
            if f not in FEATURES:
                FEATURES.append(f)
    if catalog_path is not None:
        # A5: gpu_precision no se propaga hoy en training_gpu_phases.csv --
        # se lee directo del catalogo por kernel_ref, sin tocar el postproceso.
        from common.hpc.catalog import load_catalog

        catalog = load_catalog(catalog_path)
        missing = sorted(set(frame["kernel_ref"]) - set(catalog))
        if missing:
            raise ValueError(f"kernel_ref sin entrada en el catalogo: {missing}")
        precision = frame["kernel_ref"].map(lambda k: catalog[k].gpu_precision)
        unknown = sorted(set(precision.dropna().unique()) - {"fp32", "fp64"})
        if unknown:
            raise ValueError(f"gpu_precision con valores inesperados: {unknown}")
        frame["gpu_precision_fp64"] = precision.eq("fp64").astype(float)
        if "gpu_precision_fp64" not in FEATURES:
            FEATURES.append("gpu_precision_fp64")
    n0 = len(frame)
    frame = frame.dropna(subset=FEATURES).reset_index(drop=True)
    frame.attrs["dropped_nan_rows"] = n0 - len(frame)
    return frame


def cell_weights(fam_codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    key = fam_codes.astype(np.int64) * 2 + y.astype(np.int64)
    counts = np.bincount(key)
    w = 1.0 / counts[key]
    return w / w.mean()


# --------------------------------------------------------------- modelos
def make_model(name: str, seed: int):
    if name == "mayoritaria":
        return "mayoritaria"
    if name == "arbol_prof1":
        return DecisionTreeClassifier(max_depth=1, random_state=seed)
    if name == "regresion_log":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "arbol_prof6":
        return DecisionTreeClassifier(max_depth=6, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(n_estimators=100, max_depth=6, n_jobs=1, random_state=seed, eval_metric="logloss", verbosity=0)
    raise ValueError(name)


def fit_predict(name_model: str, frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                 seed: int, fam_codes: np.ndarray) -> np.ndarray:
    y = frame["y"].to_numpy()
    if name_model == "mayoritaria":
        prior = y[train_idx].mean()
        return np.full(len(test_idx), 1.0 if prior >= 0.5 else 0.0)
    X = frame[FEATURES].to_numpy(dtype=np.float32)
    model = make_model(name_model, seed)
    w = cell_weights(fam_codes[train_idx], y[train_idx])
    if hasattr(model, "steps"):
        model.fit(X[train_idx], y[train_idx], **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X[train_idx], y[train_idx], sample_weight=w)
    return model.predict_proba(X[test_idx])[:, 1]


# -------------------------------------------------------------- metricas
def counts_from(fam: np.ndarray, y: np.ndarray, pred: np.ndarray, families: list[str]) -> np.ndarray:
    out = np.zeros((len(families), 2, 2), dtype=np.int64)
    index = {f: i for i, f in enumerate(families)}
    fi = np.fromiter((index[f] for f in fam), dtype=np.int64, count=len(fam))
    np.add.at(out, (fi, y.astype(np.int64), pred.astype(np.int64)), 1)
    return out


def metrics_from_counts(c: np.ndarray) -> dict[str, float]:
    tot = c.sum(axis=2)
    correct = np.stack([c[:, 0, 0], c[:, 1, 1]], axis=1)
    cell_recall = np.where(tot > 0, correct / np.maximum(tot, 1), np.nan)
    pooled = c.sum(axis=0)
    tn, fp, fn, tp = pooled[0, 0], pooled[0, 1], pooled[1, 0], pooled[1, 1]

    def f1(tp_, fp_, fn_):
        d = 2 * tp_ + fp_ + fn_
        return 2 * tp_ / d if d else 0.0
    f1_mem = f1(tp, fp, fn)
    f1_com = f1(tn, fn, fp)
    fam_acc = correct.sum(axis=1) / np.maximum(tot.sum(axis=1), 1)
    return {
        "cell_balanced_acc": float(np.nanmean(cell_recall)),
        "recall_compute": float(tn / max(tn + fp, 1)),
        "recall_memory": float(tp / max(tp + fn, 1)),
        "pooled_f1_macro": float((f1_mem + f1_com) / 2),
        "pooled_accuracy": float((tn + tp) / max(pooled.sum(), 1)),
        "mean_family_accuracy": float(fam_acc.mean()),
    }


def bootstrap(counts_by_cfg: dict[str, np.ndarray], reps: int, seed: int = 0) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    F = next(iter(counts_by_cfg.values())).shape[0]
    draws = rng.integers(0, F, size=(reps, F))
    keys = ["cell_balanced_acc", "pooled_f1_macro", "mean_family_accuracy", "recall_compute", "recall_memory"]
    samples = {c: {k: np.empty(reps) for k in keys} for c in counts_by_cfg}
    for r, d in enumerate(draws):
        for c, cnt in counts_by_cfg.items():
            m = metrics_from_counts(cnt[d])
            for k in keys:
                samples[c][k][r] = m[k]
    return samples


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ------------------------------------------------------------ evaluacion
def lofo(frame: pd.DataFrame, families: list[str], fam_codes: np.ndarray, cfg: str, seed: int) -> np.ndarray:
    fam_arr = frame[FAMILY_COL].to_numpy()
    y = frame["y"].to_numpy()
    counts = np.zeros((len(families), 2, 2), dtype=np.int64)
    for i, f in enumerate(families):
        train = np.flatnonzero(fam_arr != f)
        test = np.flatnonzero(fam_arr == f)
        p = fit_predict(cfg, frame, train, test, seed, fam_codes)
        pred = p >= 0.5
        counts[i] = counts_from(fam_arr[test], y[test], pred, [f])[0]
    return counts


def stage_matrix(frame: pd.DataFrame, families: list[str], fam_codes: np.ndarray, args, out: Path) -> None:
    seeds = list(range(args.seeds))
    cnt = {cfg: [] for cfg in MODEL_NAMES}
    for s in seeds:
        for cfg in MODEL_NAMES:
            t0 = time.time()
            cnt[cfg].append(lofo(frame, families, fam_codes, cfg, 2000 + s))
            print(f"[matrix] seed={s} {cfg} {time.time() - t0:.1f}s", flush=True)
    summary = {}
    for cfg in MODEL_NAMES:
        per_seed = [metrics_from_counts(c) for c in cnt[cfg]]
        mean_counts = np.mean(np.stack(cnt[cfg]), axis=0)
        summary[cfg] = {
            "mean_over_seeds": {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]},
            "sd_over_seeds": {k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in per_seed[0]},
            "counts_mean": mean_counts.tolist(),
        }
    counts_by = {c: np.array(summary[c]["counts_mean"]) for c in MODEL_NAMES}
    boot = bootstrap(counts_by, args.boot)
    for c in MODEL_NAMES:
        summary[c]["ci95_family_bootstrap"] = {k: ci(v) for k, v in boot[c].items()}
    for c in MODEL_NAMES:
        if c == CANDIDATE:
            continue
        d = boot[c]["cell_balanced_acc"] - boot[CANDIDATE]["cell_balanced_acc"]
        summary[c][f"delta_vs_{CANDIDATE}"] = {"mean": float(d.mean()), "ci95": ci(d), "prob_positive": float((d > 0).mean())}
    (out / "matrix.json").write_text(json.dumps({"families": families, "seeds": seeds, "n_rows": len(frame), "summary": summary}, indent=1))
    rows = []
    for cfg in MODEL_NAMES:
        m = summary[cfg]["mean_over_seeds"]
        s = summary[cfg]["sd_over_seeds"]
        c = summary[cfg]["ci95_family_bootstrap"]["cell_balanced_acc"]
        rows.append({"model": cfg, **{k: round(v, 4) for k, v in m.items()},
                     "sd_seed_cell_bal": round(s["cell_balanced_acc"], 4), "ci_lo": round(c[0], 4), "ci_hi": round(c[1], 4)})
    pd.DataFrame(rows).to_csv(out / "matrix.csv", index=False)
    fam_rows = []
    cm = np.array(summary[CANDIDATE]["counts_mean"])
    for i, f in enumerate(families):
        tot = cm[i].sum(axis=1)
        fam_rows.append({"family": f, "n": int(round(tot.sum())), "memory_share": round(float(tot[1] / max(tot.sum(), 1e-9)), 4),
                          "recall_compute": round(float(cm[i, 0, 0] / tot[0]), 4) if tot[0] else None,
                          "recall_memory": round(float(cm[i, 1, 1] / tot[1]), 4) if tot[1] else None,
                          "accuracy": round(float((cm[i, 0, 0] + cm[i, 1, 1]) / tot.sum()), 4)})
    pd.DataFrame(fam_rows).to_csv(out / f"{CANDIDATE}_by_family.csv", index=False)
    print(f"[matrix] {len(families)} familias, {len(frame)} corridas", flush=True)


def stage_inventory(frame: pd.DataFrame, families: list[str], out: Path) -> None:
    rows = []
    for f in families:
        sub = frame[frame[FAMILY_COL] == f]
        rows.append({"family": f, "n": len(sub), "memory_share": round(float(sub["y"].mean()), 4),
                      "single_class": bool(sub["y"].nunique() == 1)})
    pd.DataFrame(rows).to_csv(out / "inventory.csv", index=False)
    (out / "inventory.json").write_text(json.dumps({
        "n_rows": len(frame), "n_families": len(families),
        "dropped_nan_rows": int(frame.attrs.get("dropped_nan_rows", 0)),
        "single_class_families": sorted(f for f in families if frame.loc[frame[FAMILY_COL] == f, "y"].nunique() == 1),
    }, indent=1))


THRESHOLD_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98]
MIN_CELL_COVERAGE = 0.60


def stage_threshold(frame: pd.DataFrame, families: list[str], fam_codes: np.ndarray, args, out: Path) -> None:
    """Umbral de abstencion, mismo criterio que cpu_quality_report.py::stage_threshold_final:
    exactitud balanceada por celda sobre lo decidido, con cobertura media por celda >= minimo.
    """
    fam_arr = frame[FAMILY_COL].to_numpy()
    y_all = frame["y"].to_numpy()
    per_seed = []
    for s in range(args.seeds):
        idx_parts, p_parts = [], []
        for f in families:
            train = np.flatnonzero(fam_arr != f)
            test = np.flatnonzero(fam_arr == f)
            p = fit_predict(CANDIDATE, frame, train, test, 2000 + s, fam_codes)
            idx_parts.append(test)
            p_parts.append(p)
        idx = np.concatenate(idx_parts)
        p = np.concatenate(p_parts)
        y, fam = y_all[idx], fam_arr[idx]
        conf = np.maximum(p, 1 - p)
        rows = []
        for thr in THRESHOLD_GRID:
            keep = conf >= thr
            recalls, covs = [], []
            for f in families:
                for c in (0, 1):
                    cell = (fam == f) & (y == c)
                    if not cell.any():
                        continue
                    covs.append(keep[cell].mean())
                    dec = cell & keep
                    if dec.any():
                        recalls.append(float(((p[dec] >= 0.5) == c).mean()))
            rows.append({"threshold": thr, "cell_balanced_acc": float(np.mean(recalls)) if recalls else float("nan"),
                         "cell_coverage": float(np.mean(covs)) if covs else 0.0})
        per_seed.append(pd.DataFrame(rows))
        print(f"[threshold] seed={s}", flush=True)
    grid = sum(per_seed) / len(per_seed)
    ok = grid[grid["cell_coverage"] >= MIN_CELL_COVERAGE]
    if ok.empty:
        print(f"[threshold] ningun umbral alcanza cobertura minima {MIN_CELL_COVERAGE}; se reporta la rejilla sin elegir", flush=True)
        chosen = None
    else:
        best = ok.sort_values(["cell_balanced_acc", "cell_coverage", "threshold"], ascending=[False, False, True]).iloc[0]
        chosen = {"threshold": float(best["threshold"]), "cell_balanced_acc": float(best["cell_balanced_acc"]),
                  "cell_coverage": float(best["cell_coverage"])}
    grid.round(4).to_csv(out / "threshold_grid.csv", index=False)
    (out / "threshold.json").write_text(json.dumps({
        "candidate_model": CANDIDATE,
        "criterion": "exactitud balanceada por celda sobre lo decidido, cobertura media por celda >= minimo",
        "min_cell_coverage": MIN_CELL_COVERAGE, "seeds": args.seeds, "chosen": chosen,
    }, indent=1))
    print(grid.round(4).to_string(index=False), flush=True)
    print("[threshold] elegido", chosen, flush=True)


MIN_CELL_N = 10


def stage_cells(frame: pd.DataFrame, families: list[str], out: Path) -> None:
    """A3: igual que la seccion 'Limite' del informe CPU -- cuantas celdas
    familia x clase tienen menos de MIN_CELL_N corridas y cuanto pesan en la
    metrica principal, sin excluirlas del resultado (mismo criterio que CPU:
    no seleccionar datos por resultado).
    """
    matrix_path = out / "matrix.json"
    if not matrix_path.exists():
        raise SystemExit("falta matrix.json: correr la etapa matrix antes de cells")
    summary = json.loads(matrix_path.read_text())["summary"]
    cm = np.array(summary[CANDIDATE]["counts_mean"])
    rows = []
    for i, f in enumerate(families):
        for c, cls in enumerate(("compute_bound", "memory_bound")):
            n = int(round(cm[i, c].sum()))
            if n == 0:
                continue
            recall = float(cm[i, c, c] / max(n, 1))
            rows.append({"family": f, "class": cls, "n": n, "recall": round(recall, 4), "tiny": n < MIN_CELL_N})
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "cells.csv", index=False)
    tiny = cells[cells["tiny"]]
    full_mean = float(cells["recall"].mean())
    excl_mean = float(cells.loc[~cells["tiny"], "recall"].mean()) if (~cells["tiny"]).any() else float("nan")
    report = {
        "candidate_model": CANDIDATE, "min_cell_n": MIN_CELL_N,
        "n_cells_total": len(cells), "n_cells_tiny": int(tiny.shape[0]),
        "tiny_cells": tiny[["family", "class", "n", "recall"]].to_dict("records"),
        "cell_balanced_acc_all_cells": round(full_mean, 4),
        "cell_balanced_acc_excluding_tiny": round(excl_mean, 4),
        "weight_of_tiny_cells_pp": round((full_mean - excl_mean) * 100, 2) if not np.isnan(excl_mean) else None,
    }
    (out / "cells.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


STAGES = ["inventory", "matrix", "threshold", "cells"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--split-rajaperf-cuda", action="store_true",
                     help="A2: separa rajaperf_cuda en sus 3 kernels reales (gemm/heat_3d/jacobi_2d), solo en este script")
    ap.add_argument("--catalog-path", default=None,
                     help="A5: si se da, agrega gpu_precision_fp64 (leido del catalogo por kernel_ref) como 5a feature")
    ap.add_argument("--add-variability", action="store_true",
                     help="A6: agrega _std/_n_distinct (variabilidad intra-corrida) ya presentes en el CSV, nunca usadas")
    ap.add_argument("--extra-features", nargs="*", default=None,
                     help="ablacion A6: agrega columnas especificas del CSV en vez del grupo completo de --add-variability")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = load(args.source, split_rajaperf_cuda=args.split_rajaperf_cuda, catalog_path=args.catalog_path,
                 add_variability=args.add_variability, extra_features=args.extra_features)
    print(f"features: {FEATURES}", flush=True)
    families = sorted(frame[FAMILY_COL].unique())
    fam_codes = pd.Categorical(frame[FAMILY_COL], categories=families).codes
    for st in args.stages.split(","):
        t0 = time.time()
        print(f"=== stage {st}", flush=True)
        if st == "inventory":
            stage_inventory(frame, families, out)
        elif st == "matrix":
            stage_matrix(frame, families, fam_codes, args, out)
        elif st == "threshold":
            stage_threshold(frame, families, fam_codes, args, out)
        elif st == "cells":
            stage_cells(frame, families, out)
        else:
            raise ValueError(f"etapa desconocida: {st}")
        print(f"=== stage {st} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
