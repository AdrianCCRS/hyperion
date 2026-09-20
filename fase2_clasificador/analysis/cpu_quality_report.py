"""Informe de calidad del clasificador CPU (compute/memory) por intervalo.

Bateria de pruebas reproducible sobre ``feature_contract_source.csv`` (1.17 M
intervalos elegibles, 24 familias). Todas las pruebas retienen una familia
completa (LOFO) y ninguna elige nada mirando la familia de prueba, salvo las
etiquetadas explicitamente como "exploratorias" (rejilla completa reportada).

Metrica primaria: exactitud balanceada sobre celdas familia x clase (media del
recall de cada celda observada). Esta bien definida en familias de una sola
clase (donde el F1 macro por familia esta acotado a 0.5) y no depende de la
prevalencia ni del volumen de cada familia. Se informa tambien F1 macro
agrupado (OOF) para poder compararlo con el resto del proyecto.

Clase positiva: ``memory_bound``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from fase2_clasificador.analysis.cpu_feature_strategies import (
    FORBIDDEN_FEATURES, add_online_physical_features, production_variants,
)
from fase2_clasificador.eval import protocol

LEGACY = ["ipc", "mpki", "cache_miss_rate", "stall_mem_ratio", "ips", "running_ratio", "freq_khz_observed"]
VARIANTS = production_variants()
BASE = VARIANTS["baseline_pmu"]
INTER = VARIANTS["pmu_interactions"]
NOFREQ = VARIANTS["without_frequency"]
ALL_FEATURES = sorted(set(INTER) | set(LEGACY))
# Restricciones monotonas fijadas a priori (hipotesis fisica, no ajustadas):
# mas fallos/stalls de memoria -> mas probable memory_bound (+1).
MONO = {"mpki": 1, "cache_miss_rate": 1, "stall_mem_ratio": 1, "log1p_mpki": 1,
        "cache_misses_per_cycle": 1, "stalls_mem_per_ki": 1}


# ----------------------------------------------------------------- datos
def load(path: str) -> pd.DataFrame:
    frame = add_online_physical_features(pd.read_csv(path))
    frame["family"] = frame["kernel_ref"].map(protocol.derive_kernel_family)
    frame = frame[frame["phase_label_train"].isin(["compute_bound", "memory_bound"])].copy()
    frame["y"] = frame["phase_label_train"].eq("memory_bound").to_numpy()
    oi = frame["operational_intensity_uncore_real"].clip(lower=1e-3)
    frame["margin_log2"] = np.log2(oi / frame["i_ridge_used"])
    n0 = len(frame)
    frame = frame.dropna(subset=ALL_FEATURES).reset_index(drop=True)
    frame.attrs["dropped_nan_rows"] = n0 - len(frame)
    return frame


def capped_sample(frame: pd.DataFrame, cap: int, seed: int) -> np.ndarray:
    """Indices de un muestreo aleatorio con maximo ``cap`` filas por familia x clase."""
    rng = np.random.default_rng(seed)
    order = rng.random(len(frame))
    key = frame["family"].astype(str) + "|" + frame["y"].astype(str)
    rank = pd.Series(order).groupby(key.to_numpy()).rank(method="first").to_numpy()
    return np.flatnonzero(rank <= cap)


def cell_weights(fam_codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    key = fam_codes.astype(np.int64) * 2 + y.astype(np.int64)
    counts = np.bincount(key)
    w = 1.0 / counts[key]
    return w / w.mean()


# --------------------------------------------------------------- modelos
def make_model(name: str, seed: int, y_train: np.ndarray, features: list[str]):
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    xgb = dict(n_estimators=100, max_depth=6, n_jobs=1, random_state=seed, eval_metric="logloss", verbosity=0)
    if name == "majority":
        return "majority"
    if name == "stump":
        return DecisionTreeClassifier(max_depth=1, random_state=seed)
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "logistic_class_only":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=seed)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=seed)
    if name == "xgb":
        return XGBClassifier(scale_pos_weight=1.0, **xgb)
    if name == "xgb_class_only":
        return XGBClassifier(scale_pos_weight=neg / pos, **xgb)
    if name == "xgb_mono":
        constraint = tuple(MONO.get(f, 0) for f in features)
        return XGBClassifier(scale_pos_weight=1.0, monotone_constraints=constraint, **xgb)
    if name.startswith("xgb_d"):  # xgb_d{depth}_n{trees}
        d, n = name[5:].split("_n")
        p = dict(xgb, max_depth=int(d), n_estimators=int(n))
        return XGBClassifier(scale_pos_weight=1.0, **p)
    raise ValueError(name)


# name -> (model, features, weighting)
CONFIGS = {
    "majority": ("majority", BASE, "none"),
    "stump_base": ("stump", BASE, "cell"),
    "logistic_base": ("logistic", BASE, "cell"),
    "logistic_inter": ("logistic", INTER, "cell"),
    "xgb_base": ("xgb", BASE, "cell"),
    "xgb_inter": ("xgb", INTER, "cell"),          # candidato congelado
    "xgb_inter_nofreq": ("xgb", NOFREQ, "cell"),
    "xgb_freq_only": ("xgb_d2_n50", ["freq_khz_observed"], "cell"),
    "xgb_legacy_running_ratio": ("xgb", LEGACY, "cell"),
    "xgb_inter_class_only": ("xgb_class_only", INTER, "none"),
    "xgb_inter_mono": ("xgb_mono", INTER, "cell"),
    "xgb_inter_shallow": ("xgb_d2_n50", INTER, "cell"),
    "rf_inter": ("rf", INTER, "cell"),
    "et_inter": ("et", INTER, "cell"),
}
CANDIDATE = "xgb_inter"


def fit_predict(name_model: str, features: list[str], weighting: str, frame: pd.DataFrame,
                train_idx: np.ndarray, test_idx: np.ndarray, seed: int, fam_codes: np.ndarray) -> np.ndarray:
    y = frame["y"].to_numpy()
    if name_model == "majority":
        prior = y[train_idx].mean()
        return np.full(len(test_idx), 1.0 if prior >= 0.5 else 0.0)
    X = frame[features].to_numpy(dtype=np.float32)
    model = make_model(name_model, seed, y[train_idx], features)
    w = cell_weights(fam_codes[train_idx], y[train_idx]) if weighting == "cell" else None
    if hasattr(model, "steps"):
        model.fit(X[train_idx], y[train_idx], **({f"{model.steps[-1][0]}__sample_weight": w} if w is not None else {}))
    else:
        model.fit(X[train_idx], y[train_idx], sample_weight=w)
    return model.predict_proba(X[test_idx])[:, 1]


# -------------------------------------------------------------- metricas
def counts_from(fam: np.ndarray, y: np.ndarray, pred: np.ndarray, families: list[str]) -> np.ndarray:
    """Array (F, 2 clase real, 2 clase predicha) de conteos."""
    out = np.zeros((len(families), 2, 2), dtype=np.int64)
    index = {f: i for i, f in enumerate(families)}
    fi = np.fromiter((index[f] for f in fam), dtype=np.int64, count=len(fam))
    np.add.at(out, (fi, y.astype(np.int64), pred.astype(np.int64)), 1)
    return out


def metrics_from_counts(c: np.ndarray) -> dict[str, float]:
    tot = c.sum(axis=2)  # (F, 2)
    correct = np.stack([c[:, 0, 0], c[:, 1, 1]], axis=1)
    cell_recall = np.where(tot > 0, correct / np.maximum(tot, 1), np.nan)
    pooled = c.sum(axis=0)
    tn, fp, fn, tp = pooled[0, 0], pooled[0, 1], pooled[1, 0], pooled[1, 1]

    def f1(tp_, fp_, fn_):
        d = 2 * tp_ + fp_ + fn_
        return 2 * tp_ / d if d else 0.0
    f1_mem = f1(tp, fp, fn)
    f1_com = f1(tn, fn, fp)
    fam_acc = (correct.sum(axis=1) / np.maximum(tot.sum(axis=1), 1))
    return {
        "cell_balanced_acc": float(np.nanmean(cell_recall)),
        "recall_compute": float(tn / max(tn + fp, 1)),
        "recall_memory": float(tp / max(tp + fn, 1)),
        "pooled_f1_macro": float((f1_mem + f1_com) / 2),
        "pooled_accuracy": float((tn + tp) / max(pooled.sum(), 1)),
        "mean_family_accuracy": float(fam_acc.mean()),
    }


def bootstrap(counts_by_cfg: dict[str, np.ndarray], reps: int, seed: int = 0) -> dict[str, dict]:
    """IC95 por bootstrap de conglomerados (familias), con remuestreo comun entre configs."""
    rng = np.random.default_rng(seed)
    F = next(iter(counts_by_cfg.values())).shape[0]
    draws = rng.integers(0, F, size=(reps, F))
    keys = ["cell_balanced_acc", "pooled_f1_macro", "mean_family_accuracy"]
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
def lofo(frame: pd.DataFrame, families: list[str], fam_codes: np.ndarray, cfg: str, sample: np.ndarray,
         seed: int, full_test: bool, n_jobs: int, return_proba: bool = False):
    name_model, features, weighting = CONFIGS[cfg]
    fam_arr = frame["family"].to_numpy()
    y = frame["y"].to_numpy()

    def one(f: str):
        train = sample[fam_arr[sample] != f]
        test = np.flatnonzero(fam_arr == f) if full_test else sample[fam_arr[sample] == f]
        return f, test, fit_predict(name_model, features, weighting, frame, train, test, seed, fam_codes)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(f) for f in families)
    counts = np.zeros((len(families), 2, 2), dtype=np.int64)
    probas = {}
    for i, (f, test, p) in enumerate(results):
        pred = p >= 0.5
        counts[i] = counts_from(fam_arr[test], y[test], pred, [f])[0]
        if return_proba:
            probas[f] = (test, p)
    return (counts, probas) if return_proba else counts


def stage_matrix(frame, families, fam_codes, args, out: Path):
    seeds = list(range(args.seeds))
    res: dict[str, dict] = {}
    cnt = {cfg: {"capped": [], "full": []} for cfg in CONFIGS}
    for s in seeds:
        sample = capped_sample(frame, args.cap, 1000 + s)
        for cfg in CONFIGS:
            t0 = time.time()
            for mode in ("capped", "full"):
                cnt[cfg][mode].append(lofo(frame, families, fam_codes, cfg, sample, 2000 + s, mode == "full", args.n_jobs))
            print(f"[matrix] seed={s} {cfg} {time.time() - t0:.1f}s", flush=True)
    summary = {}
    for cfg in CONFIGS:
        summary[cfg] = {}
        for mode in ("capped", "full"):
            per_seed = [metrics_from_counts(c) for c in cnt[cfg][mode]]
            mean_counts = np.mean(np.stack(cnt[cfg][mode]), axis=0)
            summary[cfg][mode] = {
                "mean_over_seeds": {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]},
                "sd_over_seeds": {k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in per_seed[0]},
                "counts_mean": mean_counts.tolist(),
            }
    # IC por bootstrap de familias sobre el conteo medio entre semillas (test completo)
    for mode in ("capped", "full"):
        counts_by = {c: np.array(summary[c][mode]["counts_mean"]) for c in CONFIGS}
        boot = bootstrap(counts_by, args.boot)
        for c in CONFIGS:
            summary[c][mode]["ci95_family_bootstrap"] = {k: ci(v) for k, v in boot[c].items()}
        # diferencias pareadas frente al candidato y frente a xgb_base
        for ref in (CANDIDATE, "xgb_base"):
            for c in CONFIGS:
                if c == ref:
                    continue
                d = boot[c]["cell_balanced_acc"] - boot[ref]["cell_balanced_acc"]
                summary[c][mode].setdefault(f"delta_vs_{ref}", {"mean": float(d.mean()), "ci95": ci(d),
                                                                 "prob_positive": float((d > 0).mean())})
    (out / "matrix.json").write_text(json.dumps({"families": families, "seeds": seeds, "cap": args.cap, "summary": summary}, indent=1))
    rows = []
    for cfg in CONFIGS:
        for mode in ("capped", "full"):
            m = summary[cfg][mode]["mean_over_seeds"]
            s = summary[cfg][mode]["sd_over_seeds"]
            c = summary[cfg][mode]["ci95_family_bootstrap"]["cell_balanced_acc"]
            rows.append({"config": cfg, "test": mode, **{k: round(v, 4) for k, v in m.items()},
                         "sd_seed_cell_bal": round(s["cell_balanced_acc"], 4), "ci_lo": round(c[0], 4), "ci_hi": round(c[1], 4)})
    pd.DataFrame(rows).to_csv(out / "matrix.csv", index=False)
    # tabla por familia del candidato (media de semillas)
    fam_rows = []
    cm = np.array(summary[CANDIDATE]["full"]["counts_mean"])
    for i, f in enumerate(families):
        tot = cm[i].sum(axis=1)
        fam_rows.append({"family": f, "n_full": int(round(tot.sum())), "memory_share": round(float(tot[1] / max(tot.sum(), 1e-9)), 4),
                         "recall_compute": round(float(cm[i, 0, 0] / tot[0]), 4) if tot[0] else None,
                         "recall_memory": round(float(cm[i, 1, 1] / tot[1]), 4) if tot[1] else None,
                         "accuracy": round(float((cm[i, 0, 0] + cm[i, 1, 1]) / tot.sum()), 4)})
    pd.DataFrame(fam_rows).to_csv(out / "candidate_by_family_full.csv", index=False)


def stage_diagnostics(frame, families, fam_codes, args, out: Path):
    """Calibracion, margen al ridge, riesgo-cobertura y por frecuencia del candidato (semilla 0, test completo)."""
    sample = capped_sample(frame, args.cap, 1000)
    counts, probas = lofo(frame, families, fam_codes, CANDIDATE, sample, 2000, True, args.n_jobs, return_proba=True)
    idx = np.concatenate([probas[f][0] for f in families])
    p = np.concatenate([probas[f][1] for f in families])
    sub = frame.iloc[idx].reset_index(drop=True)
    sub["p_mem"] = p
    sub["pred"] = p >= 0.5
    sub["conf"] = np.maximum(p, 1 - p)
    sub["correct"] = sub["pred"] == sub["y"]
    diag: dict = {}
    # 1) error segun distancia al ridge
    bins = [-np.inf, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, np.inf]
    sub["margin_bin"] = pd.cut(sub["margin_log2"], bins)
    g = sub.groupby("margin_bin", observed=True)
    mt = pd.DataFrame({"n": g.size(), "accuracy": g["correct"].mean(), "share_memory": g["y"].mean(), "mean_conf": g["conf"].mean()})
    mt["share_of_all"] = mt["n"] / len(sub)
    mt.to_csv(out / "error_by_ridge_margin.csv")
    near = sub["margin_log2"].abs() < 1  # dentro de un factor 2 del ridge
    diag["near_ridge_within_x2"] = {"share_of_intervals": float(near.mean()), "accuracy_near": float(sub.loc[near, "correct"].mean()),
                                    "accuracy_far": float(sub.loc[~near, "correct"].mean()),
                                    "share_of_errors_near": float((~sub.loc[near, "correct"]).sum() / max((~sub["correct"]).sum(), 1))}
    # 2) calibracion (ECE, Brier) sobre confianza de la clase predicha
    edges = np.linspace(0.5, 1.0, 11)
    sub["conf_bin"] = pd.cut(sub["conf"], edges, include_lowest=True)
    cg = sub.groupby("conf_bin", observed=True)
    cal = pd.DataFrame({"n": cg.size(), "mean_conf": cg["conf"].mean(), "accuracy": cg["correct"].mean()})
    cal.to_csv(out / "calibration_bins.csv")
    ece = float((cal["n"] * (cal["mean_conf"] - cal["accuracy"]).abs()).sum() / cal["n"].sum())
    diag["calibration"] = {"ece": ece, "brier_memory": float(np.mean((sub["p_mem"] - sub["y"].astype(float)) ** 2)),
                           "mean_conf": float(sub["conf"].mean()), "accuracy": float(sub["correct"].mean())}
    # 3) riesgo-cobertura (umbral global sobre confianza), metricas ponderadas por celda familia x clase
    rows = []
    fam_arr = sub["family"].to_numpy()
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]:
        keep = sub["conf"].to_numpy() >= thr
        c = counts_from(fam_arr[keep], sub["y"].to_numpy()[keep], sub["pred"].to_numpy()[keep], families)
        m = metrics_from_counts(c)
        # cobertura por celda familia x clase (media de coberturas), independiente del volumen
        cov = sub.assign(k=keep).groupby(["family", "y"])["k"].mean()
        rows.append({"threshold": thr, "coverage_pooled": float(keep.mean()), "coverage_cell_mean": float(cov.mean()),
                     "coverage_min_cell": float(cov.min()), **{k: round(v, 4) for k, v in m.items()}})
    pd.DataFrame(rows).to_csv(out / "risk_coverage.csv", index=False)
    # abstencion aleatoria a la misma cobertura (control)
    rng = np.random.default_rng(0)
    ctrl = []
    for thr in [0.7, 0.85, 0.95]:
        keep = sub["conf"].to_numpy() >= thr
        rk = rng.random(len(sub)) < keep.mean()
        c = counts_from(fam_arr[rk], sub["y"].to_numpy()[rk], sub["pred"].to_numpy()[rk], families)
        ctrl.append({"threshold_equiv": thr, "coverage": float(rk.mean()), **{k: round(v, 4) for k, v in metrics_from_counts(c).items()}})
    pd.DataFrame(ctrl).to_csv(out / "random_abstention_control.csv", index=False)
    # 4) por nivel de frecuencia
    fr = []
    for lvl, part in sub.groupby("freq_level_id"):
        c = counts_from(part["family"].to_numpy(), part["y"].to_numpy(), part["pred"].to_numpy(), families)
        fr.append({"freq_level": lvl, "n": len(part), "share_memory": round(float(part["y"].mean()), 4), **{k: round(v, 4) for k, v in metrics_from_counts(c).items()}})
    pd.DataFrame(fr).to_csv(out / "by_frequency_level.csv", index=False)
    # 5) errores de alta confianza por familia
    hc = sub[sub["conf"] >= 0.9]
    tbl = hc.groupby("family").agg(n_high_conf=("correct", "size"), acc_high_conf=("correct", "mean"))
    tbl["share_of_family"] = tbl["n_high_conf"] / sub.groupby("family").size()
    tbl.to_csv(out / "high_confidence_by_family.csv")
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1))


def stage_protocols(frame, families, fam_codes, args, out: Path):
    """Aleatorio por intervalo vs. leave-one-kernel vs. leave-one-family (inflacion por fuga)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    res = {}
    for s in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s)
        sub = frame.iloc[sample].reset_index(drop=True)
        codes = fam_codes[sample]
        y = sub["y"].to_numpy()
        X = sub[INTER].to_numpy(dtype=np.float32)
        strat = pd.Series(sub["family"].astype(str) + y.astype(str)).factorize()[0]
        tr, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=s).split(X, strat))
        model = make_model("xgb", s, y[tr], INTER)
        model.fit(X[tr], y[tr], sample_weight=cell_weights(codes[tr], y[tr]))
        pred = model.predict_proba(X[te])[:, 1] >= 0.5
        cf = counts_from(sub["family"].to_numpy()[te], y[te], pred, families)
        res.setdefault("random_interval_split", []).append(metrics_from_counts(cf)["cell_balanced_acc"])
        # leave-one-kernel-out (kernel_ref): variantes de tamano de la misma familia quedan separadas
        kern = sub["kernel_ref"].to_numpy()
        kc = np.zeros((len(families), 2, 2), dtype=np.int64)
        for k in np.unique(kern):
            te_k = np.flatnonzero(kern == k)
            tr_k = np.flatnonzero(kern != k)
            if len(np.unique(y[tr_k])) < 2:
                continue
            m = make_model("xgb", s, y[tr_k], INTER)
            m.fit(X[tr_k], y[tr_k], sample_weight=cell_weights(codes[tr_k], y[tr_k]))
            pr = m.predict_proba(X[te_k])[:, 1] >= 0.5
            kc += counts_from(sub["family"].to_numpy()[te_k], y[te_k], pr, families)
        res.setdefault("leave_one_kernel_out", []).append(metrics_from_counts(kc)["cell_balanced_acc"])
        fc = lofo(frame, families, fam_codes, CANDIDATE, sample, 2000 + s, False, args.n_jobs)
        res.setdefault("leave_one_family_out", []).append(metrics_from_counts(fc)["cell_balanced_acc"])
        print(f"[protocols] seed={s}", {k: round(v[-1], 3) for k, v in res.items()}, flush=True)
    out_json = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "values": v} for k, v in res.items()}
    out_json["n_kernels"] = int(frame["kernel_ref"].nunique())
    (out / "protocols.json").write_text(json.dumps(out_json, indent=1))


def stage_learning_curve(frame, families, fam_codes, args, out: Path):
    """Metrica en familia retenida vs. numero de familias de entrenamiento."""
    sample = capped_sample(frame, args.cap, 1000)
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)
    ks = [3, 6, 9, 12, 15, 18, 21, 23]
    reps = args.lc_reps
    jobs = []
    rng = np.random.default_rng(7)
    for f in families:
        others = [g for g in families if g != f]
        for k in ks:
            for r in range(reps):
                jobs.append((f, k, r, list(rng.choice(others, size=k, replace=False))))

    def run(f, k, r, chosen):
        tr = sample[np.isin(fam_arr[sample], chosen)]
        if len(np.unique(y_all[tr])) < 2:
            return f, k, r, None
        te = sample[fam_arr[sample] == f]
        m = make_model("xgb", r, y_all[tr], INTER)
        m.fit(X[tr], y_all[tr], sample_weight=cell_weights(fam_codes[tr], y_all[tr]))
        pred = m.predict_proba(X[te])[:, 1] >= 0.5
        yt = y_all[te]
        cells = [np.mean(pred[yt == c] == c) for c in (0, 1) if (yt == c).any()]
        return f, k, r, float(np.mean(cells))

    rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame([r for r in rs if r[3] is not None], columns=["family", "k", "rep", "cell_balanced_acc"])
    df.to_csv(out / "learning_curve_raw.csv", index=False)
    agg = df.groupby("k")["cell_balanced_acc"].agg(["mean", "std", "count"])
    per_family = df.groupby(["k", "family"])["cell_balanced_acc"].mean().groupby("k")
    agg["sd_between_families"] = per_family.std()
    agg["ci95_half"] = 1.96 * agg["sd_between_families"] / np.sqrt(len(families))
    agg.to_csv(out / "learning_curve.csv")
    slope = np.polyfit(np.log(agg.index.to_numpy(dtype=float)), agg["mean"].to_numpy(), 1)
    (out / "learning_curve_fit.json").write_text(json.dumps({"slope_per_log_family": float(slope[0]), "intercept": float(slope[1])}))


def stage_nested_selective(frame, families, fam_codes, args, out: Path):
    """Umbral de abstencion elegido por LOFO interno con rejilla ampliada (0.50-0.98)."""
    grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98]
    sample = capped_sample(frame, args.cap, 1000)
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)

    def cellbal(y, pred, fam):
        cells = []
        for f in np.unique(fam):
            for c in (0, 1):
                mask = (fam == f) & (y == c)
                if mask.any():
                    cells.append(np.mean(pred[mask] == c))
        return float(np.mean(cells)) if cells else 0.0

    def outer(f):
        train_fams = [g for g in families if g != f]
        tr = sample[fam_arr[sample] != f]
        # LOFO interno
        oof_p, oof_y, oof_f = [], [], []
        for g in train_fams:
            a = tr[fam_arr[tr] != g]
            b = tr[fam_arr[tr] == g]
            m = make_model("xgb", 1, y_all[a], INTER)
            m.fit(X[a], y_all[a], sample_weight=cell_weights(fam_codes[a], y_all[a]))
            oof_p.append(m.predict_proba(X[b])[:, 1]); oof_y.append(y_all[b]); oof_f.append(fam_arr[b])
        p, yy, ff = np.concatenate(oof_p), np.concatenate(oof_y), np.concatenate(oof_f)
        conf = np.maximum(p, 1 - p)
        best, best_score = grid[0], -1
        for thr in grid:
            keep = conf >= thr
            if not keep.any():
                continue
            # cobertura media por celda >= 0.6
            cov = np.mean([keep[(ff == g) & (yy == c)].mean() for g in train_fams for c in (0, 1) if ((ff == g) & (yy == c)).any()])
            if cov < 0.6:
                continue
            score = cellbal(yy[keep], (p[keep] >= 0.5).astype(int), ff[keep])
            if score > best_score:
                best, best_score = thr, score
        m = make_model("xgb", 1, y_all[tr], INTER)
        m.fit(X[tr], y_all[tr], sample_weight=cell_weights(fam_codes[tr], y_all[tr]))
        te = np.flatnonzero(fam_arr == f)
        pt = m.predict_proba(X[te])[:, 1]
        return f, best, best_score, te, pt

    rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(outer)(f) for f in families)
    rows, kept_c, kept_y, kept_pred, kept_f = [], [], [], [], []
    for f, thr, sc, te, pt in rs:
        yt = y_all[te]
        conf = np.maximum(pt, 1 - pt)
        keep = conf >= thr
        rows.append({"family": f, "threshold": thr, "inner_score": round(sc, 4), "coverage_full": round(float(keep.mean()), 4),
                     "accuracy_auto": round(float(((pt >= 0.5) == yt)[keep].mean()), 4) if keep.any() else None})
        kept_f.append(np.full(keep.sum(), f)); kept_y.append(yt[keep]); kept_pred.append((pt >= 0.5)[keep])
    pd.DataFrame(rows).to_csv(out / "nested_selective_by_family.csv", index=False)
    fam_c, y_c, p_c = np.concatenate(kept_f), np.concatenate(kept_y), np.concatenate(kept_pred)
    m = metrics_from_counts(counts_from(fam_c, y_c, p_c, families))
    cov_cells = np.mean([r["coverage_full"] for r in rows])
    (out / "nested_selective.json").write_text(json.dumps({"grid": grid, "thresholds_chosen": {r["family"]: r["threshold"] for r in rows},
                                                           "metrics_on_covered": m, "mean_family_coverage": float(cov_cells),
                                                           "threshold_at_grid_max_count": int(sum(r["threshold"] == grid[-1] for r in rows))}, indent=1))


def stage_regularization(frame, families, fam_codes, args, out: Path):
    """Rejilla pequena de complejidad de XGBoost: LOFO exterior por configuracion y seleccion anidada."""
    grid = ["xgb_d2_n50", "xgb_d3_n100", "xgb_d4_n100", "xgb_d6_n100"]
    sample = capped_sample(frame, args.cap, 1000)
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)

    def cellbal(y, pred, fam):
        cells = [np.mean(pred[(fam == f) & (y == c)] == c) for f in np.unique(fam) for c in (0, 1) if ((fam == f) & (y == c)).any()]
        return float(np.mean(cells))

    def outer(f):
        tr = sample[fam_arr[sample] != f]
        te = np.flatnonzero(fam_arr == f)
        inner = {}
        for name in grid:
            preds, ys, fs = [], [], []
            for g in [g for g in families if g != f]:
                a = tr[fam_arr[tr] != g]; b = tr[fam_arr[tr] == g]
                m = make_model(name, 1, y_all[a], INTER)
                m.fit(X[a], y_all[a], sample_weight=cell_weights(fam_codes[a], y_all[a]))
                preds.append((m.predict_proba(X[b])[:, 1] >= 0.5).astype(int)); ys.append(y_all[b]); fs.append(fam_arr[b])
            inner[name] = cellbal(np.concatenate(ys), np.concatenate(preds), np.concatenate(fs))
        chosen = max(inner, key=inner.get)
        outs = {}
        for name in grid:
            m = make_model(name, 1, y_all[tr], INTER)
            m.fit(X[tr], y_all[tr], sample_weight=cell_weights(fam_codes[tr], y_all[tr]))
            outs[name] = (m.predict_proba(X[te])[:, 1] >= 0.5)
        return f, chosen, inner, te, outs

    rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(outer)(f) for f in families)
    per = {name: np.zeros((len(families), 2, 2), dtype=np.int64) for name in grid}
    nested = np.zeros((len(families), 2, 2), dtype=np.int64)
    chosen_rows = []
    for i, (f, chosen, inner, te, outs) in enumerate(rs):
        for name in grid:
            per[name][i] = counts_from(fam_arr[te], y_all[te], outs[name], [f])[0]
        nested[i] = counts_from(fam_arr[te], y_all[te], outs[chosen], [f])[0]
        chosen_rows.append({"family": f, "chosen": chosen, **{f"inner_{k}": round(v, 4) for k, v in inner.items()}})
    pd.DataFrame(chosen_rows).to_csv(out / "regularization_nested_choices.csv", index=False)
    summ = {name: metrics_from_counts(per[name]) for name in grid}
    summ["NESTED_SELECTION"] = metrics_from_counts(nested)
    (out / "regularization.json").write_text(json.dumps(summ, indent=1))


def stage_smoothing(frame, families, fam_codes, args, out: Path):
    """Filtro de persistencia causal: media movil de P(memory) sobre los ultimos k intervalos.

    Supuesto verificado aqui: el orden de filas del CSV conserva el orden
    temporal dentro de cada bloque (kernel_ref, freq_level_id) contiguo. Las
    repeticiones del mismo bloque van concatenadas, asi que solo las primeras
    k-1 filas de cada repeticion mezclan colas de la anterior (efecto acotado).
    """
    sample = capped_sample(frame, args.cap, 1000)
    counts, probas = lofo(frame, families, fam_codes, CANDIDATE, sample, 2000, True, args.n_jobs, return_proba=True)
    idx = np.concatenate([probas[f][0] for f in families])
    p = np.concatenate([probas[f][1] for f in families])
    order = np.argsort(idx)
    idx, p = idx[order], p[order]
    sub = frame.iloc[idx].reset_index(drop=True)
    sub["p"] = p
    block = ((sub["kernel_ref"] != sub["kernel_ref"].shift()) | (sub["freq_level_id"] != sub["freq_level_id"].shift())).cumsum()
    # ¿el orden es temporal? autocorrelacion lag-1 de log1p_mpki dentro de bloque vs. permutado
    x = sub["log1p_mpki"]
    ac = float(x.groupby(block).apply(lambda s: s.autocorr(1) if len(s) > 10 else np.nan).mean())
    perm = x.sample(frac=1, random_state=0).reset_index(drop=True)
    ac_perm = float(perm.groupby(block).apply(lambda s: s.autocorr(1) if len(s) > 10 else np.nan).mean())
    fam_arr = sub["family"].to_numpy()
    y = sub["y"].to_numpy()
    share = sub.groupby("family")["y"].mean()
    mixed = set(share[share.between(0.10, 0.90)].index)
    rows = []
    for k in [1, 3, 5, 10, 20, 50, 100]:
        ps = sub["p"].groupby(block).transform(lambda s: s.rolling(k, min_periods=1).mean()) if k > 1 else sub["p"]
        pred = ps.to_numpy() >= 0.5
        c = counts_from(fam_arr, y, pred, families)
        m = metrics_from_counts(c)
        mi = [i for i, f in enumerate(families) if f in mixed]
        mm = metrics_from_counts(c[mi])
        pure_i = [i for i, f in enumerate(families) if f not in mixed]
        mp = metrics_from_counts(c[pure_i])
        rows.append({"window_k": k, **{f"all_{a}": round(b, 4) for a, b in m.items() if a in ("cell_balanced_acc", "pooled_f1_macro", "recall_compute", "recall_memory")},
                     "mixed_cell_bal": round(mm["cell_balanced_acc"], 4), "pure_cell_bal": round(mp["cell_balanced_acc"], 4)})
        print("[smoothing]", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "smoothing_window.csv", index=False)
    fam_rows = []
    for k in (1, 10, 50):
        ps = sub["p"].groupby(block).transform(lambda s: s.rolling(k, min_periods=1).mean()) if k > 1 else sub["p"]
        pred = ps.to_numpy() >= 0.5
        c = counts_from(fam_arr, y, pred, families)
        for i, f in enumerate(families):
            tot = c[i].sum(axis=1)
            cells = [c[i, j, j] / tot[j] for j in (0, 1) if tot[j]]
            fam_rows.append({"window_k": k, "family": f, "memory_share": round(float(share[f]), 3), "cell_balanced_acc": round(float(np.mean(cells)), 4)})
    pd.DataFrame(fam_rows).to_csv(out / "smoothing_by_family.csv", index=False)
    (out / "smoothing_meta.json").write_text(json.dumps({"n_blocks": int(block.nunique()), "autocorr_lag1_in_block": ac,
                                                         "autocorr_lag1_shuffled": ac_perm, "mixed_families": sorted(mixed)}, indent=1))


def stage_oi_proxy(frame, families, fam_codes, args, out: Path):
    """Regresion de log2(OI) desde PMU y decision por margen contra el ridge del nivel de frecuencia.

    El ridge de cada nivel es una constante de calibracion (i_ridge_used es constante por nivel),
    por lo que en despliegue se obtiene de la frecuencia sin medir FLOPs ni bytes.
    Hipotesis previa: el objetivo continuo conserva la distancia al ridge que la etiqueta binaria descarta.
    """
    from xgboost import XGBRegressor
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)
    target = np.log2(frame["operational_intensity_uncore_real"].clip(lower=1e-3)).clip(-8, 12).to_numpy()
    log_ridge = np.log2(frame["i_ridge_used"].to_numpy())
    res = {"proxy": [], "clf": []}
    corr_rows = []
    for s in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s)

        def one(f):
            tr = sample[fam_arr[sample] != f]
            te = np.flatnonzero(fam_arr == f)
            w = cell_weights(fam_codes[tr], y_all[tr])
            reg = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=1, random_state=s, verbosity=0)
            reg.fit(X[tr], target[tr], sample_weight=w)
            pred_t = reg.predict(X[te])
            return f, te, pred_t

        rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(one)(f) for f in families)
        counts = np.zeros((len(families), 2, 2), dtype=np.int64)
        for i, (f, te, pt) in enumerate(rs):
            pred_mem = (pt - log_ridge[te]) < 0  # OI predicho por debajo del ridge -> memory_bound
            counts[i] = counts_from(fam_arr[te], y_all[te], pred_mem, [f])[0]
            if s == 0:
                from scipy.stats import spearmanr
                corr_rows.append({"family": f, "spearman_pred_vs_true_log2oi": round(float(spearmanr(pt, target[te])[0]), 4) if len(np.unique(target[te])) > 1 else None,
                                  "mae_log2oi": round(float(np.mean(np.abs(pt - target[te]))), 3)})
        res["proxy"].append(metrics_from_counts(counts))
        res["clf"].append(metrics_from_counts(lofo(frame, families, fam_codes, CANDIDATE, sample, 2000 + s, True, args.n_jobs)))
        print("[oi_proxy] seed", s, {k: round(v[-1]["cell_balanced_acc"], 4) for k, v in res.items()}, flush=True)
    summary = {k: {m: {"mean": float(np.mean([r[m] for r in v])), "sd": float(np.std([r[m] for r in v], ddof=1)) if len(v) > 1 else 0.0} for m in v[0]} for k, v in res.items()}
    (out / "oi_proxy.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(corr_rows).to_csv(out / "oi_proxy_by_family.csv", index=False)


def stage_cap_sensitivity(frame, families, fam_codes, args, out: Path):
    """El tope por familia x clase, limita la informacion util? Entrena con distintos topes y prueba con todo."""
    rows = []
    for cap in [100, 300, 1000, 3000, 10000, 30000]:
        vals = []
        for s in range(3):
            sample = capped_sample(frame, cap, 1000 + s)
            c = lofo(frame, families, fam_codes, CANDIDATE, sample, 2000 + s, True, args.n_jobs)
            vals.append(metrics_from_counts(c))
        rows.append({"cap": cap, "n_train_rows_approx": int(len(sample)), **{k: round(float(np.mean([v[k] for v in vals])), 4) for k in vals[0]},
                     "sd_cell_bal": round(float(np.std([v["cell_balanced_acc"] for v in vals], ddof=1)), 4)})
        print("[cap]", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "cap_sensitivity.csv", index=False)


def stage_threshold_nested(frame, families, fam_codes, args, out: Path):
    """Umbral de decision (sobre P(memory)) elegido por LOFO interno para maximizar exactitud balanceada por celda."""
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sample = capped_sample(frame, args.cap, 1000)
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)

    def cellbal(y, pred, fam):
        cells = [np.mean(pred[(fam == f) & (y == c)] == c) for f in np.unique(fam) for c in (0, 1) if ((fam == f) & (y == c)).any()]
        return float(np.mean(cells))

    def outer(f):
        tr = sample[fam_arr[sample] != f]
        ps, ys, fs = [], [], []
        for g in [g for g in families if g != f]:
            a = tr[fam_arr[tr] != g]; b = tr[fam_arr[tr] == g]
            m = make_model("xgb", 1, y_all[a], INTER)
            m.fit(X[a], y_all[a], sample_weight=cell_weights(fam_codes[a], y_all[a]))
            ps.append(m.predict_proba(X[b])[:, 1]); ys.append(y_all[b]); fs.append(fam_arr[b])
        p, yy, ff = np.concatenate(ps), np.concatenate(ys), np.concatenate(fs)
        best = max(grid, key=lambda t: cellbal(yy, (p >= t).astype(int), ff))
        m = make_model("xgb", 1, y_all[tr], INTER)
        m.fit(X[tr], y_all[tr], sample_weight=cell_weights(fam_codes[tr], y_all[tr]))
        te = np.flatnonzero(fam_arr == f)
        pt = m.predict_proba(X[te])[:, 1]
        return f, best, te, pt

    rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(outer)(f) for f in families)
    c_sel = np.zeros((len(families), 2, 2), dtype=np.int64)
    c_fix = np.zeros_like(c_sel)
    for i, (f, thr, te, pt) in enumerate(rs):
        c_sel[i] = counts_from(fam_arr[te], y_all[te], pt >= thr, [f])[0]
        c_fix[i] = counts_from(fam_arr[te], y_all[te], pt >= 0.5, [f])[0]
    (out / "threshold_nested.json").write_text(json.dumps({"grid": grid, "chosen": {r[0]: r[1] for r in rs},
                                                           "nested_selected": metrics_from_counts(c_sel), "fixed_0.5": metrics_from_counts(c_fix)}, indent=1))


def stage_knn_ceiling(frame, families, fam_codes, args, out: Path):
    """Techo independiente del modelo: k-vecinos en espacio PMU estandarizado, familia retenida fuera."""
    from sklearn.neighbors import KNeighborsClassifier
    feats = ["ipc", "mpki", "cache_miss_rate", "stall_mem_ratio"]
    rows = []
    for use_freq in (False, True):
        cols = feats + (["freq_khz_observed"] if use_freq else [])
        Xall = frame[cols].to_numpy(dtype=np.float64)
        Xall = np.column_stack([np.log1p(np.clip(Xall[:, 1], 0, None))] + [Xall[:, i] for i in range(len(cols)) if i != 1])
        fam_arr = frame["family"].to_numpy()
        y_all = frame["y"].to_numpy()
        vals = []
        for s_ in range(3):
            sample = capped_sample(frame, args.cap, 1000 + s_)
            mu, sd = Xall[sample].mean(0), Xall[sample].std(0) + 1e-9
            Z = (Xall - mu) / sd
            counts = np.zeros((len(families), 2, 2), dtype=np.int64)
            for i, f in enumerate(families):
                tr = sample[fam_arr[sample] != f]
                te = sample[fam_arr[sample] == f]
                w = cell_weights(fam_codes[tr], y_all[tr])
                # vecinos ponderados no disponibles: se remuestrea por celda para equilibrar
                rng = np.random.default_rng(s_)
                pick = rng.choice(len(tr), size=len(tr), replace=True, p=w / w.sum())
                knn = KNeighborsClassifier(n_neighbors=25, n_jobs=args.n_jobs).fit(Z[tr][pick], y_all[tr][pick])
                counts[i] = counts_from(fam_arr[te], y_all[te], knn.predict(Z[te]), [f])[0]
            vals.append(metrics_from_counts(counts))
        rows.append({"knn_k25": "with_freq" if use_freq else "pmu_only", **{k: round(float(np.mean([v[k] for v in vals])), 4) for k in vals[0]}})
        print("[knn]", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "knn_ceiling.csv", index=False)


def stage_twins(frame, families, fam_codes, args, out: Path):
    """Celdas familia x clase cuyo perfil PMU mediano se parece mas al de una celda de OTRA familia con etiqueta opuesta."""
    feats = ["ipc", "mpki", "cache_miss_rate", "stall_mem_ratio"]
    med = frame.groupby(["family", "y"])[feats].median()
    Z = med.copy()
    Z["mpki"] = np.log1p(Z["mpki"])
    Z = (Z - Z.mean()) / Z.std()
    rows = []
    for (fam, y), z in Z.iterrows():
        others = Z[(Z.index.get_level_values(0) != fam)]
        d = np.sqrt(((others - z) ** 2).sum(axis=1))
        opp = d[others.index.get_level_values(1) != y]
        same = d[others.index.get_level_values(1) == y]
        j = opp.idxmin()
        rows.append({"family": fam, "label": "memory" if y else "compute", "nearest_opposite_cell": f"{j[0]}:{'memory' if j[1] else 'compute'}",
                     "dist_opposite": round(float(opp.min()), 3), "dist_same_label": round(float(same.min()), 3) if len(same) else None,
                     "confusable": bool(opp.min() < (same.min() if len(same) else 1e9))})
    pd.DataFrame(rows).sort_values("dist_opposite").to_csv(out / "twin_cells.csv", index=False)
    med.round(4).to_csv(out / "median_pmu_by_family_class.csv")


def stage_temporal(frame, families, fam_codes, args, out: Path):
    """Variables causales de historia reciente (media/desv. movil de las senales PMU en los k intervalos previos)."""
    block = ((frame["kernel_ref"] != frame["kernel_ref"].shift()) | (frame["freq_level_id"] != frame["freq_level_id"].shift())).cumsum()
    base = frame[["ipc", "cache_miss_rate", "stall_mem_ratio"]].copy()
    base["log1p_mpki"] = frame["log1p_mpki"]
    rows = []
    variants = {"inter_only": []}
    for k in (5, 20):
        cols = []
        for c in base.columns:
            grp = base[c].groupby(block)
            frame[f"{c}_rm{k}"] = grp.transform(lambda x: x.rolling(k, min_periods=1).mean()).to_numpy()
            frame[f"{c}_rs{k}"] = grp.transform(lambda x: x.rolling(k, min_periods=2).std()).fillna(0.0).to_numpy()
            cols += [f"{c}_rm{k}", f"{c}_rs{k}"]
        variants[f"inter_hist{k}"] = cols
    variants["hist5_only_means"] = [c for c in variants["inter_hist5"] if "_rm5" in c]
    for name, extra in variants.items():
        feats = INTER + extra
        CONFIGS[f"_tmp_{name}"] = ("xgb", feats, "cell")
    vals = {n: [] for n in variants}
    for s_ in range(3):
        sample = capped_sample(frame, args.cap, 1000 + s_)
        for n in variants:
            c = lofo(frame, families, fam_codes, f"_tmp_{n}", sample, 2000 + s_, True, args.n_jobs)
            vals[n].append(metrics_from_counts(c))
        print("[temporal] seed", s_, {n: round(v[-1]["cell_balanced_acc"], 4) for n, v in vals.items()}, flush=True)
    for n, v in vals.items():
        rows.append({"variant": n, "n_features": len(INTER) + len(variants[n]), **{k: round(float(np.mean([m[k] for m in v])), 4) for k in v[0]},
                     "sd_cell_bal": round(float(np.std([m["cell_balanced_acc"] for m in v], ddof=1)), 4)})
    pd.DataFrame(rows).to_csv(out / "temporal_history.csv", index=False)


def stage_power_w(frame, families, fam_codes, args, out: Path):
    """Variable `power_w` (RAPL paquete + DRAM por intervalo) que el plan §2.5 lista como entrada CPU."""
    if "power_w" not in frame.columns:
        raise SystemExit("la fuente no tiene power_w; usar build_power_source.py")
    variants = {"inter_only": [], "inter_plus_power_w": ["power_w"]}
    for n, extra in variants.items():
        CONFIGS[f"_tmp_{n}"] = ("xgb", INTER + extra, "cell")
    vals = {n: [] for n in variants}
    for s_ in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s_)
        for n in variants:
            c = lofo(frame, families, fam_codes, f"_tmp_{n}", sample, 2000 + s_, True, args.n_jobs)
            vals[n].append(metrics_from_counts(c))
        print("[power_w] seed", s_, {n: round(v[-1]["cell_balanced_acc"], 4) for n, v in vals.items()}, flush=True)
    rows = [{"variant": n, **{k: round(float(np.mean([m[k] for m in v])), 4) for k in v[0]},
             "sd_cell_bal": round(float(np.std([m["cell_balanced_acc"] for m in v], ddof=1)), 4)} for n, v in vals.items()]
    pd.DataFrame(rows).to_csv(out / "power_w.csv", index=False)


def stage_selection(frame, families, fam_codes, args, out: Path):
    """Seleccion 6 vs 12 variables: diferencia pareada de exactitud balanceada por celda (bootstrap de familias)."""
    cnt = {c: [] for c in ("xgb_base", "xgb_inter")}
    for s_ in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s_)
        for c in cnt:
            cnt[c].append(lofo(frame, families, fam_codes, c, sample, 2000 + s_, True, args.n_jobs))
        print("[selection] seed", s_, flush=True)
    mean_cnt = {c: np.sum(v, axis=0) for c, v in cnt.items()}
    rng = np.random.default_rng(0)
    F = len(families)
    diffs = []
    for _ in range(4000):
        d = rng.integers(0, F, F)
        diffs.append(metrics_from_counts(mean_cnt["xgb_inter"][d])["cell_balanced_acc"]
                     - metrics_from_counts(mean_cnt["xgb_base"][d])["cell_balanced_acc"])
    diffs = np.array(diffs)
    res = {"cell_bal": {c: metrics_from_counts(v)["cell_balanced_acc"] for c, v in mean_cnt.items()},
           "diff_inter_minus_base": float(metrics_from_counts(mean_cnt["xgb_inter"])["cell_balanced_acc"]
                                          - metrics_from_counts(mean_cnt["xgb_base"])["cell_balanced_acc"]),
           "diff_ci95": ci(diffs), "p_diff_le_0": float((diffs <= 0).mean())}
    (out / "selection_6_vs_12.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage_external(frame, families, fam_codes, args, out: Path):
    """Prueba externa: modelo final entrenado en las 24 familias, evaluado UNA vez en familias inéditas (`--external`)."""
    ext = load(args.external)
    ext["family"] = ext["kernel_ref"]
    n0 = len(frame)
    comb = pd.concat([frame, ext], ignore_index=True)
    codes = pd.Categorical(comb["family"]).codes
    y = comb["y"].to_numpy()
    ext_fams = sorted(ext["family"].unique())
    rows = []
    per_seed = {f: [] for f in ext_fams}
    for s_ in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s_)
        test = np.arange(n0, len(comb))
        p = fit_predict("xgb", INTER, "cell", comb, sample, test, 2000 + s_, codes)
        for f in ext_fams:
            m = (comb["family"].to_numpy()[test] == f)
            pf, yf = p[m], y[test][m]
            pred = pf >= 0.5
            conf = np.maximum(pf, 1 - pf)
            keep = conf >= FINAL_THRESHOLD
            per_seed[f].append(dict(n=int(m.sum()), true_memory_share=float(yf.mean()),
                                    acc=float((pred == yf).mean()),
                                    coverage=float(keep.mean()),
                                    sel_acc=float((pred[keep] == yf[keep]).mean()) if keep.any() else float("nan"),
                                    pred_compute_share=float((~pred).mean())))
    for f, v in per_seed.items():
        rows.append({"family": f, **{k: round(float(np.nanmean([r[k] for r in v])), 4) for k in v[0]},
                     "sd_acc": round(float(np.std([r["acc"] for r in v], ddof=1)), 4)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "external_families.csv", index=False)
    print(t.to_string(index=False), flush=True)


def stage_retrain_ext(frame, families, fam_codes, args, out: Path):
    """LOFO con las familias inéditas añadidas al conjunto (cada kernel nuevo es una familia; se excluye ptrchase)."""
    ext = load(args.external)
    ext = ext[ext["kernel_ref"] != "ptrchase"].copy()
    ext["family"] = ext["kernel_ref"]
    comb = pd.concat([frame, ext], ignore_index=True)
    fams = sorted(comb["family"].unique())
    codes = pd.Categorical(comb["family"], categories=fams).codes
    old = [i for i, f in enumerate(fams) if f in set(families)]
    new = [i for i, f in enumerate(fams) if f not in set(families)]
    rows = {"all": [], "old24": [], "new6": []}
    for s_ in range(args.seeds):
        sample = capped_sample(comb, args.cap, 1000 + s_)
        c = lofo(comb, fams, codes, "xgb_inter", sample, 2000 + s_, True, args.n_jobs)
        for k, idx in (("all", list(range(len(fams)))), ("old24", old), ("new6", new)):
            rows[k].append(metrics_from_counts(c[idx]))
        print("[retrain_ext] seed", s_, {k: round(v[-1]["cell_balanced_acc"], 4) for k, v in rows.items()}, flush=True)
    res = [{"subset": k, "n_families": len(idx), **{m: round(float(np.mean([r[m] for r in v])), 4) for m in v[0]},
            "sd_cell_bal": round(float(np.std([r["cell_balanced_acc"] for r in v], ddof=1)), 4)}
           for (k, v), idx in zip(rows.items(), (range(len(fams)), old, new))]
    pd.DataFrame(res).to_csv(out / "retrain_with_new_families.csv", index=False)
    print(pd.DataFrame(res).to_string(index=False), flush=True)


def stage_adaptation(frame, families, fam_codes, args, out: Path):
    """Calibracion en linea: ¿cuanto mejora si los primeros n intervalos etiquetados (uncore) de cada bloque
    kernel x frecuencia de la familia nueva se anaden al entrenamiento? Se evalua sobre el resto de esa familia.
    Es una cota de lo que podria aportar la microcaracterizacion de la politica 'revisar'."""
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)
    block = ((frame["kernel_ref"] != frame["kernel_ref"].shift()) | (frame["freq_level_id"] != frame["freq_level_id"].shift())).cumsum().to_numpy()
    pos_in_block = frame.groupby(block).cumcount().to_numpy()
    rows = []
    for n in [0, 5, 20, 100, 500]:
        vals = []
        for s_ in range(3):
            sample = capped_sample(frame, args.cap, 1000 + s_)

            def one(f):
                tr = sample[fam_arr[sample] != f]
                idx_f = np.flatnonzero(fam_arr == f)
                adapt = idx_f[pos_in_block[idx_f] < n]
                test = idx_f[pos_in_block[idx_f] >= 500]  # mismo conjunto de evaluacion para todo n
                if n == 0:
                    adapt = idx_f[:0]
                if len(test) == 0:
                    return np.zeros((2, 2), dtype=np.int64), 0
                train = np.concatenate([tr, adapt])
                w = cell_weights(fam_codes[train], y_all[train])
                # el bloque adaptado pesa como una familia mas (peso de celda propio), sin sobreponderar
                m = make_model("xgb", s_, y_all[train], INTER)
                m.fit(X[train], y_all[train], sample_weight=w)
                pred = m.predict_proba(X[test])[:, 1] >= 0.5
                return counts_from(fam_arr[test], y_all[test], pred, [f])[0], len(adapt)

            rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(one)(f) for f in families)
            counts = np.stack([r[0] for r in rs])
            vals.append((metrics_from_counts(counts), float(np.mean([r[1] for r in rs]))))
        rows.append({"n_first_intervals_per_block": n, "mean_labeled_rows_per_family": round(float(np.mean([v[1] for v in vals])), 1),
                     **{k: round(float(np.mean([v[0][k] for v in vals])), 4) for k in vals[0][0]}})
        print("[adapt]", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "adaptation_online.csv", index=False)


def stage_adaptation_phases(frame, families, fam_codes, args, out: Path):
    """Calibracion en linea causal con cambios de fase reales.

    Para cada bloque kernel x frecuencia se etiquetan intervalos SOLO en la primera mitad y se evalua en la
    segunda mitad (posiciones >= L/2), que incluye cambios de fase posteriores a la calibracion.
    Esquemas: base (sin adaptar); adapt_start (5 primeros intervalos); adapt_periodic (5 de cada 200 en la
    primera mitad); y dos referencias triviales que solo repiten la ultima etiqueta conocida
    (sticky_start, sticky_periodic). Bloques 'mixtos': proporcion memory entre 20 y 80 %.
    """
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)
    block = ((frame["kernel_ref"] != frame["kernel_ref"].shift()) | (frame["freq_level_id"] != frame["freq_level_id"].shift())).cumsum().to_numpy()
    pos = frame.groupby(block).cumcount().to_numpy()
    length = frame.groupby(block)["y"].transform("size").to_numpy()
    half = length // 2
    share = frame.groupby(block)["y"].transform("mean").to_numpy()
    mixed_block = (share >= 0.2) & (share <= 0.8) & (length >= 400)
    eval_mask = (pos >= half) & (length >= 400)
    lab_start = (pos < 5) & (length >= 400)
    lab_periodic = (pos < half) & (pos % 200 < 5) & (length >= 400)
    # etiqueta pegajosa por bloque (ultimo grupo de 5 etiquetado)
    def sticky(lab_mask, last=False):
        df = pd.DataFrame({"b": block[lab_mask], "y": y_all[lab_mask], "p": pos[lab_mask]})
        if last:
            df = df[df.groupby("b")["p"].transform(lambda x: x >= x.max() - 4)]
        m = df.groupby("b")["y"].mean()
        return pd.Series(block).map(m).fillna(0.5).to_numpy() >= 0.5
    st_start, st_last = sticky(lab_start), sticky(lab_periodic, last=True)
    res = {}
    for s_ in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s_)

        def one(f):
            tr = sample[fam_arr[sample] != f]
            idx = np.flatnonzero(fam_arr == f)
            te = idx[eval_mask[idx]]
            outp = {}
            if len(te) == 0:
                return f, te, {k: np.zeros(0, bool) for k in ("base", "adapt_start", "adapt_periodic", "sticky_start", "sticky_periodic")}
            for name, lab in (("base", None), ("adapt_start", lab_start), ("adapt_periodic", lab_periodic)):
                train = tr if lab is None else np.concatenate([tr, idx[lab[idx]]])
                m = make_model("xgb", s_, y_all[train], INTER)
                m.fit(X[train], y_all[train], sample_weight=cell_weights(fam_codes[train], y_all[train]))
                outp[name] = m.predict_proba(X[te])[:, 1] >= 0.5
            outp["sticky_start"] = st_start[te]
            outp["sticky_periodic"] = st_last[te]
            return f, te, outp

        rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(one)(f) for f in families)
        for scope in ("all", "mixed_blocks", "pure_blocks"):
            for name in ("base", "adapt_start", "adapt_periodic", "sticky_start", "sticky_periodic"):
                counts = np.zeros((len(families), 2, 2), dtype=np.int64)
                for i, (f, te, outp) in enumerate(rs):
                    sel = np.ones(len(te), bool) if scope == "all" else (mixed_block[te] if scope == "mixed_blocks" else ~mixed_block[te])
                    if sel.any():
                        counts[i] = counts_from(fam_arr[te][sel], y_all[te][sel], outp[name][sel], [f])[0]
                res.setdefault((scope, name), []).append(metrics_from_counts(counts))
        print("[adapt_phases] seed", s_, flush=True)
    rows = []
    for (scope, name), v in res.items():
        rows.append({"scope": scope, "method": name, **{k: round(float(np.mean([m[k] for m in v])), 4) for k in v[0]},
                     "sd_cell_bal": round(float(np.std([m["cell_balanced_acc"] for m in v], ddof=1)), 4) if len(v) > 1 else 0.0})
    pd.DataFrame(rows).to_csv(out / "adaptation_phases.csv", index=False)
    labeled_frac = {"start_rows_frac_of_eval_blocks": float(lab_start.sum() / max(eval_mask.sum(), 1)),
                    "periodic_rows_frac_of_eval_blocks": float(lab_periodic.sum() / max(eval_mask.sum(), 1)),
                    "eval_rows": int(eval_mask.sum()), "mixed_block_eval_rows": int((eval_mask & mixed_block).sum()),
                    "mixed_blocks": int(len(np.unique(block[mixed_block]))),
                    "label_change_share_in_mixed_eval": float((y_all[eval_mask & mixed_block] != st_start[eval_mask & mixed_block]).mean())}
    (out / "adaptation_phases_meta.json").write_text(json.dumps(labeled_frac, indent=1))


def stage_nested_optuna(frame, families, fam_codes, args, out: Path):
    """Busqueda de hiperparametros anidada (TPE) sobre la representacion final.

    Para cada familia externa: TPE sobre el LOFO interno de las 23 familias restantes, reajuste con la
    mejor configuracion y una sola evaluacion en la familia retenida. La familia externa nunca interviene
    en la busqueda. Se compara contra la configuracion fija del candidato.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)
    sample = capped_sample(frame, args.cap, 1000)

    def cellbal(y, pred, fam):
        cells = [np.mean(pred[(fam == f) & (y == c)] == c) for f in np.unique(fam) for c in (0, 1) if ((fam == f) & (y == c)).any()]
        return float(np.mean(cells))

    def build(kind, params):
        if kind == "xgboost":
            return XGBClassifier(n_jobs=1, random_state=0, eval_metric="logloss", verbosity=0, scale_pos_weight=1.0, **params)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0, **params))

    def space(kind, trial):
        if kind == "xgboost":
            return {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.5, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)}
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}

    def score_inner(kind, params, train_idx, inner_fams):
        preds, ys, fs = [], [], []
        for g in inner_fams:
            a = train_idx[fam_arr[train_idx] != g]
            b = train_idx[fam_arr[train_idx] == g]
            m = build(kind, params)
            w = cell_weights(fam_codes[a], y_all[a])
            if hasattr(m, "steps"):
                m.fit(X[a], y_all[a], **{f"{m.steps[-1][0]}__sample_weight": w})
            else:
                m.fit(X[a], y_all[a], sample_weight=w)
            preds.append(m.predict_proba(X[b])[:, 1] >= 0.5); ys.append(y_all[b]); fs.append(fam_arr[b])
        return cellbal(np.concatenate(ys), np.concatenate(preds).astype(int), np.concatenate(fs))

    results = {}
    for kind, n_trials in (("xgboost", args.optuna_trials), ("logistic", max(args.optuna_trials // 2, 6))):
        def outer(f):
            tr = sample[fam_arr[sample] != f]
            inner_fams = [g for g in families if g != f]
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=20260918))
            study.optimize(lambda t: score_inner(kind, space(kind, t), tr, inner_fams), n_trials=n_trials, n_jobs=1)
            best = study.best_params
            m = build(kind, best)
            w = cell_weights(fam_codes[tr], y_all[tr])
            if hasattr(m, "steps"):
                m.fit(X[tr], y_all[tr], **{f"{m.steps[-1][0]}__sample_weight": w})
            else:
                m.fit(X[tr], y_all[tr], sample_weight=w)
            te = np.flatnonzero(fam_arr == f)
            return f, best, float(study.best_value), te, m.predict_proba(X[te])[:, 1] >= 0.5

        rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(outer)(f) for f in families)
        counts = np.zeros((len(families), 2, 2), dtype=np.int64)
        rows = []
        for i, (f, best, val, te, pred) in enumerate(rs):
            counts[i] = counts_from(fam_arr[te], y_all[te], pred, [f])[0]
            tot = counts[i].sum(axis=1)
            cells = [counts[i, j, j] / tot[j] for j in (0, 1) if tot[j]]
            rows.append({"model": kind, "family": f, "inner_best_score": round(val, 4),
                         "outer_cell_balanced_acc": round(float(np.mean(cells)), 4), **{f"p_{k}": v for k, v in best.items()}})
        results[kind] = metrics_from_counts(counts)
        pd.DataFrame(rows).to_csv(out / f"nested_optuna_{kind}_folds.csv", index=False)
        print(f"[optuna] {kind}", {k: round(v, 4) for k, v in results[kind].items()}, flush=True)

    # referencia: configuracion fija, mismo muestreo y mismos pliegues
    for cfg in ("xgb_inter", "logistic_inter"):
        results[f"fixed_{cfg}"] = metrics_from_counts(lofo(frame, families, fam_codes, cfg, sample, 2000, True, args.n_jobs))
    (out / "nested_optuna.json").write_text(json.dumps({"trials_xgboost": args.optuna_trials, "results": results}, indent=1))


FINAL_THRESHOLD = 0.85


def stage_final_model(frame, families, fam_codes, args, out: Path):
    """Cifras del modelo final en una sola poblacion y una sola configuracion.

    Configuracion congelada: XGBoost, 12 entradas, pesos por celda familia x clase, umbral 0.85.
    Protocolo: LOFO, entrenamiento sobre la muestra con tope por celda, evaluacion sobre TODOS los
    intervalos elegibles de la familia retenida. Promedio de ``seeds`` semillas de muestreo para los
    conteos (se redondean al reportar). Todo lo que el capitulo cita sale de aqui.
    """
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    freq = frame["freq_level_id"].to_numpy()
    acc_counts, acc_sel, per_family, per_freq = [], [], [], []
    for s_ in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s_)
        _, probas = lofo(frame, families, fam_codes, CANDIDATE, sample, 2000 + s_, True, args.n_jobs, return_proba=True)
        idx = np.concatenate([probas[f][0] for f in families])
        p = np.concatenate([probas[f][1] for f in families])
        pred = p >= 0.5
        conf = np.maximum(p, 1 - p)
        keep = conf >= FINAL_THRESHOLD
        y, fam = y_all[idx], fam_arr[idx]
        acc_counts.append(counts_from(fam, y, pred, families))
        acc_sel.append(counts_from(fam[keep], y[keep], pred[keep], families))
        d = pd.DataFrame({"family": fam, "freq": freq[idx], "y": y, "pred": pred, "keep": keep})
        d["correct"] = d["pred"] == d["y"]
        g = d.groupby("family")
        per_family.append(pd.DataFrame({
            "n": g.size(), "memory_share": g["y"].mean(), "accuracy": g["correct"].mean(),
            "coverage": g["keep"].mean(),
            "accuracy_auto": g.apply(lambda x: x.loc[x["keep"], "correct"].mean() if x["keep"].any() else np.nan, include_groups=False),
        }))
        h = d.groupby("freq")
        per_freq.append(pd.DataFrame({"n": h.size(), "accuracy": h["correct"].mean(), "coverage": h["keep"].mean(),
                                      "share_memory": h["y"].mean(),
                                      "correct_auto": h.apply(lambda x: (x["keep"] & x["correct"]).sum(), include_groups=False),
                                      "error_auto": h.apply(lambda x: (x["keep"] & ~x["correct"]).sum(), include_groups=False),
                                      "abstained": h.apply(lambda x: (~x["keep"]).sum(), include_groups=False)}))

    def summarize(stack):
        c = np.mean(np.stack(stack), axis=0)
        pooled = c.sum(axis=0)
        tn, fp, fn, tp = pooled[0, 0], pooled[0, 1], pooled[1, 0], pooled[1, 1]
        def prf(tp_, fp_, fn_):
            pr = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
            rc = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
            return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                    "support": float(tp_ + fn_)}
        m = metrics_from_counts(c)
        return {**m, "confusion": {"compute_as_compute": float(tn), "compute_as_memory": float(fp),
                                   "memory_as_compute": float(fn), "memory_as_memory": float(tp)},
                "per_class": {"compute_bound": prf(tn, fn, fp), "memory_bound": prf(tp, fp, fn)}}

    fam_tab = sum(per_family) / len(per_family)
    fam_tab.round(4).to_csv(out / "final_model_by_family.csv")
    freq_tab = sum(per_freq) / len(per_freq)
    freq_tab.round(4).to_csv(out / "final_model_by_frequency.csv")
    total = float(np.mean([c.sum() for c in acc_counts]))
    cov = float(np.mean([c.sum() for c in acc_sel])) / total
    summary = {"protocol": "LOFO 24 familias; entrenamiento en muestra con tope 1000 por celda; evaluacion en todos los intervalos elegibles de la familia retenida",
               "model": "xgboost n_estimators=100 max_depth=6 scale_pos_weight=1", "features": INTER,
               "threshold": FINAL_THRESHOLD, "seeds": args.seeds, "eval_rows_mean": total,
               "full_coverage": summarize(acc_counts), "selective": {**summarize(acc_sel), "coverage_pooled": cov,
               "coverage_mean_family": float(fam_tab["coverage"].mean()), "coverage_min_family": float(fam_tab["coverage"].min())}}
    (out / "final_model.json").write_text(json.dumps(summary, indent=1))
    print("[final]", {k: round(v, 4) for k, v in summary["full_coverage"].items() if isinstance(v, float)}, flush=True)
    print("[final sel]", {k: round(v, 4) for k, v in summary["selective"].items() if isinstance(v, float)}, flush=True)


def stage_metrics_suite(frame, families, fam_codes, args, out: Path):
    """Todas las metricas del modelo final sobre una sola poblacion, con IC95 por bootstrap de familias.

    Incluye las metricas tradicionales (exactitud, precision, sensibilidad, F1, exactitud balanceada
    clasica, AUC-ROC, MCC, kappa), la metrica por celdas y su version sin celdas de soporte minimo,
    y la tabla de las celdas familia x clase con su tamano y su acierto.
    """
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    per_seed, cell_tabs = [], []
    boot_inputs = []
    for s_ in range(args.seeds):
        sample = capped_sample(frame, args.cap, 1000 + s_)
        _, probas = lofo(frame, families, fam_codes, CANDIDATE, sample, 2000 + s_, True, args.n_jobs, return_proba=True)
        idx = np.concatenate([probas[f][0] for f in families])
        p = np.concatenate([probas[f][1] for f in families])
        y, fam, pred = y_all[idx], fam_arr[idx], p >= 0.5
        tn = int(((~y) & (~pred)).sum()); fp = int(((~y) & pred).sum()); fn = int((y & (~pred)).sum()); tp = int((y & pred).sum())
        # nota: la clase positiva es memory_bound
        rec_mem, rec_com = tp / (tp + fn), tn / (tn + fp)
        pre_mem, pre_com = tp / (tp + fp), tn / (tn + fn)
        f1 = lambda a, b: 2 * a * b / (a + b)
        d = pd.DataFrame({"f": fam, "y": y, "ok": pred == y})
        cell = d.groupby(["f", "y"])["ok"].agg(acierto="mean", n="size").reset_index()
        big = cell[cell["n"] >= 100]
        per_seed.append({
            "accuracy": (tp + tn) / len(y),
            "precision_compute": pre_com, "recall_compute": rec_com, "f1_compute": f1(pre_com, rec_com),
            "precision_memory": pre_mem, "recall_memory": rec_mem, "f1_memory": f1(pre_mem, rec_mem),
            "f1_macro_pooled": (f1(pre_com, rec_com) + f1(pre_mem, rec_mem)) / 2,
            "balanced_accuracy_pooled": (rec_com + rec_mem) / 2,
            "auc_roc_pooled": float(roc_auc_score(y, p)),
            "mcc": float(matthews_corrcoef(y, pred)), "kappa": float(cohen_kappa_score(y, pred)),
            "cell_balanced_acc": float(cell["acierto"].mean()),
            "cell_balanced_acc_min100": float(big["acierto"].mean()),
            "cell_recall_compute": float(cell[~cell["y"]]["acierto"].mean()),
            "cell_recall_memory": float(cell[cell["y"]]["acierto"].mean()),
            "n_cells": int(len(cell)), "n_cells_lt100": int((cell["n"] < 100).sum()),
            "share_of_metric_weight_lt100": float((cell["n"] < 100).mean()),
            "n_intervals_in_cells_lt100": int(cell.loc[cell["n"] < 100, "n"].sum()),
        })
        cell_tabs.append(cell.set_index(["f", "y"]))
        # entradas del bootstrap: (acierto por intervalo por familia) para IC de las metricas por volumen
        boot_inputs.append(d)
    keys = list(per_seed[0])
    mean = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
    sd = {k: float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in keys}
    # IC95 por remuestreo de familias sobre la semilla 0 (conteos por familia y clase)
    d0 = boot_inputs[0]
    rng = np.random.default_rng(0)
    g = d0.groupby(["f", "y"])["ok"].agg(["sum", "size"]).reset_index()
    fam_list = sorted(d0["f"].unique())
    tab = {f: g[g["f"] == f] for f in fam_list}
    ci = {"cell_balanced_acc": [], "balanced_accuracy_pooled": [], "accuracy": []}
    for _ in range(args.boot):
        pick = rng.integers(0, len(fam_list), len(fam_list))
        parts = pd.concat([tab[fam_list[i]] for i in pick])
        ci["cell_balanced_acc"].append((parts["sum"] / parts["size"]).mean())
        rc = parts[~parts["y"]]; rm = parts[parts["y"]]
        ci["balanced_accuracy_pooled"].append((rc["sum"].sum() / rc["size"].sum() + rm["sum"].sum() / rm["size"].sum()) / 2)
        ci["accuracy"].append(parts["sum"].sum() / parts["size"].sum())
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in ci.items()}
    avg = sum(c["acierto"] for c in cell_tabs) / len(cell_tabs)
    nn = sum(c["n"] for c in cell_tabs) / len(cell_tabs)
    pd.DataFrame({"acierto": avg, "n_intervalos": nn.round(0)}).reset_index().rename(columns={"f": "familia", "y": "clase_memory"}).to_csv(out / "metrics_cells.csv", index=False)
    (out / "metrics_suite.json").write_text(json.dumps({"mean_5_seeds": mean, "sd_over_seeds": sd, "ci95_family_bootstrap_seed0": ci}, indent=1))
    print("[metrics]", {k: round(v, 4) for k, v in mean.items()}, flush=True)


def stage_importance(frame, families, fam_codes, args, out: Path):
    """Importancia por permutacion en la familia retenida (candidato, semilla 0)."""
    sample = capped_sample(frame, args.cap, 1000)
    fam_arr = frame["family"].to_numpy()
    y_all = frame["y"].to_numpy()
    X = frame[INTER].to_numpy(dtype=np.float32)

    def one(f):
        tr = sample[fam_arr[sample] != f]; te = sample[fam_arr[sample] == f]
        m = make_model("xgb", 2000, y_all[tr], INTER)
        m.fit(X[tr], y_all[tr], sample_weight=cell_weights(fam_codes[tr], y_all[tr]))
        yt = y_all[te]

        def score(Xt):
            pr = m.predict_proba(Xt)[:, 1] >= 0.5
            return np.mean([np.mean(pr[yt == c] == c) for c in (0, 1) if (yt == c).any()])
        base = score(X[te])
        rng = np.random.default_rng(0)
        drops = {}
        for j, name in enumerate(INTER):
            Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops[name] = base - score(Xp)
        return drops
    rs = Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(one)(f) for f in families)
    df = pd.DataFrame(rs)
    pd.DataFrame({"mean_drop": df.mean(), "sd_between_families": df.std(), "share_families_positive": (df > 0).mean()}).sort_values("mean_drop", ascending=False).to_csv(out / "permutation_importance.csv")


def stage_inventory(frame, families, args, out: Path):
    inv = {"n_rows": int(len(frame)), "dropped_nan_rows": int(frame.attrs.get("dropped_nan_rows", 0)), "n_families": len(families),
           "n_kernels": int(frame["kernel_ref"].nunique()), "memory_share_global": float(frame["y"].mean())}
    g = frame.groupby("family").agg(n=("y", "size"), memory_share=("y", "mean"), n_kernels=("kernel_ref", "nunique"))
    g["pure_family"] = (g["memory_share"] < 0.02) | (g["memory_share"] > 0.98)
    g["mixed_10_90"] = g["memory_share"].between(0.10, 0.90)
    g.to_csv(out / "inventory_by_family.csv")
    kf = frame.groupby(["kernel_ref", "freq_level_id"])["y"].agg(["size", "mean"])
    inv["kernel_freq_cells"] = int(len(kf))
    inv["kernel_freq_pure_2pct"] = float(((kf["mean"] < 0.02) | (kf["mean"] > 0.98)).mean())
    inv["kernel_freq_mixed_20_80"] = int(kf["mean"].between(0.2, 0.8).sum())
    inv["pure_families"] = int(g["pure_family"].sum())
    inv["mixed_families_10_90"] = int(g["mixed_10_90"].sum())
    inv["families_with_both_classes"] = int(((g["memory_share"] > 0) & (g["memory_share"] < 1)).sum())
    inv["memory_share_by_freq"] = frame.groupby("freq_level_id")["y"].mean().round(4).to_dict()
    inv["frequency_mean_khz_by_level"] = frame.groupby("freq_level_id")["freq_khz_observed"].mean().round(0).to_dict()
    # ¿cuanto separa cada variable por si sola? AUC univariado agrupado (referencia)
    from sklearn.metrics import roc_auc_score
    inv["univariate_auc_pooled"] = {f: round(float(roc_auc_score(frame["y"], frame[f])), 4) for f in INTER}
    # que se pierde al retirar filas con NaN (miss=0): son casi siempre compute
    nan_mask = frame.attrs.get("dropped_nan_rows", 0)
    (out / "inventory.json").write_text(json.dumps(inv, indent=1))


def stage_latency(frame, args, out: Path):
    """Latencia de inferencia de una fila (mediana/p95/p99) en la maquina donde corre."""
    sample = capped_sample(frame, args.cap, 1000)
    y = frame["y"].to_numpy()[sample]
    fam_codes = pd.Categorical(frame["family"]).codes
    res = {}
    for cfg in ("stump_base", "logistic_inter", "xgb_base", "xgb_inter", "xgb_inter_shallow", "rf_inter", "et_inter"):
        name_model, features, weighting = CONFIGS[cfg]
        model = make_model(name_model, 0, y, features)
        X = frame[features].to_numpy(dtype=np.float32)[sample]
        w = cell_weights(fam_codes[sample], y)
        if hasattr(model, "steps"):
            model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        else:
            model.fit(X, y, sample_weight=w)
        rows = X[:2000]
        for _ in range(50):
            model.predict_proba(rows[:1])
        t = []
        for i in range(2000):
            a = time.perf_counter_ns(); model.predict_proba(rows[i:i + 1]); t.append(time.perf_counter_ns() - a)
        t = np.array(t) / 1e3
        res[cfg] = {"p50_us": float(np.percentile(t, 50)), "p95_us": float(np.percentile(t, 95)), "p99_us": float(np.percentile(t, 99))}
        print("[latency]", cfg, res[cfg], flush=True)
    import platform
    res["host"] = platform.node()
    (out / f"latency_{platform.node()}.json").write_text(json.dumps(res, indent=1))


STAGES = ["inventory", "final_model", "metrics_suite", "matrix", "diagnostics", "protocols", "learning_curve", "smoothing", "nested_selective", "regularization", "importance", "knn_ceiling", "twins", "nested_optuna", "adaptation", "adaptation_phases", "temporal", "oi_proxy", "power_w", "selection", "external", "retrain_ext", "cap_sensitivity", "threshold_nested", "latency"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--lc-reps", type=int, default=3)
    ap.add_argument("--optuna-trials", type=int, default=15)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--external", default=None, help="CSV de familias inéditas para la etapa external")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frame = load(args.source)
    leak = set(INTER) & FORBIDDEN_FEATURES
    assert not leak, leak
    families = sorted(frame["family"].unique())
    fam_codes = pd.Categorical(frame["family"], categories=families).codes
    for st in args.stages.split(","):
        t0 = time.time()
        print(f"=== stage {st}", flush=True)
        if st == "inventory":
            stage_inventory(frame, families, args, out)
        elif st == "latency":
            stage_latency(frame, args, out)
        else:
            globals()[f"stage_{st}"](frame, families, fam_codes, args, out)
        print(f"=== stage {st} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
