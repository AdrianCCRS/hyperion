"""Cierra las brechas de rigor entre el clasificador GPU y el de CPU
(2026-09-22): correlacion/VIF de las 5 features, LOFO del candidato final
con probabilidades guardadas (matriz de confusion, calibracion/ECE) y
validacion externa real contra 4 familias de Rodinia (Backprop, DWT2D,
LavaMD, Myocyte) de la campana de agosto (`pacca_gpu_dvfs_20260820`),
nunca tocadas por ninguna decision de la ronda de hoy (ni el umbral de A1,
ni la separacion de A2, ni la ablacion de A6, ni la eleccion de candidato).

Ese conjunto externo es el equivalente honesto al "conjunto B" que ya tenia
el capitulo GPU del libro antes de esta ronda -- se reconstruye aqui porque
el reemplazo de hoy fusiono todo en un solo LOFO sin conjunto sellado.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fase2_clasificador.analysis.gpu_quality_report import (
    FAMILY_COL, FEATURES, LABEL, fit_predict, load,
)
from fase2_clasificador.eval.protocol import derive_kernel_family

EXTERNAL_FAMILIES = {"rodinia_backprop", "rodinia_dwt2d", "rodinia_lavamd", "rodinia_myocyte"}
SIGNALS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw", "gpu_sm_clock_mhz")


def vif(X: np.ndarray, names: list[str]) -> dict[str, float]:
    """VIF_j = 1 / (1 - R^2) de regresionar la columna j contra las demas."""
    out = {}
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    for j in range(p):
        y = Xc[:, j]
        others = np.delete(Xc, j, axis=1)
        others1 = np.hstack([others, np.ones((n, 1))])
        coef, *_ = np.linalg.lstsq(others1, y, rcond=None)
        pred = others1 @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum(y ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[names[j]] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return out


def lofo_with_proba(frame: pd.DataFrame, families: list[str], fam_codes: np.ndarray, model_name: str, seeds: int):
    """LOFO del candidato, promediando la probabilidad predicha entre semillas."""
    fam_arr = frame[FAMILY_COL].to_numpy()
    y_all = frame["y"].to_numpy()
    proba_sum = np.zeros(len(frame))
    for s in range(seeds):
        for f in families:
            train = np.flatnonzero(fam_arr != f)
            test = np.flatnonzero(fam_arr == f)
            p = fit_predict(model_name, frame, train, test, 2000 + s, fam_codes)
            proba_sum[test] += p
    proba = proba_sum / seeds
    return pd.DataFrame({"family": fam_arr, "y_true": y_all, "proba_memory": proba})


def confusion_and_ece(df: pd.DataFrame, n_bins: int = 10) -> dict:
    pred = df["proba_memory"] >= 0.5
    y = df["y_true"].astype(bool)
    tn = int(((~y) & (~pred)).sum()); fp = int(((~y) & pred).sum())
    fn = int((y & (~pred)).sum()); tp = int((y & pred).sum())
    conf = np.maximum(df["proba_memory"], 1 - df["proba_memory"])
    correct = (pred == y).to_numpy()
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    bin_rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc_b = float(correct[mask].mean())
        conf_b = float(conf[mask].mean())
        w = mask.sum() / len(df)
        ece += w * abs(acc_b - conf_b)
        bin_rows.append({"bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]), "n": int(mask.sum()),
                          "confianza_media": conf_b, "exactitud_media": acc_b})
    return {"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "ece": float(ece), "bins": bin_rows}


def build_external_frame(campaign_dir: Path) -> pd.DataFrame:
    """Agrega windows.csv+verdict.json por corrida, mismas 4 medianas + std
    de utilizacion de memoria que el dataset principal, para las 4 familias
    externas. No usa historical_gpu_runs.py porque ese contrato espera una
    columna run_accepted que no esta en windows.csv de esta campana; el
    accepted real vive en verdict.json por corrida.
    """
    rows = []
    for run_dir in sorted(campaign_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        windows_path = run_dir / "windows.csv"
        verdict_path = run_dir / "verdict.json"
        if not windows_path.exists() or not verdict_path.exists():
            continue
        verdict = json.loads(verdict_path.read_text())
        if not verdict.get("accepted"):
            continue
        with open(windows_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            wrows = [r for r in reader if r.get("quality_status") == "gpu_telemetry"]
        if not wrows:
            continue
        kernel_ref = wrows[0]["kernel_ref"]
        family = derive_kernel_family(kernel_ref)
        if family not in EXTERNAL_FAMILIES:
            continue
        labels = {r.get("phase_label_train") for r in wrows if r.get("phase_label_train")}
        if len(labels) != 1 or next(iter(labels)) not in {"compute_bound", "memory_bound"}:
            continue
        row = {
            "run_id": wrows[0]["run_id"], "kernel_ref": kernel_ref, "kernel_family": family,
            "gpu_freq_level_id": wrows[0].get("gpu_freq_level_id"),
            "phase_label_train": next(iter(labels)), "n_windows": len(wrows),
        }
        ok = True
        for sig in SIGNALS:
            values = np.array([float(r[sig]) for r in wrows if r.get(sig) not in (None, "")])
            if not len(values):
                ok = False
                break
            row[f"{sig}_median"] = float(np.median(values))
            row[f"{sig}_std"] = float(np.std(values))
        if ok:
            rows.append(row)
    if not rows:
        raise ValueError(f"ninguna corrida aceptada de {sorted(EXTERNAL_FAMILIES)} en {campaign_dir}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="dataset historico +A1+A2 (relaxed020)")
    ap.add_argument("--external-campaign-dir", required=True, type=Path,
                     help="directorio de pacca_gpu_dvfs_20260820 (crudo, windows.csv+verdict.json por corrida)")
    ap.add_argument("--model", required=True, type=Path, help="joblib del candidato final ya exportado")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--candidate", default="regresion_log")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = load(args.source, split_rajaperf_cuda=True, extra_features=["gpu_mem_util_pct_std"])
    families = sorted(frame[FAMILY_COL].unique())
    fam_codes = pd.Categorical(frame[FAMILY_COL], categories=families).codes
    print(f"features: {FEATURES}", flush=True)

    # --- correlacion / VIF ---
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    corr = pd.DataFrame(X, columns=FEATURES).corr()
    vif_values = vif(X, FEATURES)
    print("VIF:", vif_values, flush=True)

    # --- LOFO con probabilidades (matriz de confusion + ECE) ---
    proba_df = lofo_with_proba(frame, families, fam_codes, args.candidate, args.seeds)
    proba_df.to_csv(args.out / "lofo_probabilities.csv", index=False)
    diag = confusion_and_ece(proba_df)
    print("Matriz de confusion + ECE:", diag["confusion_matrix"], diag["ece"], flush=True)

    # --- validacion externa real (conjunto sellado) ---
    external = build_external_frame(args.external_campaign_dir)
    external.to_csv(args.out / "external_august_frame.csv", index=False)
    model = joblib.load(args.model)
    Xe = external[FEATURES].to_numpy(dtype=np.float32)
    ye = external[LABEL].eq("memory_bound").to_numpy()
    pred = model.predict(Xe)
    per_family = {}
    for fam, g in external.assign(_pred=pred).groupby("kernel_family"):
        truth = g[LABEL].eq("memory_bound")
        per_family[fam] = float((truth == g["_pred"]).mean())
    ext_accuracy = float((ye == pred).mean())
    ext_report = {
        "n_runs": len(external), "n_families": int(external.kernel_family.nunique()),
        "families": sorted(external.kernel_family.unique().tolist()),
        "class_balance": external[LABEL].value_counts().to_dict(),
        "accuracy": ext_accuracy, "per_family_accuracy": per_family,
    }
    print("Externo (conjunto sellado, nunca visto hoy):", json.dumps(ext_report, indent=1, ensure_ascii=False), flush=True)

    (args.out / "final_rigor_audit.json").write_text(json.dumps({
        "schema": "gpu_final_rigor_audit/1",
        "correlation_matrix": corr.round(3).to_dict(),
        "vif": vif_values,
        "confusion_matrix_and_ece": diag,
        "external_sealed_evaluation": ext_report,
    }, indent=2, ensure_ascii=False))
    print(f"\nescrito en {args.out}/final_rigor_audit.json", flush=True)


if __name__ == "__main__":
    main()
