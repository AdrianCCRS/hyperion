"""Audita cercanía al ridge e inestabilidad NVML sin usarlas como features.

La cercanía al ridge explica incertidumbre de la etiqueta, pero no está
disponible en inferencia. Este informe nunca excluye filas ni altera la métrica
del modelo; solo localiza dónde se concentran los errores OOF por familia.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fase2_clasificador.analysis.evaluate_historical_gpu_runs import _family_weights
from fase2_clasificador.eval import protocol

FEATURES = [
    "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_median", "gpu_sm_clock_mhz_median",
    "gpu_util_pct_iqr", "gpu_mem_util_pct_iqr", "gpu_power_mw_iqr", "gpu_sm_clock_mhz_iqr",
]
SIGNALS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw", "gpu_sm_clock_mhz")


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=20260917,
    ))


def _bin_summary(frame: pd.DataFrame, column: str, labels: list[str]) -> list[dict[str, object]]:
    ranked = pd.qcut(frame[column], q=len(labels), labels=labels, duplicates="drop")
    out = frame.assign(bin=ranked).groupby("bin", observed=True).agg(
        n=("error", "size"), error_rate=("error", "mean"), compute_share=("truth_memory", lambda x: 1.0 - x.mean()),
        median_value=(column, "median"),
    ).reset_index()
    return out.to_dict(orient="records")


def run(frame: pd.DataFrame) -> dict[str, object]:
    required = FEATURES + ["operational_intensity", "i_ridge_used", "phase_label_train", "kernel_family"]
    clean = frame.dropna(subset=required).copy().reset_index(drop=True)
    clean = clean[(clean["operational_intensity"] > 0) & (clean["i_ridge_used"] > 0)].copy()
    X = clean[FEATURES].to_numpy(dtype=float)
    y = clean["phase_label_train"].eq("memory_bound").to_numpy()
    pred = np.empty(len(clean), dtype=bool)
    probability = np.empty(len(clean), dtype=float)
    for train_idx, test_idx, _ in protocol.leave_one_kernel_out(clean, kernel_col="kernel_family"):
        model = _model()
        model.fit(X[train_idx], y[train_idx], logisticregression__sample_weight=_family_weights(clean.iloc[train_idx]))
        pred[test_idx] = model.predict(X[test_idx])
        probability[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    clean["truth_memory"] = y
    clean["error"] = pred != y
    clean["confidence"] = np.maximum(probability, 1.0 - probability)
    clean["ridge_margin_log2"] = np.abs(np.log2(clean["operational_intensity"] / clean["i_ridge_used"]))
    relative_iqrs = []
    for signal in SIGNALS:
        relative_iqrs.append(clean[f"{signal}_iqr"] / clean[f"{signal}_median"].abs().clip(lower=1.0))
    clean["nvml_instability"] = pd.concat(relative_iqrs, axis=1).median(axis=1)
    return {
        "schema": "historical_gpu_margin_audit/1",
        "n_runs": int(len(clean)), "n_families": int(clean.kernel_family.nunique()),
        "purpose": "diagnostic_only_no_rows_excluded",
        "error_rate_total": float(clean.error.mean()),
        "by_roofline_margin_quartile": _bin_summary(clean, "ridge_margin_log2", ["Q1_cerca", "Q2", "Q3", "Q4_lejos"]),
        "by_nvml_instability_quartile": _bin_summary(clean, "nvml_instability", ["Q1_estable", "Q2", "Q3", "Q4_inestable"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset, low_memory=False)
    frame = frame[~frame["kernel_family"].isin(args.exclude_family)].copy()
    result = run(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
