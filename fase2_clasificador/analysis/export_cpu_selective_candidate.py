"""Entrena y exporta el candidato CPU selectivo sin búsqueda Optuna."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fase2_clasificador.analysis.cpu_feature_strategies import production_variants
from fase2_clasificador.analysis.evaluate_cpu_feature_strategies import _fit, _prepare, _prototype, _weights
from fase2_clasificador.analysis.evaluate_cpu_selective_nested import _inner_choice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-per-family-class", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fija el umbral operativo en lugar del elegido por el LOFO interno (se registra el elegido)")
    args = parser.parse_args()
    variant = "baseline_pmu"  # vector final: las seis tasas base
    features = production_variants()[variant]
    frame = _prepare(pd.read_csv(args.dataset, low_memory=False), args.max_per_family_class, args.seed)
    frame = frame.dropna(subset=features).reset_index(drop=True)
    # El umbral se selecciona mediante familias retenidas. La variante está
    # fijada por la hipótesis de producción, no se consulta la prueba final.
    _, selected_threshold, inner_score, inner_coverage = _inner_choice(
        frame, {variant: features}, args.seed, args.n_jobs, args.min_coverage
    )
    threshold = args.threshold if args.threshold is not None else selected_threshold
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame["_class"].to_numpy(dtype=bool)
    model = _prototype("xgboost", args.seed, y, args.n_jobs, scale_pos_weight=1.0)
    _fit(model, X, y, _weights(frame))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "cpu_xgboost_baseline_pmu.joblib"
    metadata_path = args.output_dir / "cpu_xgboost_baseline_pmu.metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps({
        "schema": "cpu_selective_candidate/2", "variant": variant, "model": "xgboost",
        "features": features, "frequency_feature": "freq_khz_observed",
        "quality_only_features": ["delta_enabled_ns", "delta_running_ns", "running_ratio"],
        "training_weight": "equal total weight per observed family×class cell",
        "xgboost_scale_pos_weight": 1.0,
        "decision": {"threshold": threshold, "automatic": "confidence >= threshold", "otherwise": "revisar"},
        "threshold_selection": {"scheme": "LOFO interno", "criterion": "exactitud balanceada por celda familia x clase sobre lo decidido",
                                "cell_balanced_accuracy": inner_score,
                                "coverage_mean": inner_coverage, "minimum_coverage": args.min_coverage,
                                "selected_threshold": selected_threshold, "threshold_fixed_by_user": args.threshold is not None},
        "n_rows": len(frame), "n_families": int(frame.kernel_family.nunique()),
        "source": str(args.dataset), "seed": args.seed,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"modelo={model_path} umbral={threshold:.2f} filas={len(frame)}")


if __name__ == "__main__":
    main()
