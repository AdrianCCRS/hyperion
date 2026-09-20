"""Compara agregados históricos GPU sin mezclar familias entre train y test.

Es un experimento de selección de representación, no reemplaza el entrenamiento
CUPTI. Cada variante se prueba con LOFO y folds mixtos; el JSON conserva todos
los resultados para que no se elija una variante mirando solo su mejor media.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from fase2_clasificador.eval import protocol
from fase2_clasificador.training import model_specs

SIGNALS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw", "gpu_sm_clock_mhz")


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva ratios físicos disponibles durante una observación NVML.

    No incorpora duración, contador de muestras ni IDs: esos valores podrían
    reconocer una carga concreta. Energía/duración solo se usa para recuperar
    potencia media, una magnitud que el daemon puede medir sin conocer kernel.
    """
    out = df.copy()
    missing = pd.Series(np.nan, index=out.index, dtype=float)
    util = pd.to_numeric(out.get("gpu_util_pct_median", missing), errors="coerce").clip(lower=1.0)
    mem = pd.to_numeric(out.get("gpu_mem_util_pct_median", missing), errors="coerce")
    power = pd.to_numeric(out.get("gpu_power_mw_median", missing), errors="coerce")
    clock = pd.to_numeric(out.get("gpu_sm_clock_mhz_median", missing), errors="coerce").clip(lower=1.0)
    temperature = pd.to_numeric(out.get("gpu_temperature_c_median", missing), errors="coerce")
    out["gpu_power_mw_per_mhz_median"] = power / clock
    out["gpu_power_mw_per_util_pct"] = power / util
    out["gpu_mem_util_to_gpu_util"] = mem / util
    out["gpu_mem_util_share"] = mem / (mem + util).clip(lower=1.0)
    out["gpu_util_gap_pct"] = util - mem
    out["gpu_activity_mhz"] = util * clock / 100.0
    out["gpu_mem_activity_mhz"] = mem * clock / 100.0
    out["gpu_power_mw_per_activity_mhz"] = power / out["gpu_activity_mhz"].clip(lower=1.0)
    out["gpu_power_mw_per_mem_activity_mhz"] = power / out["gpu_mem_activity_mhz"].clip(lower=1.0)
    out["gpu_temperature_c_per_power_w"] = temperature / (power / 1000.0).clip(lower=1.0)
    for signal in SIGNALS:
        minimum, maximum = f"{signal}_min", f"{signal}_max"
        if minimum in out and maximum in out:
            out[f"{signal}_range"] = pd.to_numeric(out[maximum], errors="coerce") - pd.to_numeric(out[minimum], errors="coerce")
        iqr, median = f"{signal}_iqr", f"{signal}_median"
        if iqr in out and median in out:
            denominator = pd.to_numeric(out[median], errors="coerce").abs().clip(lower=1.0)
            out[f"{signal}_relative_iqr"] = pd.to_numeric(out[iqr], errors="coerce") / denominator
    if {"gpu_energy_delta_mj_sum", "covered_duration_ns"}.issubset(out.columns):
        energy = pd.to_numeric(out["gpu_energy_delta_mj_sum"], errors="coerce")
        duration = pd.to_numeric(out["covered_duration_ns"], errors="coerce")
        out["gpu_energy_mean_power_w"] = energy * 1_000_000.0 / duration.where(duration > 0)
    cadence = pd.to_numeric(out.get("temporal_cadence_ms_median", missing), errors="coerce").clip(lower=0.001)
    for signal in SIGNALS:
        difference = f"temporal_{signal}_mean_abs_diff"
        if difference in out:
            out[f"temporal_{signal}_mean_abs_rate_per_s"] = (
                pd.to_numeric(out[difference], errors="coerce") * 1000.0 / cadence
            )
    return out


def feature_variants() -> dict[str, list[str]]:
    median = [f"{s}_median" for s in SIGNALS]
    temporal = [
        f"temporal_{signal}_{suffix}" for signal in SIGNALS
        for suffix in ("mean_abs_diff", "change_fraction", "lag1_autocorr", "trend_per_span")
    ] + [
        "temporal_cadence_ms_median", "temporal_cadence_ms_iqr",
        "temporal_gpu_active_fraction", "temporal_gpu_high_fraction",
        "temporal_mem_active_fraction", "temporal_mem_high_fraction",
        "temporal_gpu_burst_count_per_1k", "temporal_gpu_burst_mean_fraction", "temporal_gpu_burst_max_fraction",
        "temporal_mem_burst_count_per_1k", "temporal_mem_burst_mean_fraction", "temporal_mem_burst_max_fraction",
        "temporal_corr_util_mem", "temporal_corr_util_power", "temporal_corr_mem_power",
        "temporal_util_power_best_abs_lag_corr", "temporal_util_power_best_lag",
    ]
    temporal_core = [
        "temporal_gpu_active_fraction", "temporal_gpu_high_fraction",
        "temporal_mem_active_fraction", "temporal_mem_high_fraction",
        "temporal_gpu_burst_count_per_1k", "temporal_gpu_burst_max_fraction",
        "temporal_mem_burst_count_per_1k", "temporal_mem_burst_max_fraction",
        "temporal_corr_util_mem", "temporal_corr_util_power", "temporal_corr_mem_power",
        "temporal_gpu_util_pct_mean_abs_diff", "temporal_gpu_mem_util_pct_mean_abs_diff",
        "temporal_gpu_power_mw_mean_abs_diff",
    ]
    temporal_core_rate = [
        feature.replace("_mean_abs_diff", "_mean_abs_rate_per_s") if feature.endswith("_mean_abs_diff") else feature
        for feature in temporal_core
    ]
    return {
        "median": median,
        "time_weighted_mean": [f"{s}_time_weighted_mean" for s in SIGNALS],
        "median_iqr": median + [f"{s}_iqr" for s in SIGNALS],
        "median_p90": median + [f"{s}_p90" for s in SIGNALS],
        "median_trimmed_mean": median + [f"{s}_trimmed_mean" for s in SIGNALS],
        "median_without_clock": median[:-1],
        "median_power_per_clock": [
            "gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_power_mw_per_mhz_median",
        ],
        "physical_ratios": median + [
            "gpu_power_mw_per_util_pct", "gpu_mem_util_to_gpu_util", "gpu_energy_mean_power_w",
            "gpu_util_pct_range", "gpu_power_mw_range", "gpu_sm_clock_mhz_range",
        ],
        "physics_extended": median + [
            "gpu_mem_util_to_gpu_util", "gpu_mem_util_share", "gpu_util_gap_pct",
            "gpu_activity_mhz", "gpu_mem_activity_mhz",
            "gpu_power_mw_per_activity_mhz", "gpu_power_mw_per_mem_activity_mhz",
            "gpu_util_pct_relative_iqr", "gpu_mem_util_pct_relative_iqr",
            "gpu_power_mw_relative_iqr", "gpu_sm_clock_mhz_relative_iqr",
        ],
        "condition_normalized": median + [
            "gpu_temperature_c_median", "gpu_temperature_c_iqr",
            "gpu_power_mw_per_mhz_median", "gpu_mem_util_to_gpu_util",
            "gpu_power_mw_per_activity_mhz", "gpu_temperature_c_per_power_w",
        ],
        "temporal_dynamics": median + temporal,
        "temporal_dynamics_iqr": median + [f"{s}_iqr" for s in SIGNALS] + temporal,
        "temporal_core": median + temporal_core,
        "temporal_core_iqr": median + [f"{s}_iqr" for s in SIGNALS] + temporal_core,
        "temporal_core_rate": median + temporal_core_rate,
        "full_distribution": median + [
            f"{signal}_{stat}" for signal in SIGNALS
            for stat in ("trimmed_mean", "std", "iqr", "min", "max", "n_distinct")
        ],
    }


def _folds(df: pd.DataFrame, scheme: str, seed: int):
    if scheme == "lofo":
        yield from protocol.leave_one_kernel_out(df, kernel_col="kernel_family")
    else:
        yield from protocol.leave_mixed_families_out(
            df, kernel_col="kernel_family", label_col="phase_label_train", seed=seed
        )


def _family_weights(frame: pd.DataFrame) -> np.ndarray:
    """Da a cada familia el mismo peso total dentro del entrenamiento."""
    counts = frame["kernel_family"].value_counts()
    weights = frame["kernel_family"].map(lambda family: 1.0 / counts[family]).to_numpy(dtype=float)
    return weights / weights.mean()


def _fit(model, X, y, sample_weight: np.ndarray | None = None) -> None:
    if sample_weight is None:
        model.fit(X, y)
    elif hasattr(model, "steps"):  # Pipeline de regresión logística.
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)


def evaluate_variant(
    df: pd.DataFrame, features: list[str], scheme: str, seed: int, model_names: set[str] | None = None,
    family_balanced: bool = False,
) -> list[dict[str, object]]:
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"el dataset agregado no tiene features: {sorted(missing)}")
    clean = df.dropna(subset=features + ["phase_label_train", "kernel_family"]).reset_index(drop=True)
    if len(clean) != len(df):
        raise ValueError("hay features no finitas: corregir el agregado antes de evaluar")
    X = clean[features].to_numpy(dtype=np.float32)
    y = (clean["phase_label_train"] == "memory_bound").to_numpy()
    if len(set(y)) != 2:
        raise ValueError("la evaluación requiere ambas clases")

    prototypes = model_specs.build_models(seed)
    if model_names is not None:
        unknown = model_names - set(prototypes)
        if unknown:
            raise ValueError(f"modelos desconocidos: {sorted(unknown)}")
        prototypes = {name: model for name, model in prototypes.items() if name in model_names}
    all_scores: dict[str, dict[str, float]] = {name: {} for name in prototypes}
    all_confusions: dict[str, np.ndarray] = {name: np.zeros((2, 2), dtype=int) for name in all_scores}
    all_oof: dict[str, list[pd.DataFrame]] = {name: [] for name in all_scores}
    weights: dict[str, float] = {}
    for train_idx, test_idx, fold in _folds(clean, scheme, seed):
        protocol.assert_no_familia_leak(clean, train_idx, test_idx, kernel_col="kernel_family", family_fn=lambda x: x)
        pos = int(y[train_idx].sum())
        neg = int(len(train_idx) - pos)
        scale = neg / pos if pos else 1.0
        weights[fold] = scale
        fold_models = model_specs.build_models(seed, scale)
        for name in all_scores:
            prototype = fold_models[name]
            model = clone(prototype)
            # La evaluación contiene pocas corridas independientes; limitar el
            # paralelismo evita competir con el resto de experimentos locales.
            if "n_jobs" in model.get_params(deep=False):
                model.set_params(n_jobs=1)
            _fit(model, X[train_idx], y[train_idx], _family_weights(clean.iloc[train_idx]) if family_balanced else None)
            pred = model.predict(X[test_idx])
            all_scores[name][fold] = float(f1_score(y[test_idx], pred, labels=[False, True], average="macro", zero_division=0))
            all_confusions[name] += confusion_matrix(y[test_idx], pred, labels=[False, True])
            all_oof[name].append(pd.DataFrame({
                "kernel_family": clean.iloc[test_idx]["kernel_family"].to_numpy(),
                "truth": y[test_idx], "prediction": pred,
            }))

    rows: list[dict[str, object]] = []
    for name, scores in all_scores.items():
        summary = protocol.fold_summary(scores)
        oof = pd.concat(all_oof[name], ignore_index=True)
        per_family_accuracy = oof.groupby("kernel_family")[["truth", "prediction"]].apply(
            lambda group: accuracy_score(group["truth"], group["prediction"])
        )
        rows.append({
            "model": name, "cv_scheme": scheme, "n_runs": int(len(clean)),
            "n_families": int(clean["kernel_family"].nunique()), "features": features,
            "f1_macro_mean": summary["mean"], "f1_macro_std": summary["std"],
            "f1_macro_worst_fold": summary["min"], "worst_fold": summary["worst_kernel"],
            "f1_macro_oof_pooled": float(f1_score(oof["truth"], oof["prediction"], labels=[False, True], average="macro", zero_division=0)),
            "oof_accuracy": float(accuracy_score(oof["truth"], oof["prediction"])),
            "family_accuracy_mean": float(per_family_accuracy.mean()),
            "family_accuracy_worst": float(per_family_accuracy.min()),
            "worst_accuracy_family": str(per_family_accuracy.idxmin()),
            "per_fold": scores, "confusion_matrix": all_confusions[name].tolist(),
            "xgboost_scale_pos_weight_by_fold": weights if name == "xgboost" else None,
            "family_balanced_training": family_balanced,
        })
    return rows


def _fit_score(X, y, train_idx, test_idx, model_name: str, seed: int, train_frame: pd.DataFrame | None = None,
               family_balanced: bool = False) -> float:
    """Entrena una configuración sin reutilizar el pliegue de prueba."""
    pos = int(y[train_idx].sum())
    neg = int(len(train_idx) - pos)
    prototype = model_specs.build_models(seed, (neg / pos) if pos else 1.0)[model_name]
    model = clone(prototype)
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=1)
    _fit(model, X[train_idx], y[train_idx], _family_weights(train_frame.iloc[train_idx]) if family_balanced and train_frame is not None else None)
    return float(f1_score(y[test_idx], model.predict(X[test_idx]), labels=[False, True], average="macro", zero_division=0))


def nested_selection(
    df: pd.DataFrame, variants: dict[str, list[str]], scheme: str, seed: int, model_names: set[str] | None,
    family_balanced: bool = False,
) -> dict[str, object]:
    """Selecciona modelo+agregado dentro de cada fold externo por familia.

    El promedio interno sirve solo para elegir; el F1 devuelto se calcula en
    una familia/grupo que nunca intervino ni en la selección ni en el ajuste.
    """
    features = sorted({feature for values in variants.values() for feature in values})
    clean = df.dropna(subset=features + ["phase_label_train", "kernel_family"]).reset_index(drop=True)
    names = set(model_specs.build_models(seed)) if model_names is None else model_names
    unknown = names - set(model_specs.build_models(seed))
    if unknown:
        raise ValueError(f"modelos desconocidos: {sorted(unknown)}")
    y = (clean["phase_label_train"] == "memory_bound").to_numpy()
    outer_scores: dict[str, float] = {}
    selections: dict[str, dict[str, object]] = {}
    for outer_number, (train_idx, test_idx, fold) in enumerate(_folds(clean, scheme, seed), start=1):
        train = clean.iloc[train_idx].reset_index(drop=True)
        train_y = y[train_idx]
        candidate_scores: list[tuple[float, str, str]] = []
        for variant, variant_features in variants.items():
            X_inner = train[variant_features].to_numpy(dtype=np.float32)
            for name in sorted(names):
                values = [_fit_score(X_inner, train_y, inner_train, inner_test, name, seed, train, family_balanced)
                          for inner_train, inner_test, _ in _folds(train, scheme, seed + outer_number)]
                candidate_scores.append((float(np.mean(values)), variant, name))
        # El orden inverso conserva desempate estable por nombre, nunca por
        # orden accidental de dict o por el F1 del fold externo.
        inner_score, variant, name = max(candidate_scores, key=lambda item: (item[0], item[1], item[2]))
        X_outer = clean[variants[variant]].to_numpy(dtype=np.float32)
        outer_scores[fold] = _fit_score(X_outer, y, train_idx, test_idx, name, seed, clean, family_balanced)
        selections[fold] = {"aggregation_variant": variant, "model": name, "inner_f1_macro_mean": inner_score}
    summary = protocol.fold_summary(outer_scores)
    return {"outer_scheme": scheme, "n_runs": len(clean), "n_families": int(clean["kernel_family"].nunique()),
            "outer_f1_macro_mean": summary["mean"], "outer_f1_macro_std": summary["std"],
            "outer_f1_macro_worst_fold": summary["min"], "worst_fold": summary["worst_kernel"],
            "per_outer_fold": outer_scores, "selection_by_outer_fold": selections,
            "family_balanced_training": family_balanced}


def frequency_sensitivity(
    df: pd.DataFrame, variants: dict[str, list[str]], seed: int, model_names: set[str] | None,
    family_balanced: bool = False,
) -> list[dict[str, object]]:
    """Sensibilidad suplementaria: deja fuera un nivel GPU, no una familia."""
    if "gpu_freq_level_id" not in df:
        raise ValueError("falta gpu_freq_level_id para sensibilidad por frecuencia")
    names = set(model_specs.build_models(seed)) if model_names is None else model_names
    rows = []
    for variant, features in variants.items():
        clean = df.dropna(subset=features + ["phase_label_train", "gpu_freq_level_id"]).reset_index(drop=True)
        X, y = clean[features].to_numpy(dtype=np.float32), (clean["phase_label_train"] == "memory_bound").to_numpy()
        scores = {name: {} for name in names}
        for level in sorted(clean["gpu_freq_level_id"].unique()):
            test = np.flatnonzero(clean["gpu_freq_level_id"].to_numpy() == level)
            train = np.flatnonzero(clean["gpu_freq_level_id"].to_numpy() != level)
            for name in names:
                scores[name][str(level)] = _fit_score(X, y, train, test, name, seed, clean, family_balanced)
        for name, per_level in scores.items():
            summary = protocol.fold_summary(per_level)
            rows.append({"aggregation_variant": variant, "model": name, "n_levels": len(per_level),
                         "f1_macro_mean": summary["mean"], "f1_macro_std": summary["std"],
                         "f1_macro_worst_level": summary["min"], "worst_level": summary["worst_kernel"],
                         "per_level": per_level})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--schemes", default="lofo,mixed-families")
    parser.add_argument("--models", default="all",
                        help="all o lista separada por coma; útil para una exploración rápida")
    parser.add_argument("--nested-selection", action="store_true")
    parser.add_argument("--frequency-sensitivity", action="store_true")
    parser.add_argument("--family-balanced-training", action="store_true")
    parser.add_argument("--exclude-family", action="append", default=[])
    args = parser.parse_args()
    df = add_physical_features(pd.read_csv(args.dataset, low_memory=False))
    if args.exclude_family:
        df = df[~df["kernel_family"].isin(args.exclude_family)].copy()
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    if set(schemes) - {"lofo", "mixed-families"}:
        raise ValueError("--schemes admite lofo,mixed-families")
    model_names = None if args.models == "all" else {x.strip() for x in args.models.split(",") if x.strip()}
    results = []
    available_variants = {
        name: features for name, features in feature_variants().items()
        if set(features).issubset(df.columns)
    }
    if not available_variants:
        raise ValueError("el dataset no contiene ninguna variante de features evaluable")
    for variant, features in available_variants.items():
        for scheme in schemes:
            for row in evaluate_variant(df, features, scheme, args.seed, model_names, args.family_balanced_training):
                row["aggregation_variant"] = variant
                results.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k not in {"per_fold", "confusion_matrix", "features", "xgboost_scale_pos_weight_by_fold"}}
                  for r in results]).sort_values(["cv_scheme", "f1_macro_mean"], ascending=[True, False]).to_csv(
                      args.output_dir / "historical_gpu_variant_summary.csv", index=False)
    (args.output_dir / "historical_gpu_variant_results.json").write_text(
        json.dumps({"schema": "historical_gpu_aggregation_evaluation/1", "dataset": str(args.dataset),
                    "seed": args.seed, "results": results}, indent=2, ensure_ascii=False) + "\n")
    if args.nested_selection:
        nested = [nested_selection(df, available_variants, scheme, args.seed, model_names, args.family_balanced_training) for scheme in schemes]
        (args.output_dir / "historical_gpu_nested_selection.json").write_text(
            json.dumps({"schema": "historical_gpu_nested_selection/1", "results": nested}, indent=2, ensure_ascii=False) + "\n")
    if args.frequency_sensitivity:
        sensitivity = frequency_sensitivity(df, available_variants, args.seed, model_names, args.family_balanced_training)
        pd.DataFrame([{k: v for k, v in row.items() if k != "per_level"} for row in sensitivity]).to_csv(
            args.output_dir / "historical_gpu_frequency_sensitivity.csv", index=False)
        (args.output_dir / "historical_gpu_frequency_sensitivity.json").write_text(
            json.dumps({"schema": "historical_gpu_frequency_sensitivity/1", "results": sensitivity}, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(results)} comparaciones escritas en {args.output_dir}")


if __name__ == "__main__":
    main()
