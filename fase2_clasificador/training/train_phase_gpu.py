"""Entrena y evalúa el clasificador de fase GPU (compute_bound/memory_bound).

Espejo de ``train_phase.py`` (CPU), adaptado a la granularidad GPU: una fila
de ``training_gpu_phases.csv`` es una CORRIDA completa (kernel_ref x
gpu_freq_level_id x repetición), ya agregada con estadísticos robustos sobre
las muestras NVML de esa corrida (F1-GPU-003) -- nunca una muestra NVML
aislada. No hace falta submuestrear por corrida como en CPU: cada corrida ya
aporta exactamente una fila.

FUGA DE ETIQUETA -- igual de importante que en CPU. `phase_label_train` se
deriva de `operational_intensity` (medida offline con `ncu`) contra
`i_ridge_used` (calibración Roofline por precisión/frecuencia). Esas dos
columnas, más `phase_label_hint`/`roofline_calibration_ref`, están
PROHIBIDAS como features.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from fase2_clasificador.eval import protocol  # noqa: E402
from fase1_telemetria.gpu_phases import GPU_PHASE_DATASET_FILENAME  # noqa: E402

# Defaults: la campaña final GPU (18 kernels, F1-GPU-011/012/013) sobre el
# catálogo fusionado -- overridable por CLI (--campaign-dir/--campaign-id).
DEFAULT_CAMPAIGN_DIR = Path.home() / "hyperion-results/final/campaigns/gpu"
DEFAULT_CAMPAIGN_ID = "pacca_gpu_final_20260913"

# Contrato congelado con evidencia real (F1-XDEV-004, ejecutado sobre la
# campaña final GPU de 540 corridas -- ver
# hyperion-results/final/campaigns/gpu/_feature_contract/feature_contract_gpu.json
# en pacca). El plan (§2.5) proponía 5 señales por mediana (incluyendo
# gpu_temperature_c); el análisis de correlación real descartó temperatura:
# sus 8 variantes agregadas quedan con |rho|>0.85 contra gpu_power_mw en
# TODOS los pares -- no aporta información independiente en este catálogo
# (fases demasiado cortas para que la temperatura se desacople de la
# potencia instantánea). Las otras 4 sobreviven con VIF razonable (2.5-6.5).
_SIGNALS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw", "gpu_sm_clock_mhz")
FEATURES = [f"{sig}_median" for sig in _SIGNALS]
LABEL = "phase_label_train"
TRAINING_INPUT_FILENAME = GPU_PHASE_DATASET_FILENAME
TRAINING_GRANULARITY = "gpu_run"

# Columnas prohibidas: la etiqueta se deriva de ellas, o identifican la
# corrida/calibración sin ser una medición física reutilizable como feature.
FORBIDDEN = {
    "operational_intensity", "i_ridge_used", "phase_label_hint",
    "roofline_calibration_ref",
}

READ_COLS = [
    *FEATURES, LABEL, "kernel_ref", "kernel_family", "gpu_freq_level_id",
    "training_eligible", "phase_quality_status", "gpu_frequency_quality_status",
]


def load(
    campaign_dir: Path = DEFAULT_CAMPAIGN_DIR,
    kernels: list[str] | None = None,
    levels: list[str] | None = None,
) -> pd.DataFrame:
    """Carga todas las ``training_gpu_phases.csv`` bajo ``campaign_dir``.

    A diferencia de CPU (que reconstruye la ruta por combinación
    kernel×nivel×repetición porque cada corrida es una carpeta con miles de
    ventanas de 1 ms a submuestrear), aquí cada corrida ya es UNA fila
    agregada -- se listan directamente todos los CSV existentes con
    ``rglob``, sin necesitar la lista completa de niveles/repeticiones de
    antemano ni arriesgar construir una ruta que no coincida con el sufijo
    real del directorio (p. ej. ``__baseline``).
    """
    campaign_dir = Path(campaign_dir)
    paths = sorted(campaign_dir.rglob(TRAINING_INPUT_FILENAME))
    if not paths:
        raise FileNotFoundError(
            f"ningún {TRAINING_INPUT_FILENAME} encontrado bajo {campaign_dir} -- "
            "reprocesa Fase 1 (postprocess.py) sobre la campaña GPU real"
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda c: c in READ_COLS, low_memory=False)
        missing = set(READ_COLS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} no tiene el esquema F1-GPU-003; faltan {sorted(missing)}. "
                "Reprocesa la corrida con la versión que genera training_gpu_phases.csv."
            )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    df = df[
        (df["training_eligible"] == True)  # noqa: E712 -- bool real de pandas, no numpy
        & df[LABEL].notna() & (df[LABEL] != "")
    ]
    if kernels:
        df = df[df["kernel_ref"].isin(kernels)]
    if levels:
        df = df[df["gpu_freq_level_id"].isin(levels)]
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + [LABEL])


def build_models(seed: int, scale_pos_weight: float = 1.0):
    """Configuración de hiperparámetros FIJA de los 7 modelos candidatos
    (§3.2 del plan) -- delega en ``model_specs.build_models`` (compartido
    con ``train_phase.py``/CPU). Se conserva esta función/firma en este
    archivo porque es el respaldo explícito cuando un pliegue no tiene
    familias suficientes para un split interno de búsqueda de
    hiperparámetros (ver ``main()``/``hyperparam_search.py``), y porque los
    tests existentes la invocan directamente.
    """
    from fase2_clasificador.training import model_specs

    return model_specs.build_models(seed, scale_pos_weight=scale_pos_weight)


def measure_latency(model, sample: np.ndarray, repeats: int = 200) -> tuple[float, float, float]:
    """Latencia de inferencia de UNA fase, en microsegundos (p50, p95, p99).

    El loop de GPU decide al inicio de cada fase (§4.1 del plan), no sobre un
    lote -- mismo criterio de medición fila-a-fila que ``train_phase.py``.
    """
    one = sample[:1]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(one)
        timings.append((time.perf_counter() - start) * 1e6)
    return (
        float(np.percentile(timings, 50)),
        float(np.percentile(timings, 95)),
        float(np.percentile(timings, 99)),
    )


def select_best_model(
    results: dict[str, dict[str, float]],
    latencies: dict[str, tuple[float, float, float]],
    latency_weight: float,
) -> str:
    """Idéntico criterio que en CPU (ver train_phase.py::select_best_model):
    Score = (1 - F1_macro_medio) + latency_weight * (p99_us / p99_us_max)."""
    candidates = [name for name in results if name != "mayoritaria"]
    max_p99 = max(latencies[name][2] for name in candidates) or 1.0
    def score(name: str) -> float:
        f1_term = 1.0 - results[name]["mean"]
        latency_term = latencies[name][2] / max_p99
        return f1_term + latency_weight * latency_term
    return min(candidates, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el clasificador de fase GPU compute_bound/"
                    "memory_bound (Objetivo 2), con validación agrupada por "
                    "familia algorítmica y selección del modelo a serializar "
                    "por error de clasificación + latencia de inferencia."
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR,
        help=(f"Directorio de la campaña con {TRAINING_INPUT_FILENAME} de origen "
              f"(default: {DEFAULT_CAMPAIGN_DIR})."),
    )
    parser.add_argument(
        "--levels", default=None,
        help="Lista separada por coma de gpu_freq_level_id a incluir (p.ej. "
             "'REF,F0,F1'). Por defecto, todos los presentes en el dataset.",
    )
    parser.add_argument(
        "--kernels", default=None,
        help="Lista separada por coma para restringir el subconjunto de "
             "kernels. Por defecto, todos los presentes en el dataset.",
    )
    parser.add_argument(
        "--latency-weight", type=float, default=0.2,
        help="Peso de la latencia p99 normalizada frente al error de "
             "clasificación al elegir el modelo a serializar (0 = solo F1, "
             "1 = pesar F1 y latencia por igual). Ver select_best_model().",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Si se da, serializa el modelo elegido (joblib) + metadata.json "
             "en este directorio. Si se omite, solo imprime la comparación "
             "sin guardar nada (modo exploración).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Intentos de Optuna (TPE) por modelo sintonizable y por pliegue "
             "externo, para la búsqueda anidada de hiperparámetros (plan "
             "§3.3 punto 1). 0 desactiva la búsqueda: usa la configuración "
             "fija de build_models() en todos los pliegues -- útil para "
             "iterar rápido, nunca el modo por defecto.",
    )
    args = parser.parse_args()
    kernels = args.kernels.split(",") if args.kernels else None
    levels = args.levels.split(",") if args.levels else None

    from sklearn.base import clone
    from sklearn.metrics import confusion_matrix, f1_score

    from fase2_clasificador.eval import hyperparam_search
    from fase2_clasificador.training import model_specs

    leaking = set(FEATURES) & FORBIDDEN
    if leaking:
        raise SystemExit(f"features con fuga de etiqueta: {sorted(leaking)}")

    df = load(campaign_dir=args.campaign_dir, kernels=kernels, levels=levels)
    # kernel_family ya viene calculada por gpu_phases.py con la misma
    # protocol.derive_kernel_family que usa CPU (no se recalcula aquí).
    familias = sorted(df["kernel_family"].unique())
    print(f"matriz: {len(df):,} corridas | {df['kernel_ref'].nunique()} kernels | {len(familias)} familias algorítmicas")
    print(f"features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"distribución de fase: {dict(df[LABEL].value_counts())}\n")

    if len(familias) < 2:
        raise SystemExit(
            f"hacen falta al menos 2 familias algorítmicas para leave-one-familia-out, hay {len(familias)}"
        )

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = (df[LABEL] == "memory_bound").to_numpy()

    # scale_pos_weight de XGBoost (no tiene class_weight="balanced" como
    # sklearn): razón negativos/positivos sobre TODO el dataset, una sola
    # vez -- no por pliegue, para no complicar la comparación con una
    # ponderación que cambia de fold a fold; es una aproximación razonable,
    # no un ajuste per-fold exacto como el que sklearn hace internamente
    # para los demás modelos con class_weight="balanced".
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    fixed_models = model_specs.build_fixed_models(args.seed)
    tunable = model_specs.tunable_specs(args.seed, scale_pos_weight)
    KERNEL_COL = "kernel_family"
    FOLD_FN = protocol.leave_one_kernel_out  # kernel_family ya es la familia

    results: dict[str, dict[str, float]] = {}
    latencies: dict[str, tuple[float, float, float]] = {}
    per_class_f1: dict[str, dict[str, float]] = {}  # modelo -> {compute_bound, memory_bound} (media entre pliegues)
    confusions: dict[str, np.ndarray] = {}  # modelo -> matriz de confusión acumulada [[TN,FP],[FN,TP]] (True=memory_bound)
    per_fold_by_model: dict[str, dict[str, float]] = {name: {} for name in {**fixed_models, **tunable}}
    per_fold_compute_by_model: dict[str, list[float]] = {name: [] for name in per_fold_by_model}
    per_fold_memory_by_model: dict[str, list[float]] = {name: [] for name in per_fold_by_model}
    cm_by_model: dict[str, np.ndarray] = {name: np.zeros((2, 2), dtype=np.int64) for name in per_fold_by_model}
    best_params_por_pliegue: dict[str, dict[str, dict]] = {name: {} for name in tunable}
    n_search_omitida = 0

    for idx_train, idx_test, familia in FOLD_FN(df, kernel_col=KERNEL_COL):
        df_train_outer = df.iloc[idx_train]

        for name, prototype in fixed_models.items():
            model = clone(prototype)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            per_fold_by_model[name][familia] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
            f1_per_class = f1_score(y[idx_test], pred, average=None, labels=[False, True], zero_division=0)
            per_fold_compute_by_model[name].append(float(f1_per_class[0]))
            per_fold_memory_by_model[name].append(float(f1_per_class[1]))
            cm_by_model[name] += confusion_matrix(y[idx_test], pred, labels=[False, True])
            if familia == familias[0]:
                latencies[name] = measure_latency(model, X[idx_test])

        for name, (build_fn, space_fn) in tunable.items():
            can_search = (
                args.n_trials > 0
                and hyperparam_search.n_groups(df_train_outer, kernel_col=KERNEL_COL, fold_fn=FOLD_FN)
                >= hyperparam_search.MIN_FAMILIAS_PARA_BUSQUEDA
            )
            if can_search:
                best_params, _ = hyperparam_search.search_best_params(
                    build_fn, space_fn, df_train_outer, X[idx_train], y[idx_train],
                    kernel_col=KERNEL_COL, seed=args.seed, n_trials=args.n_trials,
                    fold_fn=FOLD_FN,
                )
            else:
                n_search_omitida += 1
                best_params = dict(model_specs.FALLBACK_PARAMS.get(name, {}))
            best_params_por_pliegue[name][familia] = best_params
            model = build_fn(**best_params)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            per_fold_by_model[name][familia] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
            f1_per_class = f1_score(y[idx_test], pred, average=None, labels=[False, True], zero_division=0)
            per_fold_compute_by_model[name].append(float(f1_per_class[0]))
            per_fold_memory_by_model[name].append(float(f1_per_class[1]))
            cm_by_model[name] += confusion_matrix(y[idx_test], pred, labels=[False, True])
            if familia == familias[0]:
                latencies[name] = measure_latency(model, X[idx_test])

    if n_search_omitida:
        print(
            f"[aviso] búsqueda de hiperparámetros omitida en {n_search_omitida} "
            "combinaciones modelo/pliegue por falta de familias para un split "
            "interno (o --n-trials=0) -- se usó la configuración fija de "
            "build_models() en esos casos, ver best_params_por_pliegue en la "
            "metadata.\n"
        )

    for name in per_fold_by_model:
        results[name] = protocol.fold_summary(per_fold_by_model[name])
        results[name]["_per_fold"] = per_fold_by_model[name]  # type: ignore[assignment]
        per_class_f1[name] = {
            "compute_bound": float(np.mean(per_fold_compute_by_model[name])),
            "memory_bound": float(np.mean(per_fold_memory_by_model[name])),
        }
        confusions[name] = cm_by_model[name]

    print(f"{'modelo':<16}{'F1 macro':>10}{'sd':>8}{'peor':>8}{'familia peor':>28}"
          f"{'F1 comp':>9}{'F1 mem':>9}{'p50 us':>9}{'p95 us':>9}{'p99 us':>9}")
    print("-" * 115)
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        p50, p95, p99 = latencies.get(name, (float("nan"), float("nan"), float("nan")))
        pc = per_class_f1[name]
        print(f"{name:<16}{summary['mean']:>10.3f}{summary['std']:>8.3f}"
              f"{summary['min']:>8.3f}{summary['worst_kernel']:>28}"
              f"{pc['compute_bound']:>9.3f}{pc['memory_bound']:>9.3f}"
              f"{p50:>9.1f}{p95:>9.1f}{p99:>9.1f}")

    print("\n\nMatriz de confusión acumulada por modelo (todas las corridas de todos los pliegues, "
          "filas=real, columnas=predicho; orden [compute_bound, memory_bound]):")
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        cm = confusions[name]
        print(f"  {name}: real=compute_bound -> pred=[{cm[0,0]:>4} compute, {cm[0,1]:>4} memory]  "
              f"| real=memory_bound -> pred=[{cm[1,0]:>4} compute, {cm[1,1]:>4} memory]")

    print("\n\nF1 macro por pliegue (familia excluida del entrenamiento):")
    fold_keys = sorted(next(iter(results.values()))["_per_fold"])  # type: ignore[index]
    print(f"{'modelo':<16}" + "".join(k[:11].rjust(12) for k in fold_keys))
    print("-" * (16 + 12 * len(fold_keys)))
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        row = summary["_per_fold"]  # type: ignore[index]
        print(f"{name:<16}" + "".join(f"{row[k]:>12.3f}" for k in fold_keys))

    best_name = select_best_model(results, latencies, args.latency_weight)
    print(
        f"\n\nModelo elegido para serializar: {best_name!r} "
        f"(F1 macro medio={results[best_name]['mean']:.3f}, "
        f"p99={latencies[best_name][2]:.1f}us, latency_weight={args.latency_weight}) "
        "-- ver select_best_model() para el criterio exacto."
    )

    if args.output_dir is not None:
        import joblib
        from sklearn.base import clone as _clone

        # El modelo a desplegar se reentrena sobre TODO el dataset (ya no
        # queda ninguna familia fuera para estimar generalización -- eso ya
        # lo hizo el leave-one-familia-out de arriba). Si es un modelo
        # sintonizable, sus hiperparámetros finales se buscan una vez más
        # aquí, ahora con las 11 familias completas como split interno (más
        # señal que cualquier pliegue externo individual) -- nunca se
        # reutiliza el hiperparámetro de un solo pliegue para el modelo de
        # producción.
        final_best_params: dict = {}
        if best_name in tunable:
            build_fn, space_fn = tunable[best_name]
            if hyperparam_search.n_groups(df, kernel_col=KERNEL_COL, fold_fn=FOLD_FN) >= hyperparam_search.MIN_FAMILIAS_PARA_BUSQUEDA and args.n_trials > 0:
                final_best_params, _ = hyperparam_search.search_best_params(
                    build_fn, space_fn, df, X, y, kernel_col=KERNEL_COL,
                    seed=args.seed, n_trials=args.n_trials, fold_fn=FOLD_FN,
                )
            else:
                final_best_params = dict(model_specs.FALLBACK_PARAMS.get(best_name, {}))
            final_model = build_fn(**final_best_params)
        else:
            final_model = _clone(fixed_models[best_name])
        final_model.fit(X, y)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.output_dir / f"{best_name}_gpu.joblib"
        joblib.dump(final_model, model_path)

        metadata = {
            "device": "gpu",
            "model_name": best_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "features": FEATURES,
            "label": LABEL,
            "training_granularity": TRAINING_GRANULARITY,
            "training_input_filename": TRAINING_INPUT_FILENAME,
            "feature_aggregation": {sig: "median_of_nvml_samples_in_run" for sig in _SIGNALS},
            "campaign_dir": str(args.campaign_dir),
            "n_runs": int(len(df)),
            "n_kernels": int(df["kernel_ref"].nunique()),
            "n_familias": len(familias),
            "familias": familias,
            "class_balance": {"compute_bound": n_neg, "memory_bound": n_pos},
            "class_weight_strategy": (
                "sklearn class_weight='balanced' (todos salvo mayoritaria/xgboost); "
                "xgboost scale_pos_weight="
                f"{scale_pos_weight:.4f} (razon global neg/pos, no por pliegue)"
            ),
            "latency_weight": args.latency_weight,
            "hyperparameter_search": {
                "method": "optuna_tpe" if best_name in tunable else "n/a (linea base fija)",
                "n_trials": args.n_trials,
                "best_params_final_model": final_best_params,
                "best_params_por_pliegue_externo": best_params_por_pliegue.get(best_name, {}),
            },
            "cv_f1_macro_mean": results[best_name]["mean"],
            "cv_f1_macro_std": results[best_name]["std"],
            "cv_f1_macro_worst_familia": results[best_name]["worst_kernel"],
            "cv_f1_per_class_mean": per_class_f1[best_name],
            "cv_confusion_matrix_pooled": {
                "labels": ["compute_bound", "memory_bound"],
                "matrix": confusions[best_name].tolist(),
                "note": "filas=real, columnas=predicho, sumada sobre todos los pliegues de leave-one-familia-out",
            },
            "latency_us_p50_p95_p99": list(latencies[best_name]),
            "all_models_compared": {
                name: {
                    "f1_macro_mean": r["mean"],
                    "f1_compute_bound_mean": per_class_f1[name]["compute_bound"],
                    "f1_memory_bound_mean": per_class_f1[name]["memory_bound"],
                    "latency_us_p99": latencies[name][2],
                }
                for name, r in results.items()
            },
        }
        metadata_path = args.output_dir / f"{best_name}_gpu.metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"\nModelo serializado en {model_path}")
        print(f"Metadata en {metadata_path}")


if __name__ == "__main__":
    main()
