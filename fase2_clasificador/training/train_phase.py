"""Entrena y evalúa el clasificador de fase (Opción D del plan).

Es el entregable que el anteproyecto promete en §5.2: clasificación
supervisada de la fase de ejecución a partir de telemetría, comparando
modelos ligeros por métricas de clasificación y latencia de inferencia.

FUGA DE ETIQUETA -- lo más importante de este archivo. La etiqueta de
Fase 1 se calcula como ``memory_bound if operational_intensity < i_ridge_used``.
Por tanto ``operational_intensity``, ``i_ridge_used`` y todo lo que entra en
su cálculo (``flops_measured_window``, ``bytes_moved_uncore_real``, los
contadores ``uncore_cas_count_*``) están PROHIBIDOS como features: un modelo
que los reciba no aprende nada, solo vuelve a aplicar el umbral, y sacaría
~100% sin ningún valor.

El punto del clasificador es precisamente inferir el régimen a partir de
contadores baratos y siempre disponibles, sin necesitar la instrumentación
de uncore ni la medición de FLOPS que hacen falta para calcular la
intensidad operacional.
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

# Defaults del dataset original (9 kernels, campaña CPU final ARC-174) --
# todos overridables por CLI (--campaign-dir/--campaign-id/--kernels), para
# no tener que editar este archivo cuando cambie la campaña de origen (p.
# ej. tras correr el catálogo ampliado de 232 kernels fusionado en
# fase1_telemetria/catalog/catalog.yaml).
DEFAULT_CAMPAIGN_DIR = Path.home() / "hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174"
DEFAULT_CAMPAIGN_ID = "pacca_cpu_final_attempt03_20260820"
DEFAULT_KERNELS = [
    "npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu",
    "dgemm_n2048", "rodinia_lavamd_omp", "rajaperf_polybench_3mm_omp",
]
DEFAULT_LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]

# Contadores baratos, disponibles en ejecución sin uncore ni medición de
# FLOPS. freq_khz_observed va incluida a propósito: el ridge se mueve con la
# frecuencia (ARC-175), así que los mismos contadores significan cosas
# distintas según a qué reloj se observen, y el modelo necesita ese contexto.
FEATURES = [
    # F1-CPU-003: `cache_miss_rate` = delta_cache_misses / delta_cache_references
    # (eventos genéricos PERF_COUNT_HW_CACHE_*). Antes se llamaba
    # `llc_miss_rate`, nombre que afirmaba semántica de último nivel sin
    # evidencia para el Ice Lake-SP de paccaA100. `load()` acepta CSV
    # históricos con el nombre viejo (renombra al leer).
    "ipc", "mpki", "cache_miss_rate", "stall_mem_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]
# F1-CPU-003: nombre legado -> nombre vigente, aplicado al leer un CSV que aún
# traiga la columna vieja. No se produce nunca la columna vieja; no coexisten.
LEGACY_COLUMN_RENAMES = {"llc_miss_rate": "cache_miss_rate"}
LABEL = "phase_label_train"
TRAINING_INPUT_FILENAME = "training_cpu_intervals.csv"
TRAINING_GRANULARITY = "uncore_interval"

# Columnas prohibidas: la etiqueta se deriva de ellas.
FORBIDDEN = {
    "operational_intensity", "operational_intensity_uncore_real",
    "i_ridge_used", "flops_measured_window", "bytes_moved_window",
    "bytes_moved_uncore_real", "uncore_cas_count_read_interval",
    "uncore_cas_count_write_interval", "phase_label_uncore_real",
    "phase_label_hint",
}

READ_COLS = [
    *FEATURES, LABEL, "kernel_ref", "freq_level_id",
    "training_quality_status", "frequency_quality_status",
]


def load(
    per_run_sample: int,
    seed: int,
    kernels: list[str] | None = None,
    campaign_dir: Path = DEFAULT_CAMPAIGN_DIR,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    levels: list[str] | None = None,
) -> pd.DataFrame:
    """Carga la matriz F1-CPU-002, submuestreando por corrida.

    El submuestreo es por CORRIDA y no global para que ningún kernel ni
    nivel de frecuencia domine la matriz por el simple hecho de haber
    producido más intervalos. Cada fila ya representa un intervalo uncore:
    ``windows.csv`` es una traza de auditoría a ~1 ms y deliberadamente no
    es una entrada válida para este entrenador.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for kernel in (kernels or DEFAULT_KERNELS):
        for level in (levels or DEFAULT_LEVELS):
            for rep in range(1, 11):
                path = (
                    campaign_dir / f"{campaign_id}__{kernel}__{level}__rep{rep:02d}"
                    / TRAINING_INPUT_FILENAME
                )
                if not path.exists():
                    continue
                frame = pd.read_csv(
                    path,
                    usecols=lambda c: c in READ_COLS or c in LEGACY_COLUMN_RENAMES,
                    low_memory=False,
                )
                # F1-CPU-003: acepta un CSV histórico con `llc_miss_rate` y lo
                # renombra a `cache_miss_rate` antes de validar el esquema.
                frame = frame.rename(columns={
                    old: new for old, new in LEGACY_COLUMN_RENAMES.items()
                    if old in frame.columns and new not in frame.columns
                })
                missing = set(READ_COLS) - set(frame.columns)
                if missing:
                    raise ValueError(
                        f"{path} no tiene el esquema F1-CPU-002; faltan {sorted(missing)}. "
                        "Reprocesa la corrida con la versión que genera training_cpu_intervals.csv."
                    )
                frame = frame[
                    (frame["training_quality_status"] == "ok")
                    & frame["frequency_quality_status"].isin(["valid", "not_applicable_native"])
                    & frame[LABEL].notna()
                    & (frame[LABEL] != "")
                ]
                if len(frame) > per_run_sample:
                    take = rng.choice(len(frame), per_run_sample, replace=False)
                    frame = frame.iloc[np.sort(take)]
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"ningún {TRAINING_INPUT_FILENAME} encontrado bajo {campaign_dir} para "
            f"campaign_id={campaign_id!r}; reprocesa Fase 1 con F1-CPU-002 "
            f"-- revisa --campaign-dir/--campaign-id/--kernels"
        )
    df = pd.concat(frames, ignore_index=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + [LABEL])


def build_models(seed: int, scale_pos_weight: float = 1.0):
    """Configuración de hiperparámetros FIJA de los 7 modelos candidatos
    (§3.2 del plan) -- delega en ``model_specs.build_models`` (compartido
    con ``train_phase_gpu.py``/GPU). Se conserva esta función/firma en este
    archivo porque es el respaldo explícito cuando un pliegue no tiene
    familias suficientes para un split interno de búsqueda de
    hiperparámetros (ver ``main()``/``hyperparam_search.py``), y porque los
    tests existentes la invocan directamente.
    """
    from fase2_clasificador.training import model_specs

    return model_specs.build_models(seed, scale_pos_weight=scale_pos_weight)


def measure_latency(model, sample: np.ndarray, repeats: int = 200) -> tuple[float, float, float]:
    """Latencia de inferencia de UNA ventana, en microsegundos (p50, p95, p99).

    Se mide fila a fila, no en lote: el agente de Fase 3 decide sobre la
    ventana que acaba de cerrar, así que el número relevante es el de una
    predicción aislada, no el rendimiento amortizado de un batch. p95 se
    reporta junto a p99 (§3.3 del plan de realineación) porque el daemon
    necesita un peor caso acotado para decidir la cadencia de muestreo
    viable, y p99 solo, con pocas repeticiones, puede ser un único outlier.
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
    """Elige el modelo a serializar combinando error de clasificación y
    latencia de inferencia (§3.3 punto 5 del plan de realineación): nunca
    por exactitud/F1 sola, porque un modelo más preciso que introduce
    latencia inaceptable compromete el objetivo de "no degradar el
    rendimiento global" (pregunta de investigación del plan).

    Score = (1 - F1_macro_medio) + latency_weight * (p99_us / p99_us_maximo
    entre candidatos). Ambos términos quedan en [0, 1] antes de ponderar, así
    que `latency_weight` es directamente interpretable: 0 = elegir solo por
    F1 (equivalente a ignorar la latencia), 1 = pesar el error de
    clasificación y la latencia por igual. Es un punto de partida
    documentado, no una fórmula validada empíricamente -- reportar la
    sensibilidad a `latency_weight` en el capítulo de resultados si se
    usa un valor distinto del default.
    """
    candidates = [name for name in results if name != "mayoritaria"]
    max_p99 = max(latencies[name][2] for name in candidates) or 1.0
    def score(name: str) -> float:
        f1_term = 1.0 - results[name]["mean"]
        latency_term = latencies[name][2] / max_p99
        return f1_term + latency_weight * latency_term
    return min(candidates, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el clasificador de fase compute_bound/"
                    "memory_bound (Objetivo 2), con validación agrupada por "
                    "familia algorítmica y selección del modelo a serializar "
                    "por error de clasificación + latencia de inferencia."
    )
    parser.add_argument("--per-run-sample", type=int, default=2000,
        help="Máximo de intervalos uncore a submuestrear por corrida individual.")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR,
        help=(f"Directorio de la campaña con {TRAINING_INPUT_FILENAME} de origen "
              f"(default: {DEFAULT_CAMPAIGN_DIR})."),
    )
    parser.add_argument(
        "--campaign-id", default=DEFAULT_CAMPAIGN_ID,
        help="campaign_id usado para construir el nombre de cada subdirectorio de corrida.",
    )
    parser.add_argument(
        "--levels", default=None,
        help="Lista separada por coma de niveles de frecuencia a incluir "
             f"(default: {','.join(DEFAULT_LEVELS)}).",
    )
    parser.add_argument(
        "--kernels", default=None,
        help="Lista separada por coma para restringir el subconjunto de "
             "kernels (p.ej. 'npb_lu,npb_bt,rajaperf_polybench_3mm_omp'). "
             f"Por defecto, los {len(DEFAULT_KERNELS)} del dataset original.",
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

    df = load(
        args.per_run_sample, args.seed, kernels=kernels,
        campaign_dir=args.campaign_dir, campaign_id=args.campaign_id, levels=levels,
    )
    familias = sorted(df["kernel_ref"].map(protocol.derive_kernel_family).unique())
    print(f"matriz: {len(df):,} intervalos uncore | {df['kernel_ref'].nunique()} kernels | {len(familias)} familias algorítmicas")
    print(f"features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"distribución de fase: {dict(df[LABEL].value_counts())}\n")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = (df[LABEL] == "memory_bound").to_numpy()

    # scale_pos_weight de XGBoost (no tiene class_weight="balanced" como
    # sklearn): razón negativos/positivos sobre TODO el dataset, solo para
    # reportar el balance de clase global en la metadata -- el peso real
    # que usa XGBoost en cada pliegue se recalcula por pliegue más abajo,
    # sobre y[idx_train], igual que class_weight="balanced" hace
    # internamente para los demás modelos (mismo fix aplicado en
    # train_phase_gpu.py, ver recordatorios/mejoras_metodologia_clasificador_gpu.md).
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight_global = (n_neg / n_pos) if n_pos > 0 else 1.0

    fixed_models = model_specs.build_fixed_models(args.seed)
    KERNEL_COL = "kernel_ref"
    FOLD_FN = protocol.leave_one_familia_out  # familia derivada de kernel_ref

    tunable_names = list(model_specs.tunable_specs(args.seed, scale_pos_weight_global).keys())
    results: dict[str, dict[str, float]] = {}
    latencies: dict[str, tuple[float, float, float]] = {}
    per_class_f1: dict[str, dict[str, float]] = {}  # modelo -> {compute_bound, memory_bound} (media entre pliegues)
    confusions: dict[str, np.ndarray] = {}  # modelo -> matriz de confusión acumulada [[TN,FP],[FN,TP]] (True=memory_bound)
    per_fold_by_model: dict[str, dict[str, float]] = {name: {} for name in set(fixed_models) | set(tunable_names)}
    per_fold_compute_by_model: dict[str, list[float]] = {name: [] for name in per_fold_by_model}
    per_fold_memory_by_model: dict[str, list[float]] = {name: [] for name in per_fold_by_model}
    cm_by_model: dict[str, np.ndarray] = {name: np.zeros((2, 2), dtype=np.int64) for name in per_fold_by_model}
    best_params_por_pliegue: dict[str, dict[str, dict]] = {name: {} for name in tunable_names}
    scale_pos_weight_por_pliegue: dict[str, float] = {}
    n_search_omitida = 0

    for idx_train, idx_test, familia in FOLD_FN(df, kernel_col=KERNEL_COL):
        protocol.assert_no_familia_leak(df, idx_train, idx_test)
        df_train_outer = df.iloc[idx_train]

        y_train_fold = y[idx_train]
        n_pos_fold = int(y_train_fold.sum())
        n_neg_fold = int(len(y_train_fold) - n_pos_fold)
        scale_pos_weight_fold = (n_neg_fold / n_pos_fold) if n_pos_fold > 0 else 1.0
        scale_pos_weight_por_pliegue[familia] = scale_pos_weight_fold
        tunable = model_specs.tunable_specs(args.seed, scale_pos_weight_fold)

        for name, prototype in fixed_models.items():
            model = clone(prototype)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            # zero_division=0: un pliegue cuya familia es 100% de una clase
            # (npb_cg es memory puro) no tiene positivos que recuperar en la
            # otra, y eso vale 0, no un error.
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

    print("\n\nMatriz de confusión acumulada por modelo (todas las ventanas de todos los pliegues, "
          "filas=real, columnas=predicho; orden [compute_bound, memory_bound]):")
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        cm = confusions[name]
        print(f"  {name}: real=compute_bound -> pred=[{cm[0,0]:>8} compute, {cm[0,1]:>8} memory]  "
              f"| real=memory_bound -> pred=[{cm[1,0]:>8} compute, {cm[1,1]:>8} memory]")

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

        # Reentrena sobre TODOS los datos disponibles (no solo el último
        # pliegue de LOKO/LOFO) -- los pliegues de arriba son solo para
        # estimar generalización; el modelo que se sirve en producción debe
        # ver todo el dataset de entrenamiento disponible. Si es un modelo
        # sintonizable, sus hiperparámetros finales se buscan una vez más
        # aquí, ahora con todas las familias completas como split interno --
        # nunca se reutiliza el hiperparámetro de un solo pliegue externo
        # para el modelo de producción.
        # El modelo final se entrena sobre TODO el dataset, así que su
        # scale_pos_weight es el global -- no queda ningún pliegue que
        # aporte una razón distinta, a diferencia del bucle de arriba.
        tunable_final = model_specs.tunable_specs(args.seed, scale_pos_weight_global)
        final_best_params: dict = {}
        if best_name in tunable_final:
            build_fn, space_fn = tunable_final[best_name]
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
        model_path = args.output_dir / f"{best_name}.joblib"
        joblib.dump(final_model, model_path)

        metadata = {
            "model_name": best_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "features": FEATURES,
            "label": LABEL,
            "training_granularity": TRAINING_GRANULARITY,
            "training_input_filename": TRAINING_INPUT_FILENAME,
            "feature_aggregation": {
                "counter_rates": "recomputed_from_interval_delta_sums",
                "freq_khz_observed": "median_of_covered_cpu_windows",
            },
            "campaign_dir": str(args.campaign_dir),
            "campaign_id": args.campaign_id,
            "n_windows": int(len(df)),
            "n_kernels": int(df["kernel_ref"].nunique()),
            "n_familias": len(familias),
            "familias": familias,
            "class_balance": {"compute_bound": n_neg, "memory_bound": n_pos},
            "class_weight_strategy": (
                "sklearn class_weight='balanced' (todos salvo mayoritaria/xgboost); "
                "xgboost scale_pos_weight recalculado por pliegue externo "
                f"(rango observado: {min(scale_pos_weight_por_pliegue.values()):.4f}-"
                f"{max(scale_pos_weight_por_pliegue.values()):.4f}; modelo final sobre "
                f"el dataset completo usa el global={scale_pos_weight_global:.4f})"
            ),
            "xgboost_scale_pos_weight_por_pliegue": scale_pos_weight_por_pliegue,
            "latency_weight": args.latency_weight,
            "hyperparameter_search": {
                "method": "optuna_tpe" if best_name in tunable_final else "n/a (linea base fija)",
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
        metadata_path = args.output_dir / f"{best_name}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"\nModelo serializado en {model_path}")
        print(f"Metadata en {metadata_path}")


if __name__ == "__main__":
    main()
