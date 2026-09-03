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
    "ipc", "mpki", "llc_miss_rate", "stall_mem_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]
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
                frame = pd.read_csv(path, usecols=lambda c: c in READ_COLS, low_memory=False)
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


def build_models(seed: int):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    return {
        # Lineas base obligatorias: si un modelo complejo no les gana, el
        # hallazgo es que las features no bastan.
        "mayoritaria": DummyClassifier(strategy="most_frequent"),
        "arbol_prof1": DecisionTreeClassifier(max_depth=1, random_state=seed),
        "regresion_log": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        # Modelos ligeros del alcance del anteproyecto (§5.2).
        "arbol_prof6": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed),
        # Techo de capacidad (§3.2 del plan de realineación): comparación,
        # no elección por defecto -- solo se prefiere sobre los árboles/RF
        # de arriba si su ganancia de F1 justifica el costo de inferencia
        # mayor, evaluado explícitamente en la selección final de main().
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, random_state=seed,
            eval_metric="logloss",
        ),
    }


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
    args = parser.parse_args()
    kernels = args.kernels.split(",") if args.kernels else None
    levels = args.levels.split(",") if args.levels else None

    from sklearn.base import clone
    from sklearn.metrics import f1_score

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

    results: dict[str, dict[str, float]] = {}
    latencies: dict[str, tuple[float, float, float]] = {}

    for name, prototype in build_models(args.seed).items():
        per_fold: dict[str, float] = {}
        for idx_train, idx_test, familia in protocol.leave_one_familia_out(df):
            protocol.assert_no_familia_leak(df, idx_train, idx_test)
            model = clone(prototype)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            # zero_division=0: un pliegue cuya familia es 100% de una clase
            # (npb_cg es memory puro) no tiene positivos que recuperar en la
            # otra, y eso vale 0, no un error.
            per_fold[familia] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
            if familia == familias[0]:  # una familia fija, para la latencia
                latencies[name] = measure_latency(model, X[idx_test])
        results[name] = protocol.fold_summary(per_fold)
        results[name]["_per_fold"] = per_fold  # type: ignore[assignment]

    print(f"{'modelo':<16}{'F1 macro':>10}{'sd':>8}{'peor':>8}{'familia peor':>28}"
          f"{'p50 us':>9}{'p95 us':>9}{'p99 us':>9}")
    print("-" * 97)
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        p50, p95, p99 = latencies.get(name, (float("nan"), float("nan"), float("nan")))
        print(f"{name:<16}{summary['mean']:>10.3f}{summary['std']:>8.3f}"
              f"{summary['min']:>8.3f}{summary['worst_kernel']:>28}"
              f"{p50:>9.1f}{p95:>9.1f}{p99:>9.1f}")

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
        # ver todo el dataset de entrenamiento disponible.
        final_model = _clone(build_models(args.seed)[best_name])
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
            "latency_weight": args.latency_weight,
            "cv_f1_macro_mean": results[best_name]["mean"],
            "cv_f1_macro_std": results[best_name]["std"],
            "cv_f1_macro_worst_familia": results[best_name]["worst_kernel"],
            "latency_us_p50_p95_p99": list(latencies[best_name]),
            "all_models_compared": {
                name: {"f1_macro_mean": r["mean"], "latency_us_p99": latencies[name][2]}
                for name, r in results.items()
            },
        }
        metadata_path = args.output_dir / f"{best_name}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"\nModelo serializado en {model_path}")
        print(f"Metadata en {metadata_path}")


if __name__ == "__main__":
    main()
