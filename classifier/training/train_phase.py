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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "hyperion"))
from classifier.eval import protocol  # noqa: E402

BASE = Path.home() / "hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174"
CID = "pacca_cpu_final_attempt03_20260820"
KERNELS = [
    "npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu",
    "dgemm_n2048", "rodinia_lavamd_omp", "rajaperf_polybench_3mm_omp",
]
LEVELS = ["REF", "F0", "F1", "F2", "F3", "F4"]

# Contadores baratos, disponibles en ejecución sin uncore ni medición de
# FLOPS. freq_khz_observed va incluida a propósito: el ridge se mueve con la
# frecuencia (ARC-175), así que los mismos contadores significan cosas
# distintas según a qué reloj se observen, y el modelo necesita ese contexto.
FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]
LABEL = "phase_label_train"

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
    "quality_status", "frequency_quality_status",
]


def load(per_run_sample: int, seed: int) -> pd.DataFrame:
    """Carga la matriz, submuestreando por corrida.

    El submuestreo es por CORRIDA y no global para que ningún kernel ni
    nivel de frecuencia domine la matriz por el simple hecho de haber
    producido más ventanas (los niveles lentos generan hasta 4x más).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for kernel in KERNELS:
        for level in LEVELS:
            for rep in range(1, 11):
                path = BASE / f"{CID}__{kernel}__{level}__rep{rep:02d}" / "windows.csv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path, usecols=lambda c: c in READ_COLS, low_memory=False)
                frame = frame[
                    (frame["quality_status"] == "ok")
                    & frame["frequency_quality_status"].isin(["valid", "not_applicable_native"])
                    & frame[LABEL].notna()
                    & (frame[LABEL] != "")
                ]
                if len(frame) > per_run_sample:
                    take = rng.choice(len(frame), per_run_sample, replace=False)
                    frame = frame.iloc[np.sort(take)]
                frames.append(frame)
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
    }


def measure_latency(model, sample: np.ndarray, repeats: int = 200) -> tuple[float, float]:
    """Latencia de inferencia de UNA ventana, en microsegundos (p50 y p99).

    Se mide fila a fila, no en lote: el agente de Fase 3 decide sobre la
    ventana que acaba de cerrar, así que el número relevante es el de una
    predicción aislada, no el rendimiento amortizado de un batch.
    """
    one = sample[:1]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(one)
        timings.append((time.perf_counter() - start) * 1e6)
    return float(np.percentile(timings, 50)), float(np.percentile(timings, 99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-run-sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    from sklearn.metrics import f1_score

    leaking = set(FEATURES) & FORBIDDEN
    if leaking:
        raise SystemExit(f"features con fuga de etiqueta: {sorted(leaking)}")

    df = load(args.per_run_sample, args.seed)
    print(f"matriz: {len(df):,} ventanas | {df['kernel_ref'].nunique()} kernels")
    print(f"features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"distribución de fase: {dict(df[LABEL].value_counts())}\n")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = (df[LABEL] == "memory_bound").to_numpy()

    results: dict[str, dict[str, float]] = {}
    latencies: dict[str, tuple[float, float]] = {}

    for name, prototype in build_models(args.seed).items():
        from sklearn.base import clone
        per_fold: dict[str, float] = {}
        for idx_train, idx_test, kernel in protocol.leave_one_kernel_out(df):
            protocol.assert_no_kernel_leak(df, idx_train, idx_test)
            model = clone(prototype)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            # zero_division=0: un pliegue cuyo kernel es 100% de una clase
            # (npb_cg es memory puro) no tiene positivos que recuperar en la
            # otra, y eso vale 0, no un error.
            per_fold[kernel] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
            if kernel == "npb_bt":  # un pliegue cualquiera, para la latencia
                latencies[name] = measure_latency(model, X[idx_test])
        results[name] = protocol.fold_summary(per_fold)
        results[name]["_per_fold"] = per_fold  # type: ignore[assignment]

    print(f"{'modelo':<16}{'F1 macro':>10}{'sd':>8}{'peor':>8}{'kernel peor':>28}"
          f"{'p50 us':>9}{'p99 us':>9}")
    print("-" * 88)
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        p50, p99 = latencies.get(name, (float("nan"), float("nan")))
        print(f"{name:<16}{summary['mean']:>10.3f}{summary['std']:>8.3f}"
              f"{summary['min']:>8.3f}{summary['worst_kernel']:>28}"
              f"{p50:>9.1f}{p99:>9.1f}")

    print("\n\nF1 macro por pliegue (kernel excluido del entrenamiento):")
    kernels = sorted(next(iter(results.values()))["_per_fold"])  # type: ignore[index]
    print(f"{'modelo':<16}" + "".join(k[:11].rjust(12) for k in kernels))
    print("-" * (16 + 12 * len(kernels)))
    for name, summary in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        row = summary["_per_fold"]  # type: ignore[index]
        print(f"{name:<16}" + "".join(f"{row[k]:>12.3f}" for k in kernels))


if __name__ == "__main__":
    main()
