#!/usr/bin/env python3
"""Compuertas C1/C2/C3: ¿la estrategia de Fase 2 es viable con el dataset
que YA existe? (ARC-178)

Las tres se responden con aritmética sobre windows.csv, sin volver a medir
nada. Corren en la partición `normal` (pacca01), no en paccaA100: los
números ya fueron medidos en la A100 y están escritos en el CSV, así que el
procesador del nodo de análisis no cambia el resultado. La ÚNICA magnitud
que no se puede medir aquí es la latencia de inferencia, porque esa sí es
una afirmación sobre el hardware de despliegue; no se toca en este script.

  C1. ¿`b` (el score continuo de acotamiento) varía DENTRO de una misma
      ejecución, o solo entre kernels? Es la compuerta que decide si hay
      proyecto. La etiqueta BINARIA de Fase 1 era casi constante dentro de
      cada kernel (4.0 % de clase minoritaria de media, cuatro kernels en
      0.0 %), y por eso 10M ventanas aportaban la información de 9
      etiquetas. Si `b` hereda ese defecto, pasar a un target continuo es
      cosmético y hay que replantear, no seguir midiendo.

  C2. ¿alpha varía entre tramos de una misma ejecución? Decide si la
      SEGUNDA salida del modelo (sensibilidad a frecuencia) tiene algo que
      predecir o si es una constante por kernel.

  C3. Con LOKO, ¿un regresor sobre `b` le gana al predictor trivial? Es la
      misma prueba que el clasificador binario NO pasó (F1 0.393 contra
      0.371). Se evalúa por kernel y no por ventana porque las ventanas de
      una corrida están correlacionadas: entrenar y probar con ventanas del
      mismo kernel mide memorización, no generalización.

Nada de lo que hace este script depende de la campaña de fases (job 6420)
ni del bloque de rejilla. Si C1 falla, esas dos campañas están midiendo
para un diseño de modelo que no funciona.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.features import align, targets  # noqa: E402

# Solo las columnas que se usan. Leer las 83 multiplicaría por cuatro el
# tiempo de parseo y la memoria sin aportar nada.
USECOLS = [
    "run_id", "kernel_ref", "phase_label_train", "freq_level_id",
    "freq_khz_observed", "frequency_quality_status", "quality_status",
    "window_index", "delta_t_ns", "delta_instructions",
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio", "ips",
    "running_ratio", "pkg_delta_uj",
    "operational_intensity_uncore_real", "i_ridge_used",
]

FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio", "ips",
    "running_ratio", "freq_khz_observed",
]

# El target se DERIVA de estas: usarlas como entrada sería fuga de etiqueta.
FORBIDDEN = {
    "operational_intensity", "operational_intensity_uncore_real",
    "i_ridge_used", "flops_measured_window", "bytes_moved_window",
    "bytes_moved_uncore_real", "uncore_cas_count_read_interval",
    "uncore_cas_count_write_interval", "phase_label_train",
    "phase_label_uncore_real",
}

RUN_DIR_RE = re.compile(r"^(?P<campaign>.+?)__(?P<kernel>.+)__(?P<level>[^_]+)__rep(?P<rep>\d+)$")


def discover_runs(campaign_dir: Path) -> pd.DataFrame:
    """Indexa las corridas reales de telemetría.

    Excluye `__baseline` (perf_enabled=False, sin telemetría real) y las de
    calibración (`rep00`). El índice de repetición se toma del NOMBRE DEL
    DIRECTORIO y no de la columna `repetition`: esa columna vale 1 en todas
    las filas porque cada corrida se lanzó con repetitions=1, y agrupar por
    ella fusionaría las 10 repeticiones en una sola pseudo-corrida.
    """
    rows = []
    for path in sorted(campaign_dir.iterdir()):
        if not path.is_dir() or path.name.endswith("__baseline"):
            continue
        match = RUN_DIR_RE.match(path.name)
        if not match:
            continue
        rep = int(match.group("rep"))
        if rep == 0:
            continue
        windows = path / "windows.csv"
        if not windows.exists():
            continue
        rows.append({
            "windows_path": str(windows),
            "kernel_ref": match.group("kernel"),
            "freq_level_id": match.group("level"),
            "rep_idx": rep,
        })
    return pd.DataFrame(rows)


def load_kernel(index: pd.DataFrame, kernel: str) -> pd.DataFrame:
    frames = []
    for row in index[index["kernel_ref"] == kernel].itertuples():
        frame = pd.read_csv(row.windows_path, usecols=USECOLS, low_memory=False)
        frame["rep_idx"] = row.rep_idx
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    mask = (
        (df["quality_status"] == "ok")
        & df["frequency_quality_status"].isin(["valid", "not_applicable_native"])
        & df["phase_label_train"].notna()
        & (df["phase_label_train"] != "")
    )
    return df.loc[mask].copy()


def level_frequencies(df: pd.DataFrame) -> dict[str, float]:
    """MHz OBSERVADO por nivel, no el solicitado. Si el hardware no alcanzó
    la frecuencia pedida, alpha debe ajustarse contra lo que realmente
    ocurrió."""
    observed = df.groupby("freq_level_id")["freq_khz_observed"].median() / 1000.0
    return {level: float(mhz) for level, mhz in observed.items() if np.isfinite(mhz)}


# --------------------------------------------------------------------- C1

def gate_c1(per_kernel: dict[str, pd.DataFrame], k: float) -> dict:
    """Descomposición anidada de la varianza de `b`.

    La cantidad decisiva NO es la varianza total sino la fracción que es
    INTRA-CORRIDA (mismo kernel, mismo nivel de frecuencia, misma
    repetición). Esa es la única que representa estructura de fase dentro de
    una ejecución, que es lo que una política DVFS en línea podría explotar.
    La varianza entre kernels no sirve: en despliegue el kernel es fijo.
    """
    ss_total = 0.0
    ss_within_run = 0.0
    n_total = 0
    grand_sum = 0.0
    per_kernel_rows = []

    # Media global en una primera pasada de estadísticos suficientes.
    for df in per_kernel.values():
        b = df["b"].to_numpy(dtype=float)
        b = b[np.isfinite(b)]
        grand_sum += b.sum()
        n_total += b.size
    grand_mean = grand_sum / n_total if n_total else float("nan")

    for kernel, df in per_kernel.items():
        b_all = df["b"].to_numpy(dtype=float)
        b_all = b_all[np.isfinite(b_all)]
        ss_total += float(((b_all - grand_mean) ** 2).sum())

        within_run_sds = []
        within_run_ranges = []
        for _, run in df.groupby(["freq_level_id", "rep_idx"], observed=True):
            b = run["b"].to_numpy(dtype=float)
            b = b[np.isfinite(b)]
            if b.size < 2:
                continue
            ss_within_run += float(((b - b.mean()) ** 2).sum())
            within_run_sds.append(float(b.std(ddof=1)))
            within_run_ranges.append(float(np.percentile(b, 95) - np.percentile(b, 5)))

        # Contraste con la etiqueta BINARIA: fracción de clase minoritaria
        # dentro de la corrida. Es la cifra que hundió a Fase 1.
        minority = []
        for _, run in df.groupby(["freq_level_id", "rep_idx"], observed=True):
            labels = run["phase_label_train"].to_numpy()
            if labels.size == 0:
                continue
            frac = (labels == "memory_bound").mean()
            minority.append(float(min(frac, 1.0 - frac)))

        per_kernel_rows.append({
            "kernel": kernel,
            "n_windows": int(b_all.size),
            "b_mean": float(b_all.mean()),
            "b_sd_overall": float(b_all.std(ddof=1)) if b_all.size > 1 else float("nan"),
            "b_sd_within_run_median": float(np.median(within_run_sds)) if within_run_sds else float("nan"),
            "b_p05_p95_within_run_median": float(np.median(within_run_ranges)) if within_run_ranges else float("nan"),
            "binary_minority_frac_mean": float(np.mean(minority)) if minority else float("nan"),
        })

    return {
        "grand_mean_b": grand_mean,
        "n_windows": n_total,
        "frac_variance_within_run": ss_within_run / ss_total if ss_total else float("nan"),
        "per_kernel": per_kernel_rows,
    }


# --------------------------------------------------------------------- C2

def gate_c2(per_kernel: dict[str, pd.DataFrame], freqs: dict[str, float],
            f_ref_mhz: float, n_bins: int) -> dict:
    """alpha ajustado por TRAMO del programa, no por kernel completo.

    Los tramos se alinean con la coordenada de avance basada en
    INSTRUCCIONES retiradas, que es invariante a la frecuencia (0.34 % de
    desviación en el peor caso, ARC-175): el mismo `progress_bin` es el
    mismo punto del programa en todos los niveles del barrido, cosa que una
    coordenada temporal no garantizaría porque bajar el reloj alarga el
    tiempo.
    """
    rows = []
    for kernel, df in per_kernel.items():
        work = df.copy()
        # `repetition` de align espera el índice real de repetición.
        work["repetition"] = work["rep_idx"]
        work = align.add_instruction_progress(work)
        work = align.assign_progress_bins(work, n_bins=n_bins)
        cells = align.aggregate_cells(work, feature_cols=["ipc"])

        alphas = []
        for (_, _, _bin), group in cells.groupby(["kernel_ref", "repetition", "progress_bin"], observed=True):
            durations = {}
            for _, cell in group.iterrows():
                mhz = freqs.get(cell["freq_level_id"])
                if mhz is None or not np.isfinite(mhz) or mhz <= 0:
                    continue
                durations[mhz] = float(cell["duration_ns"])
            if f_ref_mhz not in durations or len(durations) < 3:
                continue
            try:
                alpha, r2 = align.fit_alpha(durations, f_ref_mhz)
            except ValueError:
                continue
            if np.isfinite(alpha):
                alphas.append((alpha, r2))

        if not alphas:
            rows.append({"kernel": kernel, "n_cells": 0})
            continue
        arr = np.array([a for a, _ in alphas], dtype=float)
        r2s = np.array([r for _, r in alphas], dtype=float)
        rows.append({
            "kernel": kernel,
            "n_cells": int(arr.size),
            "alpha_mean": float(arr.mean()),
            "alpha_sd_across_bins": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "alpha_p05": float(np.percentile(arr, 5)),
            "alpha_p95": float(np.percentile(arr, 95)),
            "alpha_min": float(arr.min()),
            "r2_median": float(np.median(r2s)),
            "frac_cells_below_break_even": float((arr <= 0.226).mean()),
        })
    return {"break_even_alpha_cpu": 0.226, "per_kernel": rows}


# --------------------------------------------------------------------- C3

def gate_c3(per_kernel: dict[str, pd.DataFrame], max_rows_per_run: int, seed: int) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    leak = set(FEATURES) & FORBIDDEN
    if leak:
        raise SystemExit(f"fuga de etiqueta en las features: {sorted(leak)}")

    rng = np.random.default_rng(seed)
    samples = []
    for kernel, df in per_kernel.items():
        for _, run in df.groupby(["freq_level_id", "rep_idx"], observed=True):
            run = run.dropna(subset=[*FEATURES, "b"])
            if run.empty:
                continue
            if len(run) > max_rows_per_run:
                idx = rng.choice(len(run), size=max_rows_per_run, replace=False)
                run = run.iloc[idx]
            samples.append(run[[*FEATURES, "b", "kernel_ref"]])
    data = pd.concat(samples, ignore_index=True)

    folds = []
    for kernel in sorted(data["kernel_ref"].unique()):
        test = data[data["kernel_ref"] == kernel]
        train = data[data["kernel_ref"] != kernel]
        if train.empty or test.empty:
            continue

        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
        model.fit(train[FEATURES], train["b"])
        pred = model.predict(test[FEATURES])

        # Baseline trivial: la media de `b` en entrenamiento. Es lo que
        # predice un modelo que no aprendió nada del kernel de prueba.
        trivial = np.full(len(test), float(train["b"].mean()))

        folds.append({
            "held_out_kernel": kernel,
            "n_test": int(len(test)),
            "mae_model": float(mean_absolute_error(test["b"], pred)),
            "mae_trivial": float(mean_absolute_error(test["b"], trivial)),
            "r2_model": float(r2_score(test["b"], pred)),
        })

    wins = sum(1 for f in folds if f["mae_model"] < f["mae_trivial"])
    return {
        "features": FEATURES,
        "n_rows": int(len(data)),
        "folds": folds,
        "mae_model_mean": float(np.mean([f["mae_model"] for f in folds])),
        "mae_trivial_mean": float(np.mean([f["mae_trivial"] for f in folds])),
        "folds_where_model_wins": f"{wins}/{len(folds)}",
        "worst_kernel": max(folds, key=lambda f: f["mae_model"])["held_out_kernel"] if folds else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--f-ref-mhz", type=float, default=3200.0)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--max-rows-per-run", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    index = discover_runs(campaign_dir)
    if index.empty:
        raise SystemExit(f"ninguna corrida de telemetría bajo {campaign_dir}")
    kernels = sorted(index["kernel_ref"].unique())
    print(f"nodo de análisis: {platform.node()}", flush=True)
    print(f"corridas: {len(index)}  kernels: {len(kernels)}  -> {kernels}", flush=True)

    per_kernel: dict[str, pd.DataFrame] = {}
    for kernel in kernels:
        df = load_kernel(index, kernel)
        if df.empty:
            print(f"  {kernel}: sin ventanas utilizables", flush=True)
            continue
        per_kernel[kernel] = df
        print(f"  {kernel}: {len(df)} ventanas", flush=True)

    combined_oi = pd.concat([d["operational_intensity_uncore_real"] for d in per_kernel.values()])
    combined_ridge = pd.concat([d["i_ridge_used"] for d in per_kernel.values()])
    k = targets.calibrate_k(combined_oi, combined_ridge)
    print(f"k calibrado = {k:.6f}", flush=True)

    for df in per_kernel.values():
        df["b"] = targets.boundedness_score(
            df["operational_intensity_uncore_real"], df["i_ridge_used"], k=k
        )

    # Verifica que `b` es una GENERALIZACIÓN de la etiqueta de Fase 1 y no
    # otra cosa: umbralizado en 0.5 tiene que reproducirla fila por fila. Si
    # esto no da ~1.0, se eligió la columna de OI equivocada y todo lo demás
    # mide otra cosa.
    agree_num = agree_den = 0
    for df in per_kernel.values():
        derived = targets.binary_from_score(df["b"].to_numpy(dtype=float))
        truth = df["phase_label_train"].to_numpy()
        valid = derived != None  # noqa: E711
        agree_num += int((derived[valid] == truth[valid]).sum())
        agree_den += int(valid.sum())
    agreement = agree_num / agree_den if agree_den else float("nan")
    print(f"acuerdo b>0.5 con phase_label_train = {agreement:.6f}", flush=True)

    freqs = level_frequencies(pd.concat(per_kernel.values(), ignore_index=True))
    print(f"MHz observados por nivel: { {k2: round(v, 1) for k2, v in freqs.items()} }", flush=True)

    print("\n== C1 ==", flush=True)
    c1 = gate_c1(per_kernel, k)
    print(json.dumps(c1, indent=2), flush=True)

    print("\n== C2 ==", flush=True)
    c2 = gate_c2(per_kernel, freqs, args.f_ref_mhz, args.n_bins)
    print(json.dumps(c2, indent=2), flush=True)

    print("\n== C3 ==", flush=True)
    c3 = gate_c3(per_kernel, args.max_rows_per_run, args.seed)
    print(json.dumps(c3, indent=2), flush=True)

    report = {
        "analysis_node": platform.node(),
        "campaign_dir": str(campaign_dir),
        "k_calibrated": k,
        "agreement_with_phase1_label": agreement,
        "level_mhz_observed": freqs,
        "c1_variance_of_b": c1,
        "c2_alpha_per_progress_bin": c2,
        "c3_loko_regression": c3,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
