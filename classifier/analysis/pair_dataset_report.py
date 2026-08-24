"""Primer ejercicio real del pipeline por-par (§4 de ambas Estrategias)
sobre los datos ya exportados en `local_datasets/final_campaigns_20260821/`
-- sin nodo, sin red.

Construye, para CPU y GPU, el dataset por (carga, nivel) y reporta la
línea base honesta V6/C7: EDP loss de la mejor frecuencia constante única
elegida SOLO con folds de entrenamiento en cada pliegue LOKO, contra el
oráculo por kernel. Es el número que decide si vale la pena seguir
invirtiendo en un modelo aprendido antes de tener el dataset final de GPU
(6471/6472/6477, aún en cola).

ADVERTENCIA DE CALIDAD DE DATO, encontrada al escribir este script.
`cpu_windows.csv.gz` trae la columna `repetition` constante en 1 para las
540 corridas -- no refleja las hasta 10 repeticiones reales de la campaña,
que sí están codificadas en el sufijo `__repNN` de `run_id`. Usar la
columna tal cual habría COLAPSADO EN SILENCIO las 10 repeticiones de cada
(kernel, nivel) en una sola fila agregada, exactamente el tipo de error
que el proyecto se comprometió a no cometer. Este script deriva la
repetición real de `run_id` en vez de confiar en la columna, y lo declara
aquí para que quede trazable -- pendiente de decidir si se corrige
`local_datasets/.staging/export_final_campaigns.py` o el origen en
`orchestrator/postprocess.py` (ARC-174 reprocesó esta campaña; no se
verificó todavía en cuál de los dos pasos se perdió el valor real).

DATASET GPU: es `pacca_gpu_dvfs_20260820` (8 kernels, incluye
`rodinia_lud`), la campaña ANTERIOR a la corrección de los Anexos K/L/M y
a la campaña de núcleo activo de los jobs 6462/6463. Se reporta igual como
ejercicio del pipeline, excluyendo `rodinia_lud` (GPU en reposo, sección
3.14 del libro) y con la salvedad explícita de que no es el dataset final.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.eval import protocol
from classifier.features import pair_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "local_datasets/final_campaigns_20260821"

# rodinia_lud: GPU en reposo (alpha=0.030, r2 negativo -- sec. 3.14 del
# libro), no es un sujeto válido de DVFS. rodinia_backprop: energia de GPU
# en F0 de solo 8-49 J, dos ordenes de magnitud bajo el resto (Anexo M) --
# su EDP es ruido dividido por casi nada.
GPU_EXCLUDE = {"rodinia_lud", "rodinia_backprop"}


def _cpu_repetition_from_run_id(run_id: pd.Series) -> pd.Series:
    extracted = run_id.str.extract(r"__rep(\d+)$")[0]
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def _edp_by_level(pairs: pd.DataFrame, ref_level: str) -> pd.DataFrame:
    """EDP relativo (energy_ratio * time_ratio) promediado por repeticion,
    con una columna adicional para el propio nivel de referencia (EDP=1
    por construccion). Una fila por kernel, una columna por nivel."""
    pairs = pairs.copy()
    pairs["edp_ratio"] = pairs["energy_ratio"] * pairs["time_ratio"]
    per_kernel_level = (
        pairs.groupby(["kernel_ref", "freq_level_id"])["edp_ratio"]
        .mean().reset_index()
    )
    pivot = per_kernel_level.pivot(index="kernel_ref", columns="freq_level_id", values="edp_ratio")
    pivot[ref_level] = 1.0
    return pivot.dropna()


def report_cpu() -> None:
    print("=" * 78)
    print("CPU -- pacca_cpu_final_attempt03_20260820 (424/540 validas, arc174)")
    print("=" * 78)

    df = pd.read_csv(DATA_DIR / "cpu_windows.csv.gz")
    df["repetition"] = _cpu_repetition_from_run_id(df["run_id"])
    if df["repetition"].isna().any():
        n_bad = int(df["repetition"].isna().sum())
        print(f"AVISO: {n_bad} filas sin repeticion extraible de run_id, se descartan")
        df = df.dropna(subset=["repetition"])

    runs = pair_dataset.aggregate_cpu_runs(df)
    pairs = pair_dataset.build_pair_dataset(runs, pair_dataset.CPU_FEATURES, ref_level="F0")
    print(f"corridas agregadas: {len(runs)}  pares construidos: {len(pairs)}  "
          f"descartados sin referencia: {pairs.attrs['dropped_no_ref']}")

    edp = _edp_by_level(pairs, ref_level="F0")
    print(f"kernels con EDP completo en todos los niveles: {len(edp)}")
    print(edp.round(3).to_string())

    baselines = protocol.trivial_baselines(edp, max_level="F0")
    honest = protocol.honest_constant_baseline(edp)
    print(f"\nsiempre F0 (maxima)   EDP loss = {baselines['siempre_maxima']:.4f}")
    print(f"constante HONESTA     EDP loss = {honest['edp_loss']:.4f}  "
          f"(niveles elegidos por pliegue: {honest['n_distinct_levels']} distinto(s))")
    print(f"oraculo               EDP loss = {baselines['oraculo']:.4f}")
    print(f"margen del modelo sobre la constante honesta: "
          f"{honest['edp_loss'] - baselines['oraculo']:.4f}")


def report_gpu() -> None:
    print()
    print("=" * 78)
    print("GPU -- pacca_gpu_dvfs_20260820 (campaña ANTERIOR a K/L/M, NO es el dataset final)")
    print("=" * 78)

    df = pd.read_csv(DATA_DIR / "gpu_windows.csv.gz")
    df = df[~df["kernel_ref"].isin(GPU_EXCLUDE)]

    runs = pair_dataset.aggregate_gpu_runs(df)
    pairs = pair_dataset.build_pair_dataset(runs, pair_dataset.GPU_FEATURES, ref_level="F0")
    print(f"corridas agregadas: {len(runs)}  pares construidos: {len(pairs)}  "
          f"descartados sin referencia: {pairs.attrs['dropped_no_ref']}  "
          f"(excluidos del catálogo: {sorted(GPU_EXCLUDE)})")

    edp = _edp_by_level(pairs, ref_level="F0")
    print(f"kernels con EDP completo en todos los niveles: {len(edp)}")
    print(edp.round(3).to_string())

    if len(edp) < 2:
        print("menos de 2 kernels con cobertura completa -- LOKO no aplica, se omite")
        return

    baselines = protocol.trivial_baselines(edp, max_level="F0")
    honest = protocol.honest_constant_baseline(edp)
    print(f"\nsiempre F0 (maxima)   EDP loss = {baselines['siempre_maxima']:.4f}")
    print(f"constante HONESTA     EDP loss = {honest['edp_loss']:.4f}  "
          f"(niveles elegidos por pliegue: {honest['n_distinct_levels']} distinto(s))")
    print(f"oraculo               EDP loss = {baselines['oraculo']:.4f}")
    print(f"margen del modelo sobre la constante honesta: "
          f"{honest['edp_loss'] - baselines['oraculo']:.4f}")


def main() -> int:
    report_cpu()
    report_gpu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
