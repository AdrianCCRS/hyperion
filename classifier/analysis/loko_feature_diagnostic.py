"""¿Con cuanta informacion REAL entrena el modelo por par?

Pregunta que motiva este script (2026-08-25): el piloto LOKO
(`loko_pilot.py`) entrena sobre 162 filas en GPU y 450 en CPU, pero las
features son promedios de la corrida de REFERENCIA, no de la candidata.
Si esos promedios apenas varian entre repeticiones del mismo kernel,
entonces el modelo no ve N filas independientes: ve **un punto por
kernel**, replicado una vez por (repeticion x nivel candidato). El tamaño
efectivo de muestra para aprender "caracteristicas de la carga -> como
responde al reloj" seria entonces el numero de KERNELS, no el de filas --
6 en GPU, 9 en CPU.

Eso cambiaria por completo la lectura del resultado negativo: no seria un
modelo mal ajustado sobre datos suficientes, sino un modelo con 3 features
tratando de generalizar desde 5 puntos a un sexto.

Mide tres cosas, sin asumir ninguna:
  1. Dispersion INTRA-kernel de cada feature (entre repeticiones) frente a
     la dispersion ENTRE kernels. Si intra << entre, las repeticiones son
     replicas y no muestras nuevas.
  2. Numero de vectores de features distintos que el modelo ve de verdad.
  3. Rango efectivo de la matriz de features, que acota cuanta estructura
     puede aprender el regresor.

Uso: python3 loko_feature_diagnostic.py --axis gpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.analysis import loko_pilot  # noqa: E402
from classifier.features import pair_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["cpu", "gpu"], required=True)
    args = parser.parse_args()

    if args.axis == "gpu":
        base, ref_level = loko_pilot.GPU_BASE, loko_pilot.GPU_REF_LEVEL
        feature_cols = list(pair_dataset.GPU_FEATURES)
        exclude = set(loko_pilot.GPU_EXCLUDE)
    else:
        base, ref_level = loko_pilot.CPU_BASE, loko_pilot.CPU_REF_LEVEL
        feature_cols = list(pair_dataset.CPU_FEATURES)
        exclude = set()
    exclude |= loko_pilot.calibration_kernel_ids()

    runs = loko_pilot.collect_runs(base, args.axis, feature_cols)
    runs = runs[~runs["kernel_ref"].isin(exclude)].reset_index(drop=True)
    pairs = pair_dataset.build_pair_dataset(runs, feature_cols, ref_level=ref_level)

    ref_cols = [f"ref_{c}" for c in feature_cols]
    print(f"filas del dataset por par : {len(pairs)}")
    print(f"kernels (pliegues LOKO)   : {pairs['kernel_ref'].nunique()}")
    print(f"features del modelo       : {ref_cols + ['level_code']}")
    print()

    print("=== 1. Dispersion intra-kernel (entre repeticiones) vs entre kernels ===")
    print(f"{'feature':<28}{'CV% intra medio':>18}{'CV% entre kernels':>20}{'razon':>10}")
    for col in ref_cols:
        values = pd.to_numeric(pairs[col], errors="coerce")
        by_kernel = values.groupby(pairs["kernel_ref"])
        # CV intra: dispersion entre repeticiones DENTRO de cada kernel.
        intra = (by_kernel.std() / by_kernel.mean().abs() * 100).mean()
        # CV entre: dispersion de las medias por kernel.
        means = by_kernel.mean()
        entre = float(means.std() / abs(means.mean()) * 100) if means.mean() else float("nan")
        ratio = entre / intra if intra and np.isfinite(intra) and intra > 0 else float("inf")
        print(f"{col:<28}{intra:>18.3f}{entre:>20.3f}{ratio:>10.1f}x")
    print()
    print("  (razon alta = las repeticiones son replicas del mismo punto;")
    print("   toda la señal aprovechable vive ENTRE kernels, no dentro)")
    print()

    print("=== 2. Vectores de features realmente distintos ===")
    feats = pairs[ref_cols].astype(float)
    exact = len(feats.drop_duplicates())
    # Redondeo a 3 cifras significativas: dos corridas que difieren en el
    # sexto decimal no son informacion nueva para un arbol.
    rounded = feats.round(3).drop_duplicates()
    print(f"  filas totales                        : {len(feats)}")
    print(f"  vectores exactamente distintos       : {exact}")
    print(f"  distintos redondeando a 3 decimales  : {len(rounded)}")
    print(f"  numero de kernels                    : {pairs['kernel_ref'].nunique()}")
    print()

    print("=== 3. Rango efectivo de la matriz de features ===")
    # ESTANDARIZAR, no solo centrar. La primera version de este script solo
    # centraba, y en CPU el resultado salio inservible: `ref_ips` vive en
    # ~1e11 mientras `ref_ipc` vive en ~1, asi que el primer valor singular
    # se comia el 100% de la "varianza" por pura escala y el diagnostico
    # reportaba rango 1 -- un artefacto de unidades, no una colinealidad
    # real. Dividir por la desviacion tipica de cada columna hace que el
    # numero mida estructura compartida entre features, que es lo que se
    # queria medir. (Al regresor de arboles la escala no le afecta; a este
    # diagnostico si.)
    values = feats.to_numpy()
    centered = values - values.mean(axis=0)
    scale = centered.std(axis=0)
    scale[scale == 0] = 1.0  # columna constante: se deja en cero, no NaN
    singular = np.linalg.svd(centered / scale, compute_uv=False)
    total = singular.sum()
    print(f"  valores singulares: {np.round(singular, 4).tolist()}")
    if total > 0:
        share = np.cumsum(singular) / total
        print(f"  varianza acumulada: {np.round(share, 4).tolist()}")
        n_90 = int(np.searchsorted(share, 0.90) + 1)
        print(f"  componentes para el 90% de la señal: {n_90} de {len(ref_cols)}")
    print()

    print("=== Lectura ===")
    n_kernels = pairs["kernel_ref"].nunique()
    print(f"  En LOKO el modelo entrena con {n_kernels - 1} kernels y predice sobre 1.")
    print(f"  Si el punto 2 da ~{n_kernels} vectores distintos, el N efectivo para")
    print(f"  aprender la relacion carga->respuesta es {n_kernels - 1}, no {len(pairs)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
