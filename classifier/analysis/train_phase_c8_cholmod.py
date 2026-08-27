"""C8 + CHOLMOD: la prueba real de si un cuarto kernel con mezcla de fase
GENUINA (no por corrimiento de ridge) ayuda al clasificador a generalizar,
en vez de solo documentar el techo de C8 (~0.5, `Estrategia_CPU_Fase2.md`
§7.bis/§7.ter).

CHOLMOD (campaña `pacca_cpu_cholmod_20260827`, 2026-08-27) es distinto de
los otros 3 kernels de C8 en un punto clave: su mezcla es fuerte y
CONSISTENTE en los 6 niveles (19-34% de ventanas minoritarias, sube al
bajar el reloj), no concentrada en un extremo como npb_lu/npb_bt (ridge-
shift, mezcla solo a reloj bajo) ni como 3mm_omp (mezcla real pero solo a
reloj alto). Si el clasificador mejora al agregarlo, confirma que el
techo de C8 era composición de entrenamiento, no falta de datos de mezcla
real; si no mejora, el problema es mas profundo que "agregar un kernel
mas con mezcla".

Dos directorios de campaña distintos con prefijos distintos -- por eso
esto no es un flag mas de `train_phase.py` (que asume una sola BASE/CID).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "hyperion"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from classifier.eval import protocol
from classifier.training import train_phase

LEVELS = train_phase.LEVELS
FEATURES = train_phase.FEATURES
LABEL = train_phase.LABEL
READ_COLS = train_phase.READ_COLS
PER_RUN_SAMPLE = 2000
SEED = 20260806

SOURCES = [
    # (BASE, CID, [kernels])
    (
        Path.home() / "hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174",
        "pacca_cpu_final_attempt03_20260820",
        ["npb_lu", "npb_bt", "rajaperf_polybench_3mm_omp"],
    ),
    (
        Path.home() / "hyperion-results/campaigns/pacca_cpu_cholmod_20260827",
        "pacca_cpu_cholmod_20260827",
        ["cpu_cholmod"],
    ),
]


def load() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    frames = []
    for base, cid, kernels in SOURCES:
        for kernel in kernels:
            for level in LEVELS:
                for rep in range(1, 11):
                    path = base / f"{cid}__{kernel}__{level}__rep{rep:02d}" / "windows.csv"
                    if not path.exists():
                        continue
                    frame = pd.read_csv(path, usecols=lambda c: c in READ_COLS, low_memory=False)
                    frame = frame[
                        (frame["quality_status"] == "ok")
                        & frame["frequency_quality_status"].isin(["valid", "not_applicable_native"])
                        & frame[LABEL].notna()
                        & (frame[LABEL] != "")
                    ]
                    if len(frame) > PER_RUN_SAMPLE:
                        take = rng.choice(len(frame), PER_RUN_SAMPLE, replace=False)
                        frame = frame.iloc[np.sort(take)]
                    frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + [LABEL])


def main() -> None:
    df = load()
    print(f"matriz: {len(df):,} ventanas | kernels: {sorted(df['kernel_ref'].unique())}")
    print(f"distribucion de fase: {dict(df[LABEL].value_counts())}\n")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = (df[LABEL] == "memory_bound").to_numpy()

    models_proto = {
        "mayoritaria": DummyClassifier(strategy="most_frequent"),
        "arbol_prof6": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
    }

    for name, prototype in models_proto.items():
        per_fold = {}
        for idx_train, idx_test, kernel in protocol.leave_one_kernel_out(df):
            protocol.assert_no_kernel_leak(df, idx_train, idx_test)
            model = clone(prototype)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            per_fold[kernel] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
        summary = protocol.fold_summary(per_fold)
        per_fold_txt = "  ".join(f"{k}={v:.3f}" for k, v in per_fold.items())
        print(f"{name:<16} F1 macro={summary['mean']:.3f}  ({per_fold_txt})")


if __name__ == "__main__":
    main()
