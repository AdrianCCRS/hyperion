"""C8 + contexto temporal: en vez de clasificar cada ventana aislada,
añade la media movil causal de las ultimas K ventanas de la MISMA
corrida para cada feature -- una fase es un patron que persiste en el
tiempo, y el clasificador base (train_phase.py) tira ese patron a la
basura al ver cada ventana por separado, sin memoria de las anteriores.

Mismo protocolo LOKO, mismas 3 kernels de C8 (npb_lu/npb_bt/3mm_omp),
mismos modelos base -- el UNICO cambio es agregar 7 features de media
movil (K=20 ventanas, ~20ms de historia) a las 7 originales.

IMPORTANTE: el orden temporal de windows.csv se preserva ANTES de
submuestrear (el submuestreo de train_phase.load() es aleatorio por
corrida y destruiria la secuencia necesaria para la media movil).

RESULTADO (2026-08-27, Estrategia_CPU_Fase2.md §7.bis): mejora real pero
modesta. `extra_trees`, solo con media movil: F1 macro 0.553 (vs 0.524
baseline instantaneo). `npb_lu` sube 0.639->0.689 (su mezcla es un
desplazamiento gradual del ridge, el contexto temporal lo capta). El
pliegue debil de C8, `3mm_omp`, se queda en ~0.15-0.19 en las tres
variantes -- su problema no es falta de contexto temporal, es que
aprende el mecanismo de npb_bt/npb_lu (reloj bajo -> memory-bound), que
es el INVERSO del suyo (mezcla real, no por corrimiento de ridge). Ningun
tipo de feature arregla eso; hace falta un kernel de entrenamiento con su
mismo mecanismo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "hyperion"))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier.eval import protocol
from classifier.training import train_phase

KERNELS = ["npb_lu", "npb_bt", "rajaperf_polybench_3mm_omp"]
LEVELS = train_phase.LEVELS
BASE = train_phase.BASE
CID = train_phase.CID
FEATURES = train_phase.FEATURES
LABEL = train_phase.LABEL
ROLL_WINDOW = 20
PER_RUN_SAMPLE = 2000
SEED = 20260806

READ_COLS = [*FEATURES, LABEL, "kernel_ref", "freq_level_id", "quality_status",
             "frequency_quality_status", "t_start_ns"]


def load_temporal() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
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
                ].copy()
                if frame.empty:
                    continue
                for col in FEATURES:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
                frame = frame.sort_values("t_start_ns")
                # Media movil CAUSAL (no ve el futuro): shift(1) antes de
                # rolling, para que la feature de la ventana actual solo
                # use las K ANTERIORES, nunca la propia.
                for col in FEATURES:
                    frame[f"roll_{col}"] = (
                        frame[col].shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
                    )
                frame = frame.dropna(subset=[f"roll_{c}" for c in FEATURES])
                if len(frame) > PER_RUN_SAMPLE:
                    take = rng.choice(len(frame), PER_RUN_SAMPLE, replace=False)
                    frame = frame.iloc[np.sort(take)]
                frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    all_cols = FEATURES + [f"roll_{c}" for c in FEATURES]
    return df.dropna(subset=all_cols + [LABEL])


def main() -> None:
    from sklearn.metrics import f1_score

    df = load_temporal()
    print(f"matriz: {len(df):,} ventanas | {df['kernel_ref'].nunique()} kernels")
    print(f"distribucion de fase: {dict(df[LABEL].value_counts())}\n")

    roll_cols = [f"roll_{c}" for c in FEATURES]
    variants = {
        "solo_instantanea (baseline C8)": FEATURES,
        "instantanea+media_movil_20": FEATURES + roll_cols,
        "solo_media_movil_20": roll_cols,
    }

    models_proto = {
        "mayoritaria": DummyClassifier(strategy="most_frequent"),
        "arbol_prof6": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
    }

    y = (df[LABEL] == "memory_bound").to_numpy()

    for variant_name, cols in variants.items():
        print(f"=== {variant_name} ({len(cols)} features) ===")
        X = df[cols].to_numpy(dtype=np.float32)
        for name, prototype in models_proto.items():
            from sklearn.base import clone
            per_fold = {}
            for idx_train, idx_test, kernel in protocol.leave_one_kernel_out(df):
                protocol.assert_no_kernel_leak(df, idx_train, idx_test)
                model = clone(prototype)
                model.fit(X[idx_train], y[idx_train])
                pred = model.predict(X[idx_test])
                per_fold[kernel] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
            summary = protocol.fold_summary(per_fold)
            per_fold_txt = "  ".join(f"{k}={v:.3f}" for k, v in per_fold.items())
            print(f"  {name:<16} F1 macro={summary['mean']:.3f}  ({per_fold_txt})")
        print()


if __name__ == "__main__":
    main()
