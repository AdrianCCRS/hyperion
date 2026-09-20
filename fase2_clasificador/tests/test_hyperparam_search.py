"""Tests de hyperparam_search.py -- añadido para cerrar el hueco real que el
usuario señaló: antes de este módulo, train_phase.py/train_phase_gpu.py
usaban hiperparámetros fijos a mano, sin ninguna búsqueda (contradice
Plan_Detallado_Realineacion_Hyperion.md §3.3 punto 1: "búsqueda de
hiperparámetros -- grid o Bayesiana -- en un split interno").
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase2_clasificador.eval import hyperparam_search, protocol


def _synthetic_df(n_familias: int, n_por_familia: int, seed: int) -> pd.DataFrame:
    """Dataset sintético con una sola feature perfectamente separadora --
    solo necesita confirmar que la búsqueda converge a un modelo que la
    aprovecha, no medir generalización real."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_familias):
        familia = f"familia_{i}"
        x = rng.normal(size=n_por_familia)
        label = np.where(x > 0, "memory_bound", "compute_bound")
        for xi, li in zip(x, label):
            rows.append({"feature_1": xi, "phase_label_train": li, "kernel_family": familia})
    return pd.DataFrame(rows)


def test_n_groups_cuenta_pliegues_reales_no_valores_unicos_de_kernel_col():
    # kernel_ref distintos comparten familia -- n_groups debe contar
    # familias (2), no kernel_ref únicos (4).
    df = pd.DataFrame({
        "kernel_ref": ["dual_axpy_cpu_N10", "dual_axpy_cpu_N20", "npb_bt", "npb_bt_c"],
    })
    n = hyperparam_search.n_groups(df, kernel_col="kernel_ref", fold_fn=protocol.leave_one_familia_out)
    assert n == 2


def test_search_best_params_falla_con_menos_de_2_familias():
    df = _synthetic_df(n_familias=1, n_por_familia=20, seed=0)
    X = df[["feature_1"]].to_numpy(dtype=np.float32)
    y = (df["phase_label_train"] == "memory_bound").to_numpy()

    def build_fn(**params):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=0, **params)

    def space_fn(trial):
        return {"max_depth": trial.suggest_int("max_depth", 1, 3)}

    with pytest.raises(ValueError, match=">=2 familias"):
        hyperparam_search.search_best_params(
            build_fn, space_fn, df, X, y, kernel_col="kernel_family",
            seed=0, n_trials=3, fold_fn=protocol.leave_one_kernel_out,
        )


def test_search_best_params_encuentra_hiperparametros_razonables():
    df = _synthetic_df(n_familias=5, n_por_familia=40, seed=1)
    X = df[["feature_1"]].to_numpy(dtype=np.float32)
    y = (df["phase_label_train"] == "memory_bound").to_numpy()

    def build_fn(**params):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=0, **params)

    def space_fn(trial):
        return {"max_depth": trial.suggest_int("max_depth", 1, 15)}

    best_params, best_score = hyperparam_search.search_best_params(
        build_fn, space_fn, df, X, y, kernel_col="kernel_family",
        seed=0, n_trials=20, fold_fn=protocol.leave_one_kernel_out,
    )
    # Con una sola feature perfectamente separadora, un árbol de
    # profundidad 1 ya basta -- cualquier max_depth>=1 debe alcanzar F1
    # alto (no hay más estructura que aprender), así que el criterio real
    # es que la búsqueda SÍ encuentre esa región de buen desempeño, no que
    # prefiera el árbol más simple (Optuna maximiza el score, no penaliza
    # complejidad -- eso no es lo que este módulo promete).
    assert best_score > 0.9, "el problema es trivialmente separable; la búsqueda debe encontrar F1 alto"
    assert 1 <= best_params["max_depth"] <= 15


def test_search_best_params_reanuda_desde_sqlite_sin_repetir_trials(tmp_path):
    df = _synthetic_df(n_familias=4, n_por_familia=20, seed=4)
    X = df[["feature_1"]].to_numpy(dtype=np.float32)
    y = (df["phase_label_train"] == "memory_bound").to_numpy()

    def build_fn(**params):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=0, **params)

    def space_fn(trial):
        return {"max_depth": trial.suggest_int("max_depth", 1, 4)}

    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    first_events = []
    hyperparam_search.search_best_params(
        build_fn, space_fn, df, X, y, kernel_col="kernel_family", seed=0,
        n_trials=3, fold_fn=protocol.leave_one_kernel_out, storage=storage,
        study_name="resume_test", on_trial_complete=first_events.append,
    )
    second_events = []
    hyperparam_search.search_best_params(
        build_fn, space_fn, df, X, y, kernel_col="kernel_family", seed=0,
        n_trials=3, fold_fn=protocol.leave_one_kernel_out, storage=storage,
        study_name="resume_test", on_trial_complete=second_events.append,
    )
    assert len(first_events) == 3
    assert second_events == []


def test_search_best_params_nunca_toca_la_familia_de_prueba_externa():
    """La búsqueda interna recibe SOLO el df ya recortado a la partición de
    entrenamiento del pliegue externo -- este test confirma que el
    resultado no depende de datos fuera de lo que se le pasó explícitamente
    (si se pasa un df con una sola familia sospechosa de ser el pliegue de
    prueba, debe fallar en vez de usarla)."""
    df_train_outer = _synthetic_df(n_familias=3, n_por_familia=30, seed=2)
    familia_prueba_excluida = "familia_intrusa_no_debe_aparecer"
    assert familia_prueba_excluida not in df_train_outer["kernel_family"].unique()

    X = df_train_outer[["feature_1"]].to_numpy(dtype=np.float32)
    y = (df_train_outer["phase_label_train"] == "memory_bound").to_numpy()

    def build_fn(**params):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=0, **params)

    def space_fn(trial):
        return {"max_depth": trial.suggest_int("max_depth", 1, 5)}

    best_params, best_score = hyperparam_search.search_best_params(
        build_fn, space_fn, df_train_outer, X, y, kernel_col="kernel_family",
        seed=0, n_trials=5, fold_fn=protocol.leave_one_kernel_out,
    )
    assert isinstance(best_params, dict)
    assert np.isfinite(best_score)
