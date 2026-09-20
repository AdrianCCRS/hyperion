"""Tests de model_specs.py -- constructores/espacios de búsqueda compartidos
entre train_phase.py (CPU) y train_phase_gpu.py (GPU)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase2_clasificador.training import model_specs

_TODOS_LOS_MODELOS = {
    "mayoritaria", "arbol_prof1", "regresion_log", "arbol_prof6",
    "random_forest", "extra_trees", "xgboost",
}
_TUNABLES = {"regresion_log", "arbol_prof6", "random_forest", "extra_trees", "xgboost"}


def test_build_fixed_models_solo_tiene_las_lineas_base_no_sintonizables():
    fixed = model_specs.build_fixed_models(seed=0)
    assert set(fixed) == {"mayoritaria", "arbol_prof1"}


def test_build_models_incluye_los_7_candidatos_del_plan():
    modelos = model_specs.build_models(seed=0, scale_pos_weight=2.0)
    assert set(modelos) == _TODOS_LOS_MODELOS


def test_tunable_specs_cubre_exactamente_los_modelos_sintonizables():
    tunable = model_specs.tunable_specs(seed=0, scale_pos_weight=2.0)
    assert set(tunable) == _TUNABLES


def test_tunable_specs_propaga_el_limite_de_workers():
    tunable = model_specs.tunable_specs(seed=0, scale_pos_weight=1.0, n_jobs=7)
    for name in ("random_forest", "extra_trees", "xgboost"):
        build_fn, _ = tunable[name]
        assert build_fn(**model_specs.FALLBACK_PARAMS[name]).n_jobs == 7


def test_fallback_params_cubre_todos_los_sintonizables():
    assert set(model_specs.FALLBACK_PARAMS) == _TUNABLES


def test_tunable_builders_producen_estimadores_entrenables():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    tunable = model_specs.tunable_specs(seed=0, scale_pos_weight=1.5)
    for name, (build_fn, space_fn) in tunable.items():
        params = model_specs.FALLBACK_PARAMS[name]
        model = build_fn(**params)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape, f"{name}: predict() no devolvió la forma esperada"


def test_tunable_space_fn_devuelve_dict_con_optuna_real():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tunable = model_specs.tunable_specs(seed=0, scale_pos_weight=1.0)
    study = optuna.create_study()
    for name, (_build_fn, space_fn) in tunable.items():
        trial = study.ask()  # una prueba nueva por modelo: dos modelos
        # distintos reusan nombres de hiperparámetro (p.ej. "max_depth"),
        # y Optuna exige que el mismo trial no re-sugiera un nombre con
        # una distribución distinta.
        params = space_fn(trial)
        assert isinstance(params, dict) and params, f"{name}: espacio de búsqueda vacío"
