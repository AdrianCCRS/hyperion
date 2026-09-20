"""Modelos candidatos (§3.2 del plan) y sus espacios de búsqueda de
hiperparámetros -- compartido entre ``train_phase.py`` (CPU) y
``train_phase_gpu.py`` (GPU): ambos entrenadores comparan EXACTAMENTE los
mismos 7 modelos, solo cambian las features/granularidad de entrada de cada
dispositivo, no los modelos en sí.

Dos grupos, deliberadamente distintos:

- ``build_fixed_models``: líneas base que NO se optimizan a propósito.
  ``mayoritaria`` (DummyClassifier) representa "no hacer nada" -- optimizarla
  no tendría sentido. ``arbol_prof1`` es el punto de comparación
  interpretable del plan (un árbol de profundidad 1 es, por diseño, la
  versión más simple posible); buscarle hiperparámetros lo convertiría en
  otro modelo distinto del que el plan usa como referencia de interpretabilidad.
- ``tunable_specs``: el resto (regresión logística, árbol más profundo,
  Random Forest, Extra Trees, XGBoost) -- cada uno con un constructor
  (``build_fn(**params) -> estimador sin entrenar``) y un espacio de
  búsqueda (``space_fn(trial) -> dict``) para ``hyperparam_search.py``.
"""
from __future__ import annotations


def build_fixed_models(seed: int) -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier

    return {
        "mayoritaria": DummyClassifier(strategy="most_frequent"),
        "arbol_prof1": DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=seed),
    }


def build_models(seed: int, scale_pos_weight: float = 1.0) -> dict:
    """Configuración de hiperparámetros FIJA (sin búsqueda) para los 7
    modelos del plan -- se conserva como función independiente por dos
    razones: (1) es el punto de comparación "antes de optimizar" que el
    capítulo de resultados debe poder citar, y (2) es el respaldo explícito
    cuando un pliegue no tiene familias suficientes para un split interno de
    búsqueda (``hyperparam_search.MIN_FAMILIAS_PARA_BUSQUEDA``) -- nunca se
    fabrica un split, se usa esta configuración fija y se documenta que así
    fue.
    """
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    models = build_fixed_models(seed)
    models.update({
        "regresion_log": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        "arbol_prof6": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, random_state=seed,
            eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        ),
    })
    return models


def _build_regresion_log(seed: int, **params):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed, **params
        ),
    )


def _space_regresion_log(trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


def _build_arbol_prof6(seed: int, **params):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)


def _space_arbol_prof6(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _build_random_forest(seed: int, n_jobs: int = -1, **params):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        class_weight="balanced", n_jobs=n_jobs, random_state=seed, **params
    )


def _space_random_forest(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _build_extra_trees(seed: int, n_jobs: int = -1, **params):
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        class_weight="balanced", n_jobs=n_jobs, random_state=seed, **params
    )


def _space_extra_trees(trial) -> dict:
    # Mismo tipo de espacio que random_forest (mismos hiperparámetros de
    # bosque), pero función propia -- Optuna necesita nombres de parámetro
    # que se puedan reproducir/inspeccionar por separado para cada estudio.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _build_xgboost(seed: int, scale_pos_weight: float, n_jobs: int = -1, **params):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_jobs=n_jobs, random_state=seed, eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, **params,
    )


def _space_xgboost(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.5, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


# Hiperparámetros de respaldo para cuando un pliegue externo no tiene
# familias suficientes para un split interno de búsqueda (ver
# hyperparam_search.MIN_FAMILIAS_PARA_BUSQUEDA) -- EXACTAMENTE los mismos
# valores fijos que build_models() usaba antes de que existiera la búsqueda,
# para que "sin búsqueda posible" caiga en la configuración ya documentada y
# conocida, no en los valores por defecto de sklearn/XGBoost (que no
# coinciden, p.ej. RandomForestClassifier por defecto no limita max_depth).
FALLBACK_PARAMS: dict[str, dict] = {
    "regresion_log": {},  # C=1.0 -- ya era el valor por defecto en build_models()
    "arbol_prof6": {"max_depth": 6},
    "random_forest": {"n_estimators": 100, "max_depth": 12},
    "extra_trees": {"n_estimators": 100, "max_depth": 12},
    "xgboost": {"n_estimators": 100, "max_depth": 6},
}


def tunable_specs(seed: int, scale_pos_weight: float, n_jobs: int = -1) -> dict[str, tuple]:
    """``nombre -> (build_fn, space_fn)`` para los modelos CON búsqueda de
    hiperparámetros. ``build_fn(**params)`` ya tiene ``seed``/
    ``scale_pos_weight`` fijados por clausura -- el caller (
    ``hyperparam_search.search_best_params``) solo pasa los hiperparámetros
    muestreados por Optuna en cada intento.
    """
    return {
        "regresion_log": (
            lambda **p: _build_regresion_log(seed, **p), _space_regresion_log),
        "arbol_prof6": (
            lambda **p: _build_arbol_prof6(seed, **p), _space_arbol_prof6),
        "random_forest": (
            lambda **p: _build_random_forest(seed, n_jobs=n_jobs, **p), _space_random_forest),
        "extra_trees": (
            lambda **p: _build_extra_trees(seed, n_jobs=n_jobs, **p), _space_extra_trees),
        "xgboost": (
            lambda **p: _build_xgboost(seed, scale_pos_weight, n_jobs=n_jobs, **p), _space_xgboost),
    }
