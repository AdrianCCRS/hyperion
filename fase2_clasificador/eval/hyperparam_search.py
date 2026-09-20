"""Búsqueda de hiperparámetros (Optuna, TPE) anidada dentro de
leave-one-familia-out.

``Plan_Detallado_Realineacion_Hyperion.md`` §3.3 punto 1: "Validación
anidada: partición externa leave-one-familia-out para estimar
generalización, con búsqueda de hiperparámetros (grid o Bayesiana) en un
split interno." Antes de este módulo, ``train_phase.py``/``train_phase_gpu.py``
usaban hiperparámetros fijos a mano (``max_depth=12``, ``n_estimators=100``,
etc.) sin ninguna búsqueda -- confirmado al auditar el código, no asumido.

Se elige Optuna (TPE, Bayesiana) sobre una grilla exhaustiva porque los
espacios de ``random_forest``/``extra_trees``/``xgboost`` combinan enteros y
reales en varias dimensiones; una grilla que cubra ese espacio con la misma
resolución sería cara, y TPE aprovecha mejor un presupuesto de intentos
fijo (``n_trials``) que una grilla gruesa fijada a mano.

La búsqueda es "anidada" en el sentido estricto del plan: el ``df`` que
recibe ``search_best_params`` debe ser ya el conjunto de ENTRENAMIENTO de un
pliegue externo (leave-one-familia-out) -- la familia de prueba de ese
pliegue externo nunca participa en la búsqueda de hiperparámetros. El
objetivo que Optuna maximiza es, a su vez, el F1 macro medio de un
leave-one-familia-out INTERNO sobre ese ``df`` de entrenamiento -- dos
niveles de agrupación por familia, ninguno de los dos por fila ni por
``kernel_ref`` individual (ver ``protocol.py``).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
from sklearn.metrics import f1_score

from fase2_clasificador.eval import protocol

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Firma de un espacio de búsqueda: recibe un optuna.Trial, devuelve un dict
# de hiperparámetros muestreados para esa prueba.
SearchSpaceFn = Callable[["optuna.trial.Trial"], dict]
# Firma de un constructor: recibe los hiperparámetros muestreados como
# kwargs, devuelve un estimador sklearn/XGBoost SIN entrenar todavía.
BuilderFn = Callable[..., object]

MIN_FAMILIAS_PARA_BUSQUEDA = 2

FoldFn = Callable[..., "object"]  # protocol.leave_one_familia_out o leave_one_kernel_out


def n_groups(df, kernel_col: str, fold_fn: FoldFn) -> int:
    """Cuenta cuántos pliegues produciría ``fold_fn`` sobre ``df`` -- nunca
    se cuenta ``df[kernel_col].nunique()`` directamente, porque para CPU
    ``kernel_col="kernel_ref"`` y el agrupamiento real es por FAMILIA
    (derivada dentro de ``leave_one_familia_out``): dos ``kernel_ref``
    distintos pueden compartir familia, así que contar valores únicos de
    ``kernel_col`` sobreestimaría cuántos pliegues realmente hay.

    ``fold_fn`` (``leave_one_kernel_out``/``leave_one_familia_out``) lanza
    ``ValueError`` con menos de 2 grupos en vez de devolver un iterador
    vacío -- se captura aquí y se traduce a ``0`` para que el caller pueda
    decidir con un simple ``< MIN_FAMILIAS_PARA_BUSQUEDA`` sin tener que
    conocer ese detalle de las funciones de ``protocol.py``.
    """
    try:
        return sum(1 for _ in fold_fn(df, kernel_col=kernel_col))
    except ValueError:
        return 0


def _inner_cv_score(
    build_fn: BuilderFn,
    params: dict,
    df,
    X: np.ndarray,
    y: np.ndarray,
    kernel_col: str,
    fold_fn: FoldFn,
) -> float:
    """F1 macro medio de un split agrupado INTERNO sobre ``df`` (``fold_fn``
    decide la unidad de agrupación -- ``leave_one_familia_out`` en CPU,
    donde la familia se deriva de ``kernel_ref``; ``leave_one_kernel_out``
    sobre la columna ``kernel_family`` ya precomputada en GPU -- ambas
    agrupan por familia algorítmica, nunca por fila ni por ``kernel_ref``
    individual cuando hay tamaños repetidos de la misma familia)."""
    scores = []
    for idx_train, idx_test, _familia in fold_fn(df, kernel_col=kernel_col):
        model = build_fn(**params)
        model.fit(X[idx_train], y[idx_train])
        pred = model.predict(X[idx_test])
        scores.append(f1_score(y[idx_test], pred, average="macro", zero_division=0))
    return float(np.mean(scores)) if scores else float("nan")


def search_best_params(
    build_fn: BuilderFn,
    space_fn: SearchSpaceFn,
    df,
    X: np.ndarray,
    y: np.ndarray,
    kernel_col: str,
    seed: int,
    n_trials: int,
    fold_fn: FoldFn = protocol.leave_one_familia_out,
    storage: str | None = None,
    study_name: str | None = None,
    on_trial_complete: Callable[[dict], None] | None = None,
) -> tuple[dict, float]:
    """Busca hiperparámetros maximizando el F1 macro medio de un split
    agrupado interno (``fold_fn``) sobre ``df``.

    Precondición explícita, verificada aquí (no en el caller): ``fold_fn``
    debe poder producir al menos ``MIN_FAMILIAS_PARA_BUSQUEDA`` pliegues
    sobre ``df`` -- si no, no hay ningún split interno posible y se lanza
    ``ValueError`` en vez de devolver un resultado fabricado sobre un único
    grupo.
    """
    groups = n_groups(df, kernel_col=kernel_col, fold_fn=fold_fn)
    if groups < MIN_FAMILIAS_PARA_BUSQUEDA:
        raise ValueError(
            "hacen falta >=2 familias en el split interno para buscar "
            f"hiperparámetros, hay {groups}"
        )

    def objective(trial: "optuna.trial.Trial") -> float:
        params = space_fn(trial)
        return _inner_cv_score(build_fn, params, df, X, y, kernel_col, fold_fn)

    sampler = optuna.samplers.TPESampler(seed=seed)
    if storage is not None:
        if not study_name:
            raise ValueError("study_name es obligatorio cuando se usa almacenamiento Optuna")
        study = optuna.create_study(
            direction="maximize", sampler=sampler, storage=storage,
            study_name=study_name, load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    # ``n_trials`` es el presupuesto total de este estudio, no un número
    # adicional en cada reanudación. Así, una nueva ejecución recupera los
    # trials ya guardados en SQLite en vez de repetirlos.
    remaining_trials = max(0, n_trials - len(study.trials))

    def _trial_completed(completed_study, trial) -> None:
        if on_trial_complete is None:
            return
        on_trial_complete({
            "study_name": completed_study.study_name,
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": dict(trial.params),
            "completed_trials": len(completed_study.trials),
            "target_trials": n_trials,
        })

    if remaining_trials:
        study.optimize(
            objective, n_trials=remaining_trials, show_progress_bar=False,
            callbacks=[_trial_completed],
        )
    return study.best_params, study.best_value
