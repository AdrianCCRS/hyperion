"""Optimizacion anidada y comparacion de modelos del selector."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import pickle
import platform
import time

import numpy as np
import pandas as pd

from . import label_health, models


FAMILIES = ("logistic", "decision_tree", "random_forest", "xgboost")
SEED = 20260828


def leave_one_operation_out(frame: pd.DataFrame) -> Iterable[tuple[np.ndarray, np.ndarray, str]]:
    operations = sorted(frame["operation"].dropna().unique())
    if len(operations) < 2:
        raise ValueError(f"se requieren al menos dos operaciones, hay {len(operations)}")
    values = frame["operation"].to_numpy()
    positions = np.arange(len(frame))
    for operation in operations:
        test = values == operation
        yield positions[~test], positions[test], str(operation)


def assert_no_group_leak(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_ops = set(frame.iloc[train_idx]["operation"])
    test_ops = set(frame.iloc[test_idx]["operation"])
    if train_ops & test_ops:
        raise AssertionError(f"fuga de operacion: {sorted(train_ops & test_ops)}")
    train_configs = set(frame.iloc[train_idx]["config_id"])
    test_configs = set(frame.iloc[test_idx]["config_id"])
    if train_configs & test_configs:
        raise AssertionError(f"fuga de config_id: {sorted(train_configs & test_configs)[:5]}")


def selected_rows(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    work = frame.copy()
    work["predicted_probability"] = probabilities
    indices = work.groupby("decision_group_id", observed=True)["predicted_probability"].idxmax()
    return work.loc[indices].sort_values("decision_group_id").reset_index(drop=True)


def selection_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score

    chosen = selected_rows(frame, probabilities)
    oracle_idx = frame.groupby("decision_group_id", observed=True)["edp_mean"].idxmin()
    oracle = frame.loc[oracle_idx].sort_values("decision_group_id").reset_index(drop=True)
    if list(chosen["decision_group_id"]) != list(oracle["decision_group_id"]):
        raise AssertionError("grupos elegidos y oraculo no coinciden")
    oracle_sum = float(oracle["edp_mean"].sum())
    chosen_sum = float(chosen["edp_mean"].sum())
    action_accuracy = float((chosen["action_id"].to_numpy() == oracle["action_id"].to_numpy()).mean())
    device_accuracy = float((chosen["candidate_device"].to_numpy() == oracle["candidate_device"].to_numpy()).mean())
    predicted_binary = np.zeros(len(frame), dtype=int)
    # Una sola prediccion positiva por grupo, coherente con la decision final.
    chosen_original_indices = (
        frame.assign(_p=probabilities)
        .groupby("decision_group_id", observed=True)["_p"].idxmax().to_numpy()
    )
    positions = {index: pos for pos, index in enumerate(frame.index)}
    for index in chosen_original_indices:
        predicted_binary[positions[index]] = 1
    y = frame["is_optimal"].to_numpy(dtype=int)
    level_order = {level: position for position, level in enumerate(
        ("REF", "F0", "F1", "F2", "F3", "F4", "F5", "F6")
    )}
    distances = []
    for (_, selected), (_, optimal) in zip(chosen.iterrows(), oracle.iterrows()):
        if selected["candidate_device"] != optimal["candidate_device"]:
            continue
        distance = abs(level_order[str(selected["cpu_level"])] - level_order[str(optimal["cpu_level"])])
        if selected["candidate_device"] == "gpu":
            distance += abs(level_order[str(selected["gpu_level"])] - level_order[str(optimal["gpu_level"])])
        distances.append(distance)
    return {
        "edp_loss": chosen_sum / oracle_sum if oracle_sum > 0 else float("nan"),
        "regret_pct": 100.0 * (chosen_sum / oracle_sum - 1.0) if oracle_sum > 0 else float("nan"),
        "action_accuracy": action_accuracy,
        "device_accuracy": device_accuracy,
        "frequency_level_distance": float(np.mean(distances)) if distances else float("nan"),
        "f1_positive": float(f1_score(y, predicted_binary, zero_division=0)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "n_decisions": int(len(chosen)),
    }


def measure_selector_latency(
    model: Any,
    frame: pd.DataFrame,
    *,
    warmups: int = 50,
    repeats: int = 200,
) -> dict[str, float]:
    from threadpoolctl import threadpool_limits

    sizes = frame.groupby("decision_group_id", observed=True).size()
    group_id = sizes.idxmax()
    sample = frame[frame["decision_group_id"] == group_id]
    timings = []
    # RF/XGBoost ya usan n_jobs=1; este limite cubre tambien BLAS/OpenMP del
    # preprocesador y la regresion logistica durante la medicion comparable.
    with threadpool_limits(limits=1):
        for _ in range(warmups):
            models.positive_probability(model, sample)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            models.positive_probability(model, sample)
            timings.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "latency_p50_us": float(np.percentile(timings, 50)),
        "latency_p95_us": float(np.percentile(timings, 95)),
        "latency_p99_us": float(np.percentile(timings, 99)),
        "latency_candidate_count": int(len(sample)),
    }


def _fit(family: str, params: Mapping[str, Any], train: pd.DataFrame, seed: int):
    ratio = models.class_weight_ratio(train["is_optimal"])
    pipeline = models.build_pipeline(
        family, params, train, seed=seed,
        scale_pos_weight=ratio if family == "xgboost" else None,
    )
    start = time.perf_counter()
    pipeline.fit(train, train["is_optimal"].astype(int))
    return pipeline, time.perf_counter() - start


def inner_edp_loss(family: str, params: Mapping[str, Any], train: pd.DataFrame, seed: int) -> float:
    losses = []
    for inner_train_idx, inner_valid_idx, _ in leave_one_operation_out(train):
        assert_no_group_leak(train, inner_train_idx, inner_valid_idx)
        inner_train = train.iloc[inner_train_idx]
        inner_valid = train.iloc[inner_valid_idx]
        model, _ = _fit(family, params, inner_train, seed)
        probabilities = models.positive_probability(model, inner_valid)
        losses.append(selection_metrics(inner_valid, probabilities)["edp_loss"])
    return float(np.mean(losses))


def _study_name(strategy: str, family: str, outer_fold: str) -> str:
    safe = lambda value: "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return safe(f"selector_{strategy}_{family}_{outer_fold}")


def tune_family(
    family: str,
    train: pd.DataFrame,
    *,
    strategy: str,
    outer_fold: str,
    output_dir: Path,
    trials: int,
    seed: int = SEED,
    latency_warmups: int = 50,
    latency_repeats: int = 200,
) -> tuple[dict[str, Any], Any]:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Optuna no esta instalado; instale classifier/requirements.txt") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    database = (output_dir / "optuna.sqlite3").resolve()
    study = optuna.create_study(
        study_name=_study_name(strategy, family, outer_fold),
        storage=f"sqlite:///{database}",
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
        load_if_exists=True,
    )

    def objective(trial: Any) -> tuple[float, float]:
        params = models.suggest_parameters(trial, family)
        edp_loss = inner_edp_loss(family, params, train, seed + trial.number)
        fitted, fit_seconds = _fit(family, params, train, seed + trial.number)
        latency = measure_selector_latency(
            fitted, train, warmups=latency_warmups, repeats=latency_repeats,
        )
        trial.set_user_attr("fit_seconds", fit_seconds)
        trial.set_user_attr("latency_p50_us", latency["latency_p50_us"])
        trial.set_user_attr("latency_p95_us", latency["latency_p95_us"])
        return edp_loss, latency["latency_p99_us"]

    completed = sum(trial.state.name == "COMPLETE" for trial in study.trials)
    remaining = max(0, trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    if not study.best_trials:
        raise RuntimeError(f"estudio sin trials Pareto: {study.study_name}")
    best_edp = min(float(trial.values[0]) for trial in study.best_trials)
    eligible = [trial for trial in study.best_trials if float(trial.values[0]) <= best_edp * 1.01]
    selected = min(eligible, key=lambda trial: (float(trial.values[1]), float(trial.values[0]), trial.number))
    fitted, fit_seconds = _fit(family, selected.params, train, seed)
    result = {
        "study_name": study.study_name,
        "selected_trial": selected.number,
        "selected_params": selected.params,
        "inner_edp_loss": float(selected.values[0]),
        "inner_latency_p99_us": float(selected.values[1]),
        "fit_seconds": fit_seconds,
        "completed_trials": sum(trial.state.name == "COMPLETE" for trial in study.trials),
        "pareto_trials": len(study.best_trials),
    }
    return result, fitted


def _serialize_model(model: Any, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(payload)
    return len(payload)


def _baseline_fold(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    constant_by_state = (
        train.groupby(["resource_state", "action_id"], observed=True)["edp_mean"].mean()
        .groupby(level=0).idxmin().map(lambda pair: pair[1]).to_dict()
    )
    baselines: list[tuple[str, str | None]] = [
        ("always_cpu_ref", "cpu:REF"),
        ("always_cpu_f0", "cpu:F0"),
        ("always_gpu_ref", "gpu:REF:REF"),
    ]
    oracle = test.groupby("decision_group_id", observed=True)["edp_mean"].min()
    rows = []
    constant_values = []
    constant_actions: dict[str, str] = {}
    for group_id, group in test.groupby("decision_group_id", observed=True):
        state = str(group["resource_state"].iloc[0])
        action = constant_by_state.get(state)
        selected = group[group["action_id"] == action]
        if len(selected) != 1:
            constant_values = []
            break
        constant_values.append(float(selected.iloc[0]["edp_mean"]))
        constant_actions[state] = str(action)
    if constant_values:
        loss = float(sum(constant_values) / oracle.sum())
        rows.append({
            "family": "best_constant_train", "edp_loss": loss,
            "regret_pct": 100.0 * (loss - 1.0),
            "action": json.dumps(constant_actions, sort_keys=True),
        })
    for name, action in baselines:
        chosen = test[test["action_id"] == action].set_index("decision_group_id")["edp_mean"] if action else pd.Series(dtype=float)
        common = oracle.index.intersection(chosen.index)
        if len(common) != len(oracle):
            continue
        loss = float(chosen.loc[common].sum() / oracle.loc[common].sum())
        rows.append({"family": name, "edp_loss": loss, "regret_pct": 100.0 * (loss - 1.0), "action": action})
    rng = np.random.default_rng(seed)
    random_values = []
    for _, group in test.groupby("decision_group_id", observed=True):
        random_values.append(float(group.iloc[int(rng.integers(0, len(group)))]["edp_mean"]))
    random_loss = float(sum(random_values) / oracle.sum())
    rows.extend([
        {"family": "random", "edp_loss": random_loss, "regret_pct": 100.0 * (random_loss - 1.0)},
        {"family": "oracle", "edp_loss": 1.0, "regret_pct": 0.0},
    ])
    return rows


def best_baseline_comparison(
    baseline_records: list[dict[str, Any]], selected_edp_loss_mean: float,
) -> dict[str, Any]:
    """Compara la familia elegida contra la mejor baseline no-oraculo.

    El contrato antes callaba cuando el modelo elegido quedaba peor que una
    alternativa trivial (p.ej. best_constant_train); esto lo deja explicito.
    """
    baselines_df = pd.DataFrame(baseline_records)
    non_oracle = baselines_df[baselines_df["family"] != "oracle"] if not baselines_df.empty else baselines_df
    if non_oracle.empty:
        return {
            "best_baseline_family": None,
            "best_baseline_edp_loss": None,
            "beats_best_baseline": None,
        }
    summary = non_oracle.groupby("family", observed=True)["edp_loss"].mean()
    best_baseline_family = str(summary.idxmin())
    best_baseline_edp_loss = float(summary.min())
    return {
        "best_baseline_family": best_baseline_family,
        "best_baseline_edp_loss": best_baseline_edp_loss,
        "beats_best_baseline": bool(selected_edp_loss_mean < best_baseline_edp_loss),
    }


def run_nested_tuning(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    families: Iterable[str] = FAMILIES,
    trials: int = 100,
    seed: int = SEED,
    latency_warmups: int = 50,
    latency_repeats: int = 200,
) -> dict[str, Path]:
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(dataset_path, low_memory=False)
    if frame.empty or not {0, 1} <= set(frame["is_optimal"].astype(int).unique()):
        raise ValueError("dataset vacio o sin ambas clases")
    strategy = str(frame["strategy"].iloc[0])
    fold_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    for outer_train_idx, outer_test_idx, held_out in leave_one_operation_out(frame):
        assert_no_group_leak(frame, outer_train_idx, outer_test_idx)
        train, test = frame.iloc[outer_train_idx], frame.iloc[outer_test_idx]
        baseline_records.extend({"held_out_operation": held_out, **record}
                                for record in _baseline_fold(train, test, seed))
        for family in families:
            tuning, model = tune_family(
                family, train, strategy=strategy, outer_fold=held_out,
                output_dir=output_dir / "studies", trials=trials, seed=seed,
                latency_warmups=latency_warmups, latency_repeats=latency_repeats,
            )
            start = time.perf_counter()
            probabilities = models.positive_probability(model, test)
            predict_seconds = time.perf_counter() - start
            metrics = selection_metrics(test, probabilities)
            latency = measure_selector_latency(
                model, test, warmups=latency_warmups, repeats=latency_repeats,
            )
            size = _serialize_model(model, output_dir / "fold_models" / f"{family}__holdout_{held_out}.pkl")
            fold_records.append({
                "strategy": strategy, "family": family,
                "held_out_operation": held_out, **metrics, **latency, **tuning,
                "model_size_bytes": size, "batch_predict_seconds": predict_seconds,
                "tree_count": models.model_tree_count(model),
            })
    folds = pd.DataFrame(fold_records)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(baseline_records).to_csv(output_dir / "baseline_fold_metrics.csv", index=False)
    summary = folds.groupby("family", observed=True).agg(
        edp_loss_mean=("edp_loss", "mean"),
        edp_loss_std=("edp_loss", "std"),
        edp_loss_worst=("edp_loss", "max"),
        action_accuracy_mean=("action_accuracy", "mean"),
        action_accuracy_std=("action_accuracy", "std"),
        device_accuracy_mean=("device_accuracy", "mean"),
        frequency_level_distance_mean=("frequency_level_distance", "mean"),
        f1_positive_mean=("f1_positive", "mean"),
        average_precision_mean=("average_precision", "mean"),
        latency_p50_us=("latency_p50_us", "median"),
        latency_p95_us=("latency_p95_us", "median"),
        latency_p99_us=("latency_p99_us", "median"),
        model_size_bytes=("model_size_bytes", "median"),
        fit_seconds_mean=("fit_seconds", "mean"),
        tree_count_mean=("tree_count", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "model_comparison.csv", index=False)

    n_folds = int(folds["held_out_operation"].nunique())
    summary["edp_loss_stderr"] = summary["edp_loss_std"] / np.sqrt(max(n_folds, 1))
    best_row = summary.loc[summary["edp_loss_mean"].idxmin()]
    best_loss = float(best_row["edp_loss_mean"])
    best_stderr = float(best_row["edp_loss_stderr"]) if pd.notna(best_row["edp_loss_stderr"]) else 0.0
    # Una familia es indistinguible del ganador si su brecha frente a el no
    # supera la dispersion combinada entre pliegues externos (ARC critica
    # punto 2: sin esto, 6 numeros con ruido decidian la familia final).
    dispersion_gap = summary["edp_loss_stderr"].fillna(0.0) + best_stderr
    within_relative = summary["edp_loss_mean"] <= best_loss * 1.01
    within_dispersion = (summary["edp_loss_mean"] - best_loss) <= dispersion_gap
    eligible = summary[within_relative | within_dispersion]
    winner = eligible.sort_values(
        ["latency_p99_us", "model_size_bytes", "family"], kind="mergesort"
    ).iloc[0]
    winner_family = str(winner["family"])
    families_indistinguishable_from_winner = sorted(eligible["family"].astype(str))
    final_tuning, final_model = tune_family(
        winner_family, frame, strategy=strategy, outer_fold="full",
        output_dir=output_dir / "studies", trials=trials, seed=seed,
        latency_warmups=latency_warmups, latency_repeats=latency_repeats,
    )
    final_size = _serialize_model(final_model, output_dir / "selected_model.pkl")
    categorical, numeric = models.feature_columns(frame)
    health = label_health.assess_label_health(frame)
    selected_edp_loss_mean = float(winner["edp_loss_mean"])
    baseline_comparison = best_baseline_comparison(baseline_records, selected_edp_loss_mean)
    contract = {
        "strategy": strategy,
        "selected_family": winner_family,
        "selection_rule": "edp_loss_within_1pct_or_within_fold_dispersion_then_p99_then_size",
        "families_indistinguishable_from_winner": families_indistinguishable_from_winner,
        "selected_family_edp_loss_mean": selected_edp_loss_mean,
        **baseline_comparison,
        "external_folds": n_folds,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "action_ids": sorted(frame["action_id"].unique()),
        "final_tuning": final_tuning,
        "model_size_bytes": final_size,
        "seed": seed,
        "label_health": health,
        "result_status": health["verdict"],
        "result_status_note": (
            "pipeline_smoke_only: la etiqueta is_optimal no tiene variedad "
            "suficiente para que esta tabla se lea como comparacion de "
            "modelos -- sirve solo para validar que build/eda/tune/evaluate "
            "corren de punta a punta. Ver label_health para el detalle."
            if health["verdict"] == "pipeline_smoke_only" else
            "comparison_valid: la etiqueta tiene variedad suficiente para "
            "interpretar model_comparison.csv como comparacion real."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "hostname": platform.node(),
        },
    }
    try:
        import sklearn
        contract["environment"]["scikit_learn"] = sklearn.__version__
        import optuna
        contract["environment"]["optuna"] = optuna.__version__
        import xgboost
        contract["environment"]["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    (output_dir / "model_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "fold_metrics": output_dir / "fold_metrics.csv",
        "comparison": output_dir / "model_comparison.csv",
        "model": output_dir / "selected_model.pkl",
        "contract": output_dir / "model_contract.json",
    }


def evaluate_existing(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    folds = pd.read_csv(output_dir / "fold_metrics.csv")
    baselines_path = output_dir / "baseline_fold_metrics.csv"
    result = {
        "folds": int(len(folds)),
        "families": sorted(folds["family"].unique()),
        "worst_operation_by_family": {},
    }
    for family, group in folds.groupby("family", observed=True):
        worst = group.sort_values("edp_loss", ascending=False).iloc[0]
        result["worst_operation_by_family"][str(family)] = {
            "operation": str(worst["held_out_operation"]),
            "edp_loss": float(worst["edp_loss"]),
        }
    if baselines_path.exists():
        baselines = pd.read_csv(baselines_path)
        result["baseline_edp_loss_mean"] = baselines.groupby("family")["edp_loss"].mean().to_dict()
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
