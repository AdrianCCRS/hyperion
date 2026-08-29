from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import models, search


def _frame():
    rows = []
    for operation in ("gemm", "fft", "axpy"):
        for config in range(2):
            for action, optimal, edp in (("cpu:REF", 0, 2.0), ("cpu:F0", 1, 1.0)):
                rows.append({
                    "operation": operation, "config_id": f"{operation}_N{config}",
                    "decision_group_id": f"A:{operation}:{config}", "action_id": action,
                    "candidate_device": "cpu", "cpu_level": action.split(":")[1],
                    "gpu_level": np.nan, "is_optimal": optimal, "edp_mean": edp,
                })
    return pd.DataFrame(rows)


def test_leave_one_operation_out_no_comparte_configuraciones():
    frame = _frame()
    folds = list(search.leave_one_operation_out(frame))
    assert len(folds) == 3
    for train, test, operation in folds:
        search.assert_no_group_leak(frame, train, test)
        assert set(frame.iloc[test]["operation"]) == {operation}


def test_selection_metrics_evalua_decision_y_no_exactitud_de_negativos():
    pytest.importorskip("sklearn")
    frame = _frame()
    probabilities = np.tile([0.1, 0.9], len(frame) // 2)
    metrics = search.selection_metrics(frame, probabilities)
    assert metrics["edp_loss"] == pytest.approx(1.0)
    assert metrics["action_accuracy"] == pytest.approx(1.0)
    assert metrics["f1_positive"] == pytest.approx(1.0)


def test_feature_columns_excluye_outcomes_e_iterations():
    frame = pd.DataFrame({
        "operation": ["gemm"], "family": ["matrix"],
        "candidate_device": ["cpu"], "resource_state": ["none_ready"],
        "log10_n": [2.0], "candidate_cpu_fraction": [1.0],
        "is_optimal": [1], "edp_mean": [1.0], "iterations": [100],
    })
    categorical, numeric = models.feature_columns(frame)
    assert "operation" in categorical
    assert "log10_n" in numeric
    assert "edp_mean" not in numeric
    assert "iterations" not in numeric


def test_spaces_cubren_las_cuatro_familias():
    class Trial:
        def suggest_float(self, name, low, high, **kwargs): return low
        def suggest_int(self, name, low, high, **kwargs): return low
        def suggest_categorical(self, name, choices): return choices[0]

    for family in search.FAMILIES:
        assert models.suggest_parameters(Trial(), family)


def _model_frame():
    frame = _frame()
    frame["family"] = np.where(frame["operation"].eq("axpy"), "vector", "matrix")
    frame["resource_state"] = "none_ready"
    frame["log10_n"] = 2.0
    frame["log10_flops_per_dispatch"] = 4.0
    frame["log10_logical_bytes"] = 3.0
    frame["arithmetic_intensity_analytic"] = 2.0
    frame["candidate_cpu_fraction"] = np.where(frame["cpu_level"].eq("REF"), 1.0, 0.8)
    frame["candidate_gpu_fraction"] = 0.0
    frame["candidate_cpu_is_ref"] = frame["cpu_level"].eq("REF").astype(int)
    frame["candidate_gpu_is_ref"] = 0
    frame["requires_cold_start"] = 1
    return frame


def test_las_cuatro_familias_serializan_y_predicen_probabilidad():
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    import pickle

    frame = _model_frame()
    for family in search.FAMILIES:
        model = models.build_pipeline(
            family, models.default_parameters(family), frame, seed=20260828,
            scale_pos_weight=models.class_weight_ratio(frame.is_optimal) if family == "xgboost" else None,
        )
        model.fit(frame, frame.is_optimal)
        assert len(models.positive_probability(model, frame)) == len(frame)
        assert pickle.loads(pickle.dumps(model)).predict_proba(frame).shape[0] == len(frame)


def test_optuna_reanuda_hasta_total_de_trials_sin_duplicarlos(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    frame = _model_frame()
    first, _ = search.tune_family(
        "logistic", frame, strategy="A", outer_fold="test",
        output_dir=tmp_path, trials=1, latency_warmups=1, latency_repeats=1,
    )
    second, _ = search.tune_family(
        "logistic", frame, strategy="A", outer_fold="test",
        output_dir=tmp_path, trials=1, latency_warmups=1, latency_repeats=1,
    )
    assert first["completed_trials"] == 1
    assert second["completed_trials"] == 1
