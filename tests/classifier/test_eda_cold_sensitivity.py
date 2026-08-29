from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import eda


def _strategy_a(rows):
    frame = pd.DataFrame(rows)
    frame["decision_group_id"] = "A:" + frame["config_id"]
    return frame


def _runs(rows):
    frame = pd.DataFrame(rows)
    frame["region"] = "cold"
    frame["device"] = "cpu"
    return frame


def test_grupo_sin_baja_resolucion_no_reporta_cambio():
    strategy_a = _strategy_a([
        {"config_id": "gemm_N512", "action_id": "cpu:REF", "is_optimal": 1,
         "edp_mean": 1.0, "energy_mean": 1.0, "time_mean": 1.0},
        {"config_id": "gemm_N512", "action_id": "cpu:F0", "is_optimal": 0,
         "edp_mean": 2.0, "energy_mean": 2.0, "time_mean": 1.0},
    ])
    runs = _runs([
        {"config_id": "gemm_N512", "action_id": "cpu:REF", "energy_resolution_status": "nominal"},
        {"config_id": "gemm_N512", "action_id": "cpu:F0", "energy_resolution_status": "nominal"},
    ])
    result = eda._strategy_a_cold_sensitivity(strategy_a, runs)
    assert result["applicable"] is True
    assert result["groups_with_low_resolution_winner"] == 0
    assert result["groups_whose_winner_changes_if_low_resolution_excluded"] == 0
    assert result["groups_with_no_nominal_action"] == 0


def test_ganador_de_baja_resolucion_cambia_al_excluirlo():
    strategy_a = _strategy_a([
        {"config_id": "axpy_N100000", "action_id": "cpu:F0", "is_optimal": 1,
         "edp_mean": 1.0, "energy_mean": 1.0, "time_mean": 1.0},
        {"config_id": "axpy_N100000", "action_id": "cpu:F1", "is_optimal": 0,
         "edp_mean": 1.5, "energy_mean": 1.5, "time_mean": 1.0},
    ])
    runs = _runs([
        {"config_id": "axpy_N100000", "action_id": "cpu:F0", "energy_resolution_status": "low"},
        {"config_id": "axpy_N100000", "action_id": "cpu:F1", "energy_resolution_status": "nominal"},
    ])
    result = eda._strategy_a_cold_sensitivity(strategy_a, runs)
    assert result["groups_with_low_resolution_winner"] == 1
    assert result["groups_whose_winner_changes_if_low_resolution_excluded"] == 1
    assert result["changed_winners"][0]["winner_all_actions"] == "cpu:F0"
    assert result["changed_winners"][0]["winner_nominal_only"] == "cpu:F1"


def test_grupo_totalmente_de_baja_resolucion_no_se_recalcula():
    strategy_a = _strategy_a([
        {"config_id": "gemm_N64", "action_id": "cpu:REF", "is_optimal": 1,
         "edp_mean": 1.0, "energy_mean": 1.0, "time_mean": 1.0},
        {"config_id": "gemm_N64", "action_id": "cpu:F0", "is_optimal": 0,
         "edp_mean": 2.0, "energy_mean": 2.0, "time_mean": 1.0},
    ])
    runs = _runs([
        {"config_id": "gemm_N64", "action_id": "cpu:REF", "energy_resolution_status": "low"},
        {"config_id": "gemm_N64", "action_id": "cpu:F0", "energy_resolution_status": "low"},
    ])
    result = eda._strategy_a_cold_sensitivity(strategy_a, runs)
    assert result["groups_with_no_nominal_action"] == 1
    assert result["groups_whose_winner_changes_if_low_resolution_excluded"] == 0
