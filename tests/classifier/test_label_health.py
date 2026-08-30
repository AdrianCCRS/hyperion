from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.selector import label_health


def _grouped(rows):
    return pd.DataFrame(rows)


def test_etiqueta_degenerada_marca_smoke_only():
    rows = []
    for i in range(20):
        rows.append({"decision_group_id": f"g{i}", "action_id": "cpu:REF",
                      "is_optimal": 1, "margin_edp_pct": 0.5})
        rows.append({"decision_group_id": f"g{i}", "action_id": "cpu:F1",
                      "is_optimal": 0, "margin_edp_pct": 0.5})
    result = label_health.assess_label_health(_grouped(rows))
    assert result["verdict"] == "pipeline_smoke_only"
    assert result["top1_share"] == 1.0
    assert result["effective_classes"] == 1


def test_etiqueta_con_variedad_y_margen_amplio_es_valida():
    rows = []
    actions = ["cpu:REF", "gpu:REF:REF", "gpu:F0:F0", "gpu:F1:F1"]
    for i in range(40):
        winner = actions[i % len(actions)]
        for action in actions:
            rows.append({
                "decision_group_id": f"g{i}", "action_id": action,
                "is_optimal": 1 if action == winner else 0,
                "margin_edp_pct": 15.0,
            })
    result = label_health.assess_label_health(_grouped(rows))
    assert result["verdict"] == "comparison_valid"
    assert result["effective_classes"] == 4
    assert not result["reasons"]


def test_margen_estrecho_marca_smoke_aunque_haya_variedad():
    rows = []
    actions = ["cpu:REF", "gpu:REF:REF", "gpu:F0:F0", "gpu:F1:F1"]
    for i in range(40):
        winner = actions[i % len(actions)]
        for action in actions:
            rows.append({
                "decision_group_id": f"g{i}", "action_id": action,
                "is_optimal": 1 if action == winner else 0,
                "margin_edp_pct": 0.3,
            })
    result = label_health.assess_label_health(_grouped(rows))
    assert result["verdict"] == "pipeline_smoke_only"
    assert any("margen mediano" in reason for reason in result["reasons"])


def test_device_aplastante_frequency_ruidoso():
    """GPU gana por ordenes de magnitud (device claro) pero entre las dos
    frecuencias GPU la diferencia es ruido (frequency angosto)."""
    rows = []
    for i in range(20):
        rows.extend([
            {"decision_group_id": f"g{i}", "action_id": "cpu:REF", "candidate_device": "cpu",
             "is_optimal": 0, "edp_mean": 2.85e-4, "margin_edp_pct": 21800.0},
            {"decision_group_id": f"g{i}", "action_id": "gpu:REF:REF", "candidate_device": "gpu",
             "is_optimal": 1, "edp_mean": 1.297e-5, "margin_edp_pct": 1.16},
            {"decision_group_id": f"g{i}", "action_id": "gpu:F0:F0", "candidate_device": "gpu",
             "is_optimal": 0, "edp_mean": 1.312e-5, "margin_edp_pct": 1.16},
        ])
    result = label_health.assess_label_health(_grouped(rows))
    assert result["device_decision"]["verdict"] == "comparison_valid"
    assert result["device_decision"]["median_margin_pct"] > 100
    assert result["frequency_decision"]["verdict"] == "pipeline_smoke_only"
    assert result["frequency_decision"]["median_margin_pct"] < label_health.MEDIAN_MARGIN_FLOOR_PCT


def test_device_y_frequency_ambos_validos():
    rows = []
    for i in range(20):
        rows.extend([
            {"decision_group_id": f"g{i}", "action_id": "cpu:REF", "candidate_device": "cpu",
             "is_optimal": 0, "edp_mean": 2.0, "margin_edp_pct": 100.0},
            {"decision_group_id": f"g{i}", "action_id": "gpu:REF:REF", "candidate_device": "gpu",
             "is_optimal": 1, "edp_mean": 1.0, "margin_edp_pct": 100.0},
            {"decision_group_id": f"g{i}", "action_id": "gpu:F0:F0", "candidate_device": "gpu",
             "is_optimal": 0, "edp_mean": 2.0, "margin_edp_pct": 100.0},
        ])
    result = label_health.assess_label_health(_grouped(rows))
    assert result["device_decision"]["verdict"] == "comparison_valid"
    assert result["frequency_decision"]["verdict"] == "comparison_valid"


def test_device_y_frequency_ambos_degenerados():
    rows = []
    for i in range(20):
        rows.extend([
            {"decision_group_id": f"g{i}", "action_id": "cpu:REF", "candidate_device": "cpu",
             "is_optimal": 0, "edp_mean": 1.001, "margin_edp_pct": 0.1},
            {"decision_group_id": f"g{i}", "action_id": "gpu:REF:REF", "candidate_device": "gpu",
             "is_optimal": 1, "edp_mean": 1.0, "margin_edp_pct": 0.1},
            {"decision_group_id": f"g{i}", "action_id": "gpu:F0:F0", "candidate_device": "gpu",
             "is_optimal": 0, "edp_mean": 1.0005, "margin_edp_pct": 0.1},
        ])
    result = label_health.assess_label_health(_grouped(rows))
    assert result["device_decision"]["verdict"] == "pipeline_smoke_only"
    assert result["frequency_decision"]["verdict"] == "pipeline_smoke_only"


def test_grupo_con_un_solo_device_se_excluye_de_device_margin():
    """Un grupo sin ambos dispositivos presentes no debe crashear; se
    excluye del calculo de device_decision y queda contado."""
    rows = []
    for i in range(20):
        rows.extend([
            {"decision_group_id": f"g{i}", "action_id": "gpu:REF:REF", "candidate_device": "gpu",
             "is_optimal": 1, "edp_mean": 1.0, "margin_edp_pct": 100.0},
            {"decision_group_id": f"g{i}", "action_id": "gpu:F0:F0", "candidate_device": "gpu",
             "is_optimal": 0, "edp_mean": 2.0, "margin_edp_pct": 100.0},
        ])
    result = label_health.assess_label_health(_grouped(rows))
    assert result["device_decision"]["device_margin_excluded_groups"] == 20
    assert result["device_decision"]["median_margin_pct"] is None
    assert result["device_decision"]["verdict"] == "pipeline_smoke_only"
    # frequency si se puede calcular: hay 2 acciones GPU en el device ganador
    assert result["frequency_decision"]["verdict"] == "comparison_valid"
