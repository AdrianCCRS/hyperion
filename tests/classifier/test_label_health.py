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
