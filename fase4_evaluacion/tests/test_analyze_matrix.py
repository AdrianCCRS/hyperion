from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase4_evaluacion.analyze_matrix import score_cell, summarize


def _row(arm, wall, ecpu, egpu, rep=1):
    return {"scope": "gpu", "set": "known", "arm": arm, "wall_s": wall, "e_cpu_j": ecpu, "e_gpu_j": egpu,
            "e_total_j": ecpu + egpu, "edp": (ecpu + egpu) * wall, "rep": rep}


def test_razones_frente_a_base():
    s = {r["arm"]: r for r in summarize([_row("base", 10, 100, 100), _row("activo", 11, 90, 100)])}
    assert s["activo"]["ratio_T"] == 1.1
    assert s["activo"]["ratio_E_cpu"] == 0.9
    assert s["activo"]["ratio_EDP"] == round((190 * 11) / (200 * 10), 3)
    assert s["base"]["ratio_EDP"] == 1.0


def test_puntuacion_atribuye_por_ts_y_cuenta_abstenciones():
    phases = [{"begin_ns": 0, "end_ns": 100, "phase_label_hint": "memory_bound", "kernel_id": "a"},
              {"begin_ns": 200, "end_ns": 300, "phase_label_hint": "compute_bound", "kernel_id": "b"}]
    decisions = [{"ts_ns": 50, "label": "memory_bound"}, {"ts_ns": 60, "label": "compute_bound"},
                 {"ts_ns": 250, "label": "revisar"}, {"ts_ns": 150, "label": "memory_bound"}, {"ts_ns": 10, "label": None}]
    s = score_cell(phases, decisions)
    assert (s["ok"], s["wrong"], s["abst"], s["covered"]) == (1, 1, 1, 2)
