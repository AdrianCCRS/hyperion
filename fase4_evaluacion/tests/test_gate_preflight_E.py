import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase4_evaluacion.gate_preflight_E import check, check_noturbo

S = 1_000_000_000


def _make(tmp, mem_s=40, memory_decision=True, arms=("base", "base_noturbo", "activo_gpu", "activo_gpu_cpuobs")):
    rows = []
    for st in ("known", "unseen"):
        for arm in arms:
            cell = f"gpumem_{st}_{arm}_1"
            d = tmp / "cells" / cell
            d.mkdir(parents=True)
            ph = [{"begin_ns": 0, "end_ns": mem_s * S, "phase_label_hint": "memory_bound", "kernel_id": "m1"},
                  {"begin_ns": (mem_s + 1) * S, "end_ns": (mem_s + 11) * S, "phase_label_hint": "compute_bound", "kernel_id": "c1"}]
            (d / "phases.jsonl").write_text("\n".join(json.dumps(p) for p in ph))
            if arm.startswith("activo"):
                dec = [{"ts_ns": 10 * S, "label": "memory_bound" if memory_decision else "revisar", "written": memory_decision}]
                (d / "gpu_decisions.jsonl").write_text("\n".join(json.dumps(x) for x in dec))
            rows.append({"cell": cell, "scope": "gpumem", "set": st, "arm": arm, "rep": 1, "wall_s": 1, "e_cpu_j": 1, "e_gpu_j": 1,
                         "app_rc": 0, "rc_cpu_daemon": "NA", "rc_gpu_daemon": "NA", "state_ok": 1})
    with (tmp / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def test_pasa_con_todo_en_orden(tmp_path):
    _make(tmp_path)
    assert check(tmp_path) == []


def test_falla_si_la_fase_de_memoria_es_corta(tmp_path):
    _make(tmp_path, mem_s=18)
    assert any("dura" in p for p in check(tmp_path))


def test_falla_si_el_agente_se_abstiene(tmp_path):
    _make(tmp_path, memory_decision=False)
    assert any("F1 aplicado" in p for p in check(tmp_path))


def test_falla_si_falta_un_brazo(tmp_path):
    _make(tmp_path, arms=("base", "activo_gpu", "activo_gpu_cpuobs"))
    assert any("base_noturbo" in p for p in check(tmp_path))


def test_noturbo_no_depende_de_las_aplicaciones_de_e(tmp_path):
    _make(tmp_path, memory_decision=False)  # E fallaria, el complemento sin turbo no
    assert check(tmp_path) != []
    assert check_noturbo(tmp_path) == []
