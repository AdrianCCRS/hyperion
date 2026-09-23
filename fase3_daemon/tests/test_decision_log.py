"""Pruebas de `fase3_daemon/decision_log.py` (Bloque C, Plan_Fase3_Daemon.md
§0.1 requisito 2). Sin dependencia de GPU/CPU real -- registros construidos
a mano, como haría `run_daemon.py`/`cpu_loop_main.cpp` en producción."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.decision_log import DecisionLogWriter, DecisionRecord


def _record(**overrides) -> DecisionRecord:
    base = dict(
        ts_ns=1, arm="sombra", device="gpu", label="memory_bound", confidence=0.9,
        features={"gpu_util_pct": 80.0}, policy_action="actuar",
        target_freq_khz=1410000, applied_freq_khz=0, written=False, write_failed=False,
    )
    base.update(overrides)
    return DecisionRecord(**base)


def test_arm_invalido_lanza():
    with pytest.raises(ValueError):
        _record(arm="ninguno")


def test_device_invalido_lanza():
    with pytest.raises(ValueError):
        _record(device="tpu")


def test_to_json_es_deserializable_y_conserva_campos():
    rec = _record()
    parsed = json.loads(rec.to_json())
    assert parsed["arm"] == "sombra"
    assert parsed["device"] == "gpu"
    assert parsed["label"] == "memory_bound"
    assert parsed["confidence"] == 0.9
    assert parsed["features"] == {"gpu_util_pct": 80.0}
    assert parsed["target_freq_khz"] == 1410000


def test_writer_una_linea_por_registro(tmp_path):
    path = tmp_path / "sub" / "decisions.jsonl"
    with DecisionLogWriter(path) as w:
        w.write(_record(ts_ns=1))
        w.write(_record(ts_ns=2))

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["ts_ns"] == 1
    assert json.loads(lines[1])["ts_ns"] == 2


def test_writer_append_no_trunca_entre_instancias(tmp_path):
    path = tmp_path / "decisions.jsonl"
    with DecisionLogWriter(path) as w:
        w.write(_record(ts_ns=1))
    with DecisionLogWriter(path) as w:
        w.write(_record(ts_ns=2))

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
