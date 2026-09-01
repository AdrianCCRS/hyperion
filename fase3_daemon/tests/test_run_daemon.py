"""Test de integración de run_daemon.py::build_daemon_gpu_loop() en modo
--dry-run: policy_table.yaml real (escrito a disco), fuente de eventos por
sondeo (sin sleep real, sin GPU real -- query_features_fn/sleep_fn
inyectados), y se verifica que efectivamente clasifica y "aplica" (en
dry-run, solo registra en log) el reloj correcto según la política.
"""
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.controller import GpuPhaseLabel
from fase3_daemon.gpu_loop.loop import GpuFeatures
from fase3_daemon.run_daemon import build_daemon_gpu_loop


def _active_features() -> GpuFeatures:
    return GpuFeatures(
        gpu_util_pct=80.0, gpu_mem_util_pct=5.0, gpu_power_mw=100000.0,
        gpu_sm_clock_mhz=1410.0, gpu_temperature_c=70.0,
    )


@pytest.fixture
def policy_table_path(tmp_path) -> Path:
    doc = {
        "schema_version": 1,
        "policy": {
            "gpu-compute_bound": {"action": "actuar", "chosen_level": "F0", "resolved_clock_mhz": 1410.0},
            "gpu-memory_bound": {"action": "actuar", "chosen_level": "F4", "resolved_clock_mhz": 765.0},
            "cpu-compute_bound": {"action": "no_actuar", "chosen_level": None},
            "cpu-memory_bound": {"action": "no_actuar", "chosen_level": None},
        },
    }
    path = tmp_path / "policy_table.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_build_daemon_gpu_loop_dry_run_clasifica_y_aplica_segun_politica(policy_table_path, caplog):
    def classify_by_util(features: GpuFeatures) -> GpuPhaseLabel:
        return GpuPhaseLabel.COMPUTE_BOUND if features.gpu_util_pct > 50 else GpuPhaseLabel.MEMORY_BOUND

    with caplog.at_level("INFO"):
        decisions = build_daemon_gpu_loop(
            policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
            classify_fn=classify_by_util, query_features_fn=_active_features,
            max_events=1, sleep_fn=lambda _s: None,
        )

    assert len(decisions) == 1
    decision = decisions[0]
    # gpu_util_pct=80 -> compute_bound -> política gpu-compute_bound -> 1410 MHz
    assert decision.target_clock_mhz == 1410
    assert any("aplicaría reloj GPU -> 1410" in r.message for r in caplog.records)


def test_build_daemon_gpu_loop_gpu_idle_no_genera_ningun_evento(policy_table_path):
    def idle_features() -> GpuFeatures:
        return GpuFeatures(
            gpu_util_pct=0.0, gpu_mem_util_pct=0.0, gpu_power_mw=15000.0,
            gpu_sm_clock_mhz=300.0, gpu_temperature_c=40.0,
        )

    calls = {"n": 0}

    def bounded_idle():
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("el poller no debería seguir sondeando sin actividad ni límite")
        return idle_features()

    with pytest.raises(RuntimeError):
        build_daemon_gpu_loop(
            policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
            classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
            query_features_fn=bounded_idle, max_events=1, sleep_fn=lambda _s: None,
        )
