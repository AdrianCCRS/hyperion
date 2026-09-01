"""Test de integración de run_daemon.py::build_daemon_gpu_loop() en modo
--dry-run: policy_table.yaml real (escrito a disco), socket Unix real
(un cliente envía los mismos datagramas que el shim), y se verifica que
efectivamente clasifica y "aplica" (en dry-run, solo registra en log) el
reloj correcto según la política.
"""
import os
import socket
import tempfile
import threading
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.controller import GpuPhaseLabel
from fase3_daemon.gpu_loop.loop import GpuFeatures
from fase3_daemon.run_daemon import build_daemon_gpu_loop


def _fixed_features() -> GpuFeatures:
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        socket_path = os.path.join(tmp_dir, "gpu_phase.sock")

        def classify_by_util(features: GpuFeatures) -> GpuPhaseLabel:
            return GpuPhaseLabel.COMPUTE_BOUND if features.gpu_util_pct > 50 else GpuPhaseLabel.MEMORY_BOUND

        # build_daemon_gpu_loop() en dry-run usa event_listener.listen()
        # internamente, que bloquea hasta max_events -- pero run_daemon no
        # expone max_events (es infinito en producción). Se envuelve en un
        # hilo con timeout y se corta el socket para terminar el generador.
        results = {}

        def target():
            # Monkeypatch puntual: reemplaza listen() por una versión con
            # max_events=1 para que el test termine solo.
            import fase3_daemon.run_daemon as run_daemon_module
            from fase3_daemon.shim import event_listener as el_module

            original_listen = el_module.listen

            def listen_once(*args, **kwargs):
                kwargs["max_events"] = 1
                return original_listen(*args, **kwargs)

            run_daemon_module.event_listener.listen = listen_once
            try:
                decisions = build_daemon_gpu_loop(
                    policy_table_path, gpu_phase_socket=socket_path, gpu_index=None,
                    min_dwell_ns=0, dry_run=True, classify_fn=classify_by_util,
                    query_features_fn=_fixed_features,
                )
                results["decisions"] = decisions
            finally:
                run_daemon_module.event_listener.listen = original_listen

        with caplog.at_level("INFO"):
            server_thread = threading.Thread(target=target, daemon=True)
            server_thread.start()

            for _ in range(200):
                if os.path.exists(socket_path):
                    break
                threading.Event().wait(0.01)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            client.sendto(b"BEGIN,42\n", socket_path)
            client.close()

            server_thread.join(timeout=5.0)

        assert not server_thread.is_alive()
        assert len(results["decisions"]) == 1
        decision = results["decisions"][0]
        # gpu_util_pct=80 -> compute_bound -> política gpu-compute_bound -> 1410 MHz
        assert decision.target_clock_mhz == 1410
        assert any("aplicaría reloj GPU -> 1410" in r.message for r in caplog.records)
