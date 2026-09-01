"""Prueba de punta a punta del listener contra un socket Unix REAL (no
mockeado) -- un cliente envía exactamente los mismos datagramas que
fase3_daemon/shim/blocking_sync_shim.cpp emite ("BEGIN,<ns>\\n"/
"END,<ns>\\n"), para verificar el protocolo de socket sin necesitar CUDA ni
compilar el shim (que no se pudo verificar en este entorno, ver el
docstring de event_listener.py)."""
import os
import socket
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.loop import GpuFeatures
from fase3_daemon.shim.event_listener import listen, parse_message, query_gpu_features


def _fake_features() -> GpuFeatures:
    return GpuFeatures(
        gpu_util_pct=42.0, gpu_mem_util_pct=10.0, gpu_power_mw=55000.0,
        gpu_sm_clock_mhz=1200.0, gpu_temperature_c=65.0,
    )


def test_parse_message_begin_y_end():
    assert parse_message(b"BEGIN,12345\n") == ("BEGIN", 12345)
    assert parse_message(b"END,67890\n") == ("END", 67890)


def test_parse_message_formato_invalido_devuelve_none():
    assert parse_message(b"") is None
    assert parse_message(b"GARBAGE") is None
    assert parse_message(b"BEGIN,no_es_numero\n") is None
    assert parse_message(b"OTRO_KIND,123\n") is None


def test_listen_end_to_end_con_socket_unix_real():
    with tempfile.TemporaryDirectory() as tmp_dir:
        socket_path = os.path.join(tmp_dir, "gpu_phase.sock")

        events = []
        ends = []

        def consume():
            for event in listen(socket_path, query_features_fn=_fake_features,
                                 on_end=ends.append, max_events=2):
                events.append(event)

        server_thread = threading.Thread(target=consume, daemon=True)
        server_thread.start()

        # Espera activa breve a que el server haga bind() antes de enviar --
        # evita una carrera donde el cliente manda antes de que exista el
        # socket (datagram Unix no tiene reintento automático como TCP).
        for _ in range(200):
            if os.path.exists(socket_path):
                break
            threading.Event().wait(0.01)

        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        client.sendto(b"BEGIN,100\n", socket_path)
        client.sendto(b"END,200\n", socket_path)
        client.sendto(b"BEGIN,300\n", socket_path)
        client.close()

        server_thread.join(timeout=5.0)
        assert not server_thread.is_alive(), "el listener no terminó tras max_events=2"

        assert len(events) == 2
        assert events[0].now_ns == 100
        assert events[0].features == _fake_features()
        assert events[1].now_ns == 300
        assert ends == [200]


def test_query_gpu_features_parsea_salida_de_nvidia_smi():
    fake_stdout = "42, 10, 55.0, 1200, 65\n"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_stdout
        features = query_gpu_features()
    assert features == GpuFeatures(
        gpu_util_pct=42.0, gpu_mem_util_pct=10.0, gpu_power_mw=55000.0,
        gpu_sm_clock_mhz=1200.0, gpu_temperature_c=65.0,
    )


def test_query_gpu_features_devuelve_none_si_nvidia_smi_falla():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        assert query_gpu_features() is None


def test_query_gpu_features_devuelve_none_si_nvidia_smi_no_existe():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert query_gpu_features() is None
