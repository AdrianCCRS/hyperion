"""Prueba de caos de restauración para run_daemon.py (Bloque C, ítem C5,
Plan_Fase3_Daemon.md §0.1 punto 5).

Mismo patrón que la regresión real ya establecida en
`common/tests/test_freqctl.py::test_frq05_sigint_heredada_como_ignorada_restaura_y_termina`
(que prueba `freqctl.install_emergency_handlers` en aislamiento) -- aquí
se prueba el WIRING propio de `run_daemon.py::_install_restore_handlers`
en un proceso real y separado, bajo la misma herencia adversa de
disposición de señal (SIGINT heredada como `SIG_IGN`, el caso real que
motivó la regresión original). No repite la cobertura de
`gpu_freqctl.restore_gpu_state` en sí (ya cubierta en
`common/tests/test_gpu_freqctl.py`, incluida la garantía de "nunca
lanza") -- ese lado se sustituye aquí por un doble de prueba que solo
confirma que SE LLAMÓ, para poder correr sin GPU real ni depender de que
el candado de reloj de GPU esté reparado (H1, Bloque D, todavía
averiado)."""
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_arm_activo_sigint_heredada_ignorada_restaura_y_termina(tmp_path):
    marker = tmp_path / "restored"
    code = textwrap.dedent(
        f"""
        import signal
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace

        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from common.hpc import gpu_freqctl as gpu_freqctl_module
        from fase3_daemon.run_daemon import _install_restore_handlers

        marker = Path({str(marker)!r})

        def fake_restore(env, gpu_index=None):
            marker.write_text("restored")
            return True

        gpu_freqctl_module.restore_gpu_state = fake_restore

        # Misma herencia adversa que motivo la regresion original de
        # freqctl (una shell no interactiva que ya trae SIGINT ignorada).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _install_restore_handlers(SimpleNamespace(), None)
        print("READY", flush=True)
        while True:
            time.sleep(0.1)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == -signal.SIGINT
        assert marker.read_text() == "restored"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_arm_activo_sigterm_restaura_y_termina(tmp_path):
    # SIGTERM es lo que Slurm envia por defecto al cancelar un job -- el
    # camino real de "alguien nos quita el nodo antes de tiempo", no solo
    # Ctrl+C interactivo.
    marker = tmp_path / "restored"
    code = textwrap.dedent(
        f"""
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace

        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from common.hpc import gpu_freqctl as gpu_freqctl_module
        from fase3_daemon.run_daemon import _install_restore_handlers

        marker = Path({str(marker)!r})

        def fake_restore(env, gpu_index=None):
            marker.write_text("restored")
            return True

        gpu_freqctl_module.restore_gpu_state = fake_restore
        _install_restore_handlers(SimpleNamespace(), None)
        print("READY", flush=True)
        while True:
            time.sleep(0.1)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.terminate()  # SIGTERM
        assert process.wait(timeout=5) == -signal.SIGTERM
        assert marker.read_text() == "restored"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_arm_sombra_no_deja_manejador_registrado(tmp_path):
    # Contraparte del caos: en sombra, matar el proceso no debe invocar
    # NINGUNA restauracion (no hay nada que limpiar -- nunca se registro
    # un manejador porque nunca se toco hardware real). Se verifica
    # arrancando run_daemon.py de verdad en modo sombra contra un
    # policy_table.yaml minimo, matandolo, y confirmando que el marcador
    # de restauracion nunca aparece.
    import yaml

    marker = tmp_path / "restored"
    policy_path = tmp_path / "policy_table.yaml"
    policy_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "policy": {
            "gpu-compute_bound": {"action": "no_actuar", "chosen_level": None},
            "gpu-memory_bound": {"action": "no_actuar", "chosen_level": None},
            "cpu-compute_bound": {"action": "no_actuar", "chosen_level": None},
            "cpu-memory_bound": {"action": "no_actuar", "chosen_level": None},
        },
    }))
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from common.hpc import gpu_freqctl as gpu_freqctl_module
        from fase3_daemon.gpu_loop.controller import GpuPhaseLabel
        from fase3_daemon.run_daemon import build_daemon_gpu_loop

        marker = Path({str(marker)!r})

        def fake_restore(env, gpu_index=None):
            marker.write_text("restored")
            return True

        gpu_freqctl_module.restore_gpu_state = fake_restore

        def idle_features():
            return SimpleNamespace(
                gpu_util_pct=1.0, gpu_mem_util_pct=1.0, gpu_power_mw=1.0,
                gpu_sm_clock_mhz=1.0, gpu_temperature_c=1.0,
            )

        print("READY", flush=True)
        build_daemon_gpu_loop(
            Path({str(policy_path)!r}), gpu_index=None, min_dwell_ns=0, dry_run=True,
            classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
            query_features_fn=idle_features, max_events=None, arm="sombra",
        )
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.terminate()  # SIGTERM: sin manejador propio, el proceso simplemente muere
        assert process.wait(timeout=5) == -signal.SIGTERM
        assert not marker.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
