"""Test de integración de run_daemon.py::build_daemon_gpu_loop() en modo
--dry-run: policy_table.yaml real (escrito a disco), fuente de eventos por
sondeo (sin sleep real, sin GPU real -- query_features_fn/sleep_fn
inyectados), y se verifica que efectivamente clasifica y "aplica" (en
dry-run, solo registra en log) el reloj correcto según la política.
"""
from pathlib import Path
import json
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

from fase3_daemon.decision_log import DecisionLogWriter
from fase3_daemon.gpu_loop.controller import GpuPhaseLabel
from fase3_daemon.gpu_loop.loop import GpuFeatures
from fase3_daemon.run_daemon import build_daemon_gpu_loop, pid_alive


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


def test_arm_base_es_rechazado(policy_table_path):
    # 'base' significa, literalmente, no correr este script (Plan_Fase3_Daemon.md
    # §0.1) -- build_daemon_gpu_loop nunca debe aceptarlo como brazo válido.
    with pytest.raises(ValueError, match="base"):
        build_daemon_gpu_loop(
            policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
            classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
            query_features_fn=_active_features, max_events=1, sleep_fn=lambda _s: None,
            arm="base",
        )


def test_arm_inconsistente_con_dry_run_es_rechazado(policy_table_path):
    # arm='activo' con dry_run=True (o viceversa) es un estado contradictorio
    # -- requisito 1 §0.1: sombra hace el mismo trabajo que activo y se
    # detiene justo antes de escribir, nunca al revés silenciosamente.
    with pytest.raises(ValueError, match="inconsistente"):
        build_daemon_gpu_loop(
            policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
            classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
            query_features_fn=_active_features, max_events=1, sleep_fn=lambda _s: None,
            arm="activo",
        )


def test_arm_sombra_escribe_registro_estructurado_con_esquema_completo(policy_table_path, tmp_path):
    def classify_by_util(features: GpuFeatures) -> GpuPhaseLabel:
        return GpuPhaseLabel.COMPUTE_BOUND if features.gpu_util_pct > 50 else GpuPhaseLabel.MEMORY_BOUND

    log_path = tmp_path / "decisions.jsonl"
    with DecisionLogWriter(log_path) as writer:
        build_daemon_gpu_loop(
            policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
            classify_fn=classify_by_util, query_features_fn=_active_features,
            max_events=1, sleep_fn=lambda _s: None, arm="sombra", decision_log=writer,
        )

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["arm"] == "sombra"
    assert record["device"] == "gpu"
    assert record["label"] == "compute_bound"
    assert record["policy_action"] == "actuar"
    assert record["target_freq_khz"] == 1410000
    # brazo sombra: nunca se escribe de verdad, sin importar lo que diga la política
    assert record["written"] is False
    assert record["inference_time_ns"] is not None
    assert record["actuation_time_ns"] is not None
    assert record["features"]["gpu_util_pct"] == 80.0


def test_arm_sombra_escribe_senal_gpu_active_en_cada_transicion(policy_table_path, tmp_path):
    # Bloque C, item C4: run_daemon.py debe escribir True en cada inicio de
    # fase (on_decision) y False en cada fin de fase (on_end) -- no solo en
    # el inicio, o un lector que llegara tarde veria "activa" para siempre.
    readings = iter([80.0, 1.0, 80.0])  # activo(begin1), idle(on_end), activo(begin2)

    def query_features():
        util = next(readings)
        return GpuFeatures(gpu_util_pct=util, gpu_mem_util_pct=1.0, gpu_power_mw=1.0,
                            gpu_sm_clock_mhz=1.0, gpu_temperature_c=1.0)

    writes: list[bool] = []

    class _SpyWriter:
        def write(self, active: bool) -> None:
            writes.append(active)

    build_daemon_gpu_loop(
        policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
        classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
        query_features_fn=query_features, max_events=2, sleep_fn=lambda _s: None,
        arm="sombra", gpu_active_signal=_SpyWriter(),
    )
    assert writes == [True, False, True]


def test_arm_sombra_nunca_instala_manejadores_de_emergencia(policy_table_path, monkeypatch):
    # Bloque C, item C5 (SS0.1 punto 5): "en base y sombra no hay nada que
    # restaurar, y eso tambien debe verificarse: que el daemon no dejo
    # rastro". La forma mas directa de probarlo: en sombra, el daemon ni
    # siquiera REGISTRA un manejador de restauracion -- no hay estado que
    # limpiar porque nunca se toco hardware real.
    import fase3_daemon.run_daemon as run_daemon_module

    def fail_if_called(*_a, **_kw):
        raise AssertionError("no debe llamarse en brazo sombra")

    monkeypatch.setattr(run_daemon_module.environment_module, "detect_environment", fail_if_called)
    monkeypatch.setattr(run_daemon_module.freqctl, "install_emergency_handlers", fail_if_called)

    build_daemon_gpu_loop(
        policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
        classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
        query_features_fn=_active_features, max_events=1, sleep_fn=lambda _s: None,
        arm="sombra",
    )


def test_arm_activo_instala_manejador_que_restaura_gpu(policy_table_path, monkeypatch):
    # Contraparte: en activo SI debe registrarse un manejador, y ese
    # manejador debe llamar a gpu_freqctl.restore_gpu_state() cuando se
    # invoque -- no basta con que exista, tiene que restaurar de verdad.
    import fase3_daemon.run_daemon as run_daemon_module

    fake_env = object()
    detect_calls = []
    # Firma REAL: detect_environment(delegated_cpus, ...) exige el cpuset. Un lambda sin parametros
    # enmascaro durante meses que run_daemon lo llamaba sin argumentos (TypeError al arrancar 'activo').
    monkeypatch.setattr(
        run_daemon_module.environment_module, "detect_environment",
        lambda delegated_cpus, *_a, **_kw: detect_calls.append(delegated_cpus) or fake_env,
    )
    monkeypatch.setattr(
        run_daemon_module.gpu_loop_module, "make_gpu_freqctl_setter", lambda *_a, **_kw: (lambda _mhz: True),
    )
    restore_calls = []
    monkeypatch.setattr(
        run_daemon_module.gpu_freqctl, "restore_gpu_state",
        lambda env, gpu_index=None: restore_calls.append((env, gpu_index)) or True,
    )
    registered = {}
    monkeypatch.setattr(
        run_daemon_module.freqctl, "install_emergency_handlers", lambda fn: registered.update(restore_fn=fn),
    )

    build_daemon_gpu_loop(
        policy_table_path, gpu_index="7", min_dwell_ns=0, dry_run=False,
        classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
        query_features_fn=_active_features, max_events=1, sleep_fn=lambda _s: None,
        arm="activo", delegated_cpus="0-5",
    )
    assert detect_calls == ["0-5"]

    assert "restore_fn" in registered
    assert registered["restore_fn"]() is True
    assert restore_calls == [(fake_env, "7")]


def test_pid_alive_proceso_propio_es_true():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_pid_inexistente_es_false():
    # PID improbable de existir -- si algun dia colisiona en el entorno de
    # CI, el test fallaria de forma obvia (no en silencio).
    assert pid_alive(2**30) is False


def test_should_continue_false_detiene_el_loop_sin_eventos(policy_table_path):
    decisions = build_daemon_gpu_loop(
        policy_table_path, gpu_index=None, min_dwell_ns=0, dry_run=True,
        classify_fn=lambda _f: GpuPhaseLabel.COMPUTE_BOUND,
        query_features_fn=_active_features, sleep_fn=lambda _s: None,
        arm="sombra", should_continue=lambda: False,
    )
    assert decisions == []
