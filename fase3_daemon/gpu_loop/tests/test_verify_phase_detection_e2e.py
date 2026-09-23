"""Pruebas locales de `verify_phase_detection_e2e.py` (Bloque C, ítem C3).

Cubre solo la lógica determinista y sin hilos (verificación de checksum,
criterio de éxito, propiedades de `E2EResult`) -- el camino feliz
completo (sondeo NVML real concurrente con un kernel GPU real) es
intrínsecamente una verificación de punta a punta contra hardware, es lo
que corresponde a este ítem del plan, y forzarlo a un test unitario con
hilos y relojes falsos arriesgaría ser fragil sin aportar una garantía
real que el smoke de paccaA100 no dé ya. Ver
`scripts/pacca/hyp_verify_phase_detection_e2e.sbatch`."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import KernelEntry
from fase3_daemon.gpu_loop.loop import GpuFeatures, PhaseBeginEvent
from fase3_daemon.gpu_loop.verify_phase_detection_e2e import E2EResult, _check_success, run_e2e


def _gpu_entry(id_: str = "k") -> KernelEntry:
    return KernelEntry(
        id=id_, suite="TEST", role="dataset", exec_path=f"bin/{id_}",
        binary_checksum={"pacca-a100": "sha256:abc"}, phase_label_hint="compute_bound",
        size_variant="TEST", expected_runtime_seconds=1, warmup_seconds=0.0,
        estimated_memory_bytes=1024, success_check={"type": "exit_code", "expected": 0},
        exec_args="", device="gpu",
        operational_intensity_flops_per_byte=1.0, gpu_precision="fp64",
    )


def test_check_success_exit_code():
    entry = _gpu_entry()
    assert _check_success(entry, 0, "") is True
    assert _check_success(entry, 1, "") is False


def test_check_success_stdout_regex():
    entry = _gpu_entry()
    entry.success_check = {"type": "stdout_regex", "pattern": "Verification = SUCCESSFUL"}
    assert _check_success(entry, 0, "algo\nVerification = SUCCESSFUL\n") is True
    assert _check_success(entry, 0, "algo mas") is False


def test_e2eresult_detected_begin_end_vacio():
    r = E2EResult(process_begin_ns=0, process_end_ns=100, begin_events=[], end_ns_values=[],
                  success=True, exit_code=0)
    assert r.detected_begin is False
    assert r.detected_end is False
    assert r.begin_latency_ns is None
    assert r.end_latency_ns is None


def test_e2eresult_detected_begin_end_con_eventos():
    features = GpuFeatures(gpu_util_pct=50, gpu_mem_util_pct=1, gpu_power_mw=1,
                            gpu_sm_clock_mhz=1, gpu_temperature_c=1)
    r = E2EResult(
        process_begin_ns=1000, process_end_ns=5000,
        begin_events=[PhaseBeginEvent(now_ns=1200, features=features)],
        end_ns_values=[5300], success=True, exit_code=0,
    )
    assert r.detected_begin is True
    assert r.detected_end is True
    assert r.begin_latency_ns == 200   # el sondeo detecto 200ns despues del inicio real
    assert r.end_latency_ns == 300     # y 300ns despues del fin real


def test_run_e2e_nunca_lanza_sin_checksum_verificado(monkeypatch):
    entry = _gpu_entry()
    monkeypatch.setattr(
        "fase3_daemon.gpu_loop.verify_phase_detection_e2e.verify_binary", lambda *_a, **_kw: False,
    )
    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append(args)
        raise AssertionError("run_fn no debe llamarse si el checksum no verifica")

    with pytest.raises(RuntimeError, match="C02"):
        run_e2e(
            entry, node_id="pacca-a100", kernels_root=Path("/kroot"), gpu_index=None,
            poll_interval_s=0.01, activity_threshold_pct=5.0, end_margin_s=0.01,
            poller_warmup_s=0.0, run_fn=fake_run, query_features_fn=lambda: None,
            sleep_fn=lambda _s: None, now_fn=lambda: 0, monotonic_fn=lambda: 0.0,
        )
    assert run_calls == []
