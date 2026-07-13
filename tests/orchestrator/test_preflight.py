from pathlib import Path
import hashlib
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import preflight
from orchestrator.config import SysfsPaths


@pytest.fixture
def env():
    value = SimpleNamespace(
        rapl_capable=False,
        rapl_domains_available=[],
        freq_control_capable=False,
        smt_siblings={2: [2, 3], 3: [2, 3]},
    )
    value.numa_cpu_map = {0: [0, 1, 2, 3]}
    return value


def test_pre_t01_numa_en_dos_nodos():
    result = preflight.check_numa([2, 5], {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})
    assert (result.passed, result.factor_id, result.blocking) == (False, "E04", True)


def test_pre_t02_politica_smt_obligatoria(env):
    result = preflight.check_smt(env, {})
    assert result.passed is False
    assert result.factor_id == "E05"
    assert "Declare smt_policy" in result.message


def test_pre_t03_cgroup_con_procesos(tmp_path):
    (tmp_path / "cgroup.procs").write_text("1234\n")
    result = preflight.check_cgroup_clean(tmp_path)
    assert (result.passed, result.factor_id) == (False, "E03")


def test_pre_t04_governor_distinto(tmp_path):
    path = tmp_path / "cpu2/cpufreq"
    path.mkdir(parents=True)
    (path / "scaling_governor").write_text("schedutil")
    result = preflight.check_governor([2], "userspace", tmp_path)
    assert (result.passed, result.factor_id) == (False, "E07")


def test_pre_t05_carga_externa_supera_umbral():
    result = preflight.check_external_load(0.8, lambda: (0.9, 0.1, 0.1))
    assert (result.passed, result.factor_id) == (False, "E08")


def test_pre_t06_run_id_existente(tmp_path):
    (tmp_path / "run-1").mkdir()
    result = preflight.check_run_id_unique(tmp_path, "run-1")
    assert (result.passed, result.factor_id) == (False, "I07")


def test_pre_t07_rapl_wrap_no_disponible(tmp_path, env):
    env.rapl_capable = True
    result = preflight.check_rapl_wrap(env, rapl_root=tmp_path)
    assert result.passed is True and result.blocking is False
    assert result.observed == {"rapl_wrap_correction": "unavailable", "max_energy_range_uj": None}


def test_pre_t08_binario_inexistente():
    result = preflight.check_binary_exists(SimpleNamespace(exec_path="/ruta/inexistente"))
    assert (result.passed, result.factor_id) == (False, "C01")


def test_pre_t09_checksum_incorrecto(tmp_path):
    binary = tmp_path / "kernel"
    binary.write_text("hola")
    binary.chmod(0o755)
    result = preflight.check_binary_checksum(SimpleNamespace(exec_path=binary, binary_checksum="sha256:incorrecto"))
    assert (result.passed, result.factor_id) == (False, "C02")


def test_pre_t10_calibracion_no_parseable():
    result = preflight.check_calibration_output("sin ancho de banda", "sin flops")
    assert (result.passed, result.factor_id, result.blocking) == (False, "D02", True)


def test_pre_t11_calibracion_no_plausible():
    result = preflight.check_calibration_plausibility(0.5e9, 200e9, 40e9, 400e9)
    assert (result.passed, result.factor_id, result.blocking) == (False, "D03", True)


def test_pre_t12_calibracion_inestable_es_advertencia():
    result = preflight.check_calibration_stability(8.5)
    assert (result.passed, result.blocking) == (False, False)


def test_pre_t13_preflight_completo_correcto(tmp_path, env):
    binary = tmp_path / "kernel"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    checksum = f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}"
    entry = SimpleNamespace(exec_path=binary, binary_checksum=checksum, success_check={"type": "exit_code"}, estimated_memory_bytes=1)
    output = tmp_path / "nueva-campana"
    manifest = {
        "cores": {"delegated_cpus": [2, 3]},
        "smt_policy": "all_threads",
        "rapl": {"enabled": False},
        "gpu": {"enabled": False},
        "output_dir": output,
        "overwrite": False,
        "calibration": ["cal"],
        "kernels": ["dataset"],
    }
    results = preflight.run_campaign_preflight(
        manifest, env, {"cal": entry, "dataset": entry}, sysfs=SysfsPaths.from_base(tmp_path / "sys")
    )
    assert results and all(result.passed for result in results)


def test_preflight_no_corta_despues_de_la_primera_falla(tmp_path, env):
    env.numa_cpu_map = {0: [2], 1: [3]}
    manifest = {
        "cores": {"delegated_cpus": [2, 3]},
        "smt_policy": "all_threads",
        "rapl": {"enabled": False},
        "gpu": {"enabled": False},
        "output_dir": tmp_path / "salida",
        "overwrite": False,
        "calibration": ["cal"],
        "kernels": [],
    }
    inexistente = SimpleNamespace(exec_path=tmp_path / "ausente", binary_checksum="sha256:x", success_check={"type": "exit_code"})
    results = preflight.run_campaign_preflight(
        manifest, env, {"cal": inexistente}, sysfs=SysfsPaths.from_base(tmp_path / "sys")
    )
    failed_ids = {result.factor_id for result in results if not result.passed}
    assert {"E04", "C01", "C02"}.issubset(failed_ids)


def test_e09_requiere_permisos_en_todos_los_cores(monkeypatch):
    monkeypatch.setattr(preflight.os, "access", lambda path, mode: "cpu3" not in str(path))
    result = preflight.check_frequency_write_permission([2, 3], "/sys-falso")
    assert (result.factor_id, result.passed, result.blocking) == ("E09", False, True)


def test_c05_memoria_declarada_no_cabe():
    entry = SimpleNamespace(estimated_memory_bytes=9)
    result = preflight.check_memory_size(entry, ram_bytes=8)
    assert (result.factor_id, result.passed) == ("C05", False)


def test_checks_gpu_mediante_adaptador_mock():
    class GpuMock:
        def active_processes(self): return []
        def persistence_mode(self): return True
        def mig_configuration(self): return "disabled"

    assert all(result.passed for result in preflight.check_gpu(GpuMock()))


def test_post_calibracion_rechaza_datos_no_plausibles():
    calibration = SimpleNamespace(stream_stdout="Best Rate MB/s: 0.5", ert_stdout="Peak GFLOPS: 200", bw_pico_bytes_per_s=0.5e9, p_pico_flops_per_s=200e9)
    results = preflight.run_post_calibration_preflight(
        calibration, SimpleNamespace(cv_pct=8.5), {"bw_pico_bytes_per_s": 40e9, "p_pico_flops_per_s": 400e9}
    )
    assert [(result.factor_id, result.blocking) for result in results] == [("D02", True), ("D03", True), ("D04", False)]
    assert results[1].passed is False and results[2].passed is False
