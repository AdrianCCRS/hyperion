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
            "projected_campaign_bytes": 0,
            "remaining_core_hours": 1.0,
            "projected_core_hours": 0.5,
        }
    results = preflight.run_campaign_preflight(
        manifest, env, {"cal": entry, "dataset": entry}, sysfs=SysfsPaths.from_base(tmp_path / "sys"), node_profile=SimpleNamespace(pmc_count=0)
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


def _crear_proceso_falso(proc_root: Path, pid: int, *, cpus_allowed_hex: str, cmdline: bytes = b"algo\0"):
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True)
    (proc_dir / "cmdline").write_bytes(cmdline)
    (proc_dir / "status").write_text(f"Name:\tfake\nCpus_allowed:\t{cpus_allowed_hex}\n")


def test_e06_detecta_proceso_ajeno_con_afinidad_solapada(tmp_path):
    # mascara 0xf = cpus 0-3
    _crear_proceso_falso(tmp_path, 4242, cpus_allowed_hex="f")
    result = preflight.detect_foreign_affinity_pids([2, 3], proc_root=tmp_path)
    assert result == [4242]


def test_e06_ignora_hilos_de_kernel_sin_cmdline(tmp_path):
    _crear_proceso_falso(tmp_path, 99, cpus_allowed_hex="f", cmdline=b"")
    assert preflight.detect_foreign_affinity_pids([2, 3], proc_root=tmp_path) == []


def test_e06_excluye_own_pids(tmp_path):
    _crear_proceso_falso(tmp_path, 555, cpus_allowed_hex="ffffffff")
    assert preflight.detect_foreign_affinity_pids([2, 3], proc_root=tmp_path, own_pids=[555]) == []


def test_e06_sin_solapamiento_no_es_ajeno(tmp_path):
    # mascara 0x30 = cpus 4-5, no se solapa con delegated_cpus=[2,3]
    _crear_proceso_falso(tmp_path, 4242, cpus_allowed_hex="30")
    assert preflight.detect_foreign_affinity_pids([2, 3], proc_root=tmp_path) == []


def test_e06_maneja_mascara_con_comas_de_kernels_multisocket(tmp_path):
    # formato real de /proc/<pid>/status en sistemas de muchos cores: grupos
    # de 8 hex separados por coma. "00000000,0000000f" = cpus 0-3.
    _crear_proceso_falso(tmp_path, 777, cpus_allowed_hex="00000000,0000000f")
    assert preflight.detect_foreign_affinity_pids([2], proc_root=tmp_path) == [777]


def test_e06_check_foreign_processes_usa_la_lista_detectada(tmp_path):
    _crear_proceso_falso(tmp_path, 4242, cpus_allowed_hex="f")
    detectados = preflight.detect_foreign_affinity_pids([2, 3], proc_root=tmp_path)
    result = preflight.check_foreign_processes(detectados)
    assert (result.factor_id, result.passed, result.blocking) == ("E06", False, True)
    assert result.observed["foreign_pids"] == [4242]


def test_e10_dominio_de_frecuencia_excede_los_cores_delegados():
    result = preflight.check_frequency_domain([2, 3], {2: [0, 1, 2, 3, 4, 5, 6, 7], 3: [0, 1, 2, 3, 4, 5, 6, 7]})
    assert (result.factor_id, result.passed, result.blocking) == ("E10", False, True)
    assert result.observed["leaking_cpus"][2] == [0, 1, 4, 5, 6, 7]


def test_e10_dominio_cubierto_por_completo_pasa():
    result = preflight.check_frequency_domain([2, 3], {2: [2, 3], 3: [2, 3]})
    assert (result.factor_id, result.passed) == ("E10", True)


def test_e10_sin_datos_de_dominio_no_bloquea():
    result = preflight.check_frequency_domain([2, 3], None)
    assert (result.factor_id, result.passed) == ("E10", True)


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


def test_e01_snapshot_y_deriva_turbo_hwp(tmp_path):
    intel_pstate = tmp_path / "cpu/intel_pstate"
    intel_pstate.mkdir(parents=True)
    (intel_pstate / "no_turbo").write_text("0")
    (intel_pstate / "status").write_text("active")
    snapshot = preflight.check_turbo_hwp(tmp_path / "cpu")
    assert snapshot.passed is True and snapshot.observed["no_turbo"] == "0"
    (intel_pstate / "no_turbo").write_text("1")
    changed = preflight.check_turbo_hwp_unchanged(snapshot.observed, tmp_path / "cpu")
    assert (changed.factor_id, changed.passed, changed.blocking) == ("E01", False, True)


def test_e02_temperatura_y_e06_procesos_ajenos():
    assert preflight.check_temperature(55.0).passed is True
    temperature = preflight.check_temperature(95.0)
    foreign = preflight.check_foreign_processes([1234])
    assert (temperature.factor_id, temperature.passed) == ("E02", False)
    assert (foreign.factor_id, foreign.passed) == ("E06", False)


def test_c03_success_check_valido_e_invalido():
    valid = SimpleNamespace(success_check={"type": "stdout_regex", "pattern": "OK$"})
    invalid = SimpleNamespace(success_check={"type": "stdout_regex", "pattern": "["})
    assert preflight.check_success_check(valid).passed is True
    assert preflight.check_success_check(invalid).passed is False


def test_d01_toolchain_y_checks_de_recursos(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/bin/{tool}")
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(free=10))
    assert preflight.check_toolchain(True).passed is True
    assert preflight.check_rapl_domains(["package"], ["package", "dram"], True).passed is True
    assert preflight.check_disk_space(tmp_path / "salida", 11).passed is False
    assert preflight.check_perf_counter_capacity(["cycles", "instructions"], 1).passed is False
    assert preflight.check_core_hour_budget(1.0, 2.0).passed is False


def test_gpu_reporta_actividad_y_estado_indisponible():
    class GpuConProblemas:
        def active_processes(self): return [4321]
        def persistence_mode(self): return None
        def mig_configuration(self): return None

    results = preflight.check_gpu(GpuConProblemas())
    assert [(result.factor_id, result.passed) for result in results] == [("G01", False), ("G02", False), ("G03", False)]


def test_caminos_validos_e05_e07_d02_y_preflight_reducido(tmp_path):
    governor = tmp_path / "cpu2/cpufreq"
    governor.mkdir(parents=True)
    (governor / "scaling_governor").write_text("userspace")
    binary = tmp_path / "kernel"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    checksum = f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}"
    entry = SimpleNamespace(exec_path=binary, binary_checksum=checksum, success_check={"type": "exit_code"}, estimated_memory_bytes=1)
    manifest = {"cores": {"delegated_cpus": [2]}, "smt_policy": "all_threads", "output_dir": tmp_path / "salida", "overwrite": False}
    assert preflight.check_governor([2], "userspace", tmp_path).passed is True
    assert preflight.check_calibration_output("Best Rate MB/s: 42", "Peak GFLOPS: 350").passed is True
    results = preflight.run_reduced_preflight(manifest, None, entry, "run-1", cpu_root=tmp_path, load_reader=lambda: (0.1, 0.0, 0.0))
    assert all(result.passed for result in results)


def test_d04_es_advertencia_con_d02_y_d03_validos():
    calibration = SimpleNamespace(stream_stdout="Best Rate MB/s: 40", ert_stdout="Peak GFLOPS: 400", bw_pico_bytes_per_s=40e9, p_pico_flops_per_s=400e9)
    results = preflight.run_post_calibration_preflight(
        calibration, SimpleNamespace(cv_pct=8.5), {"bw_pico_bytes_per_s": 40e9, "p_pico_flops_per_s": 400e9}
    )
    assert results[0].passed and results[1].passed
    assert (results[2].factor_id, results[2].passed, results[2].blocking) == ("D04", False, False)


def test_nivel_nativo_no_exige_governor_ni_permiso_de_frecuencia(tmp_path, env):
    binary = tmp_path / "kernel"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    entry = SimpleNamespace(
        exec_path=binary,
        binary_checksum=f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}",
        success_check={"type": "exit_code"},
        estimated_memory_bytes=1,
    )
    env.freq_control_capable = True
    manifest = {
        "cores": {"delegated_cpus": [2]}, "smt_policy": "all_threads",
        "rapl": {"enabled": False}, "gpu": {"enabled": False},
        "output_dir": tmp_path / "campaign", "overwrite": False,
        "frequency_levels": [{"id": "REF", "mode": "native_governor"}],
        "calibration": ["cal"], "kernels": [], "projected_campaign_bytes": 0,
        "remaining_core_hours": 1.0, "projected_core_hours": 0.5,
    }
    results = preflight.run_campaign_preflight(
        manifest, env, {"cal": entry}, sysfs=SysfsPaths.from_base(tmp_path / "sys"), node_profile=SimpleNamespace(pmc_count=0)
    )
    assert {result.factor_id for result in results}.isdisjoint({"E07", "E09"})
