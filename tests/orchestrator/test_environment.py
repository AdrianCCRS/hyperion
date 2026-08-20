from pathlib import Path
import os
import sys
import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import environment
from orchestrator.config import load_config


def crear_sysfs(tmp_path, *, driver="intel_pstate", frecuencias="1200000 2400000 3600000", rapl=None):
    raiz = tmp_path / "sys"
    for cpu in range(8):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        topologia = raiz / f"devices/system/cpu/cpu{cpu}/topology"
        cpufreq.mkdir(parents=True)
        topologia.mkdir(parents=True)
        (cpufreq / "scaling_driver").write_text(driver)
        (cpufreq / "scaling_available_frequencies").write_text(frecuencias)
        hermano = cpu + 1 if cpu % 2 == 0 else cpu - 1
        (topologia / "thread_siblings_list").write_text(f"{min(cpu, hermano)}-{max(cpu, hermano)}")
    for nodo, cpus in ((0, "0-3"), (1, "4-7")):
        ruta = raiz / f"devices/system/node/node{nodo}"
        ruta.mkdir(parents=True)
        (ruta / "cpulist").write_text(cpus)
    eventos = raiz / "bus/event_source/devices/cpu/events"
    eventos.mkdir(parents=True)
    (eventos / "cycles").write_text("event=0x3c")
    if rapl is not None:
        dominio = raiz / "class/powercap/intel-rapl/intel-rapl:0"
        dominio.mkdir(parents=True)
        (dominio / "name").write_text("package-0")
        (dominio / "energy_uj").write_text(str(rapl))
    return raiz


@pytest.fixture(autouse=True)
def sin_espera(monkeypatch):
    monkeypatch.setattr(environment.time, "sleep", lambda _: None)


def test_env_t01_frecuencia_compatible(tmp_path):
    raiz = crear_sysfs(tmp_path, rapl=10)
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.freq_control_capable is True
    assert perfil.scaling_driver == "intel_pstate"
    assert perfil.available_frequencies_khz == [1200000, 2400000, 3600000]
    assert perfil.smt_siblings[2] == [2, 3]


def crear_sysfs_con_hermanos_externos(tmp_path, *, escribible=True):
    """ARC-163: a diferencia de crear_sysfs() (hermanos dentro del mismo
    rango pequeño 0-7), aquí los hermanos SMT de los CPUs delegados quedan
    FUERA del rango delegado -- igual que en paccaA100 real (delegados
    0-5, hermanos 16-21, ver ARC-162)."""
    raiz = tmp_path / "sys"
    parejas = {0: 16, 1: 17, 2: 18}
    for cpu, hermano in parejas.items():
        for c in (cpu, hermano):
            cpufreq = raiz / f"devices/system/cpu/cpu{c}/cpufreq"
            topologia = raiz / f"devices/system/cpu/cpu{c}/topology"
            cpufreq.mkdir(parents=True)
            topologia.mkdir(parents=True)
            (cpufreq / "scaling_driver").write_text("intel_pstate")
            (cpufreq / "scaling_available_frequencies").write_text("800000 1600000 3200000")
            (cpufreq / "scaling_min_freq").write_text("800000")
            (cpufreq / "scaling_max_freq").write_text("3200000")
            if escribible:
                os.chmod(cpufreq / "scaling_min_freq", 0o644)
                os.chmod(cpufreq / "scaling_max_freq", 0o644)
            (topologia / "thread_siblings_list").write_text(f"{cpu},{hermano}")
    eventos = raiz / "bus/event_source/devices/cpu/events"
    eventos.mkdir(parents=True)
    (eventos / "cycles").write_text("event=0x3c")
    return raiz


def test_arc163_frequency_control_paths_incluye_hermanos_smt_externos(tmp_path):
    raiz = crear_sysfs_con_hermanos_externos(tmp_path)
    perfil = environment.detect_environment("0-2", str(raiz))

    # Los hermanos (16, 17, 18) nunca se declaran como delegados, pero
    # freqctl.py necesita sus rutas de control para poder restringirlos
    # junto con los delegados (ARC-162).
    assert set(perfil.frequency_control_paths.keys()) == {0, 1, 2, 16, 17, 18}
    assert perfil.smt_siblings[0] == [0, 16]


def test_arc163_frequency_write_capable_exige_tambien_los_hermanos(tmp_path):
    raiz = crear_sysfs_con_hermanos_externos(tmp_path, escribible=False)
    for c in (16, 17, 18):
        os.chmod(raiz / f"devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq", 0o444)
    perfil = environment.detect_environment("0-2", str(raiz))
    assert perfil.frequency_write_capable is False


def test_arc163_hermanos_ya_dentro_del_rango_delegado_no_agregan_claves_nuevas(tmp_path):
    # crear_sysfs() (fixture pre-existente) empareja hermanos DENTRO del
    # mismo rango pequeño (cpu2<->cpu3, cpu4<->cpu5) -- a diferencia de
    # crear_sysfs_con_hermanos_externos(), aquí todo hermano de un CPU
    # delegado ya es tambien delegado. sibling_only_cpus queda vacío, así
    # que no se dispara una segunda llamada a _frequency_data() ni se
    # agrega ninguna clave nueva más allá de lo que el fixture ya expone
    # (que no incluye scaling_min/max_freq -- solo confirma que el nuevo
    # camino de ARC-163 no rompe este caso, ya cubierto por
    # test_env_t01_frecuencia_compatible).
    raiz = crear_sysfs(tmp_path)
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.smt_siblings == {2: [2, 3], 3: [2, 3], 4: [4, 5], 5: [4, 5]}
    assert perfil.frequency_control_paths == {}


def test_env_t02_una_frecuencia_no_es_controlable(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq", frecuencias="2400000")
    assert environment.detect_environment("2-5", str(raiz)).freq_control_capable is False


def test_env_t03_driver_desconocido_no_es_controlable(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="hypervisor-virtual")
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.freq_control_capable is False
    assert perfil.frequency_control_strategy == "unavailable"  # ENV-10


def test_env_t01b_intel_pstate_sin_scaling_available_frequencies(tmp_path):
    """ARC-94: intel_pstate en modo activo (confirmado en vivo en
    paccaA100) nunca expone scaling_available_frequencies -- antes de este
    fallback, freq_capable quedaba en False para siempre en ese driver,
    sin importar si scaling_min_freq/scaling_max_freq eran escribibles."""
    raiz = tmp_path / "sys"
    for cpu in range(8):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        topologia = raiz / f"devices/system/cpu/cpu{cpu}/topology"
        cpufreq.mkdir(parents=True)
        topologia.mkdir(parents=True)
        (cpufreq / "scaling_driver").write_text("intel_pstate")
        (cpufreq / "cpuinfo_min_freq").write_text("800000")
        (cpufreq / "cpuinfo_max_freq").write_text("3600000")
        hermano = cpu + 1 if cpu % 2 == 0 else cpu - 1
        (topologia / "thread_siblings_list").write_text(f"{min(cpu, hermano)}-{max(cpu, hermano)}")
    for nodo, cpus in ((0, "0-3"), (1, "4-7")):
        ruta = raiz / f"devices/system/node/node{nodo}"
        ruta.mkdir(parents=True)
        (ruta / "cpulist").write_text(cpus)
    eventos = raiz / "bus/event_source/devices/cpu/events"
    eventos.mkdir(parents=True)
    (eventos / "cycles").write_text("event=0x3c")

    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.freq_control_capable is True
    assert perfil.frequency_control_strategy == "bounded_range"
    assert perfil.available_frequencies_khz == [800000, 3600000]


def test_arc138_no_turbo_limita_el_rango_a_base_frequency(tmp_path):
    raiz = crear_sysfs(tmp_path, rapl=10)
    intel_pstate = raiz / "devices/system/cpu/intel_pstate"
    intel_pstate.mkdir(parents=True)
    (intel_pstate / "no_turbo").write_text("1")
    (intel_pstate / "status").write_text("active")
    for cpu in range(2, 6):
        (raiz / f"devices/system/cpu/cpu{cpu}/cpufreq/base_frequency").write_text("3200000")

    perfil = environment.detect_environment("2-5", str(raiz))

    assert perfil.available_frequencies_khz == [1200000, 2400000, 3200000]
    assert perfil.base_frequency_khz == 3200000
    assert perfil.turbo_hwp_state["no_turbo"] == "1"


def test_env_t02_amd_pstate_es_controlable(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="amd-pstate")
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.freq_control_capable is True
    assert perfil.frequency_control_strategy == "bounded_range"  # ENV-10


def test_env_t04_rapl_ausente(tmp_path):
    raiz = crear_sysfs(tmp_path)
    assert environment.detect_environment("2-5", str(raiz)).rapl_capable is False


def test_env_t05_rapl_estancado(tmp_path):
    raiz = crear_sysfs(tmp_path, rapl=100)
    assert environment.detect_environment("2-5", str(raiz)).rapl_capable is False


def test_env_t06_rapl_cambia_entre_lecturas(tmp_path, monkeypatch):
    raiz = crear_sysfs(tmp_path, rapl=100)
    energia = raiz / "class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    lectura_original = environment._read_text
    valores = iter(["100", "101"])

    def leer(path):
        return next(valores) if path == energia else lectura_original(path)

    monkeypatch.setattr(environment, "_read_text", leer)
    assert environment.detect_environment("2-5", str(raiz)).rapl_capable is True


def test_env_t07_rapl_del_manifest_se_anula(tmp_path, caplog):
    perfil = environment.EnvironmentProfile("local", False, [], True, "intel_pstate", [], 0, {}, False, False)
    manifest = {"rapl": {"enabled": True}}
    with caplog.at_level("WARNING"):
        resultado = environment.validate_environment_vs_manifest(perfil, manifest)
    assert resultado["rapl"]["enabled"] is False
    assert resultado["environment_overrides"]["rapl_forced_disabled"] is True
    assert "RAPL fue deshabilitado" in caplog.text


def test_env_t04_rapl_capable_no_se_anula():
    perfil = environment.EnvironmentProfile("local", True, [], True, "intel_pstate", [], 0, {}, False, False)
    resultado = environment.validate_environment_vs_manifest(perfil, {"rapl": {"enabled": True}})
    assert resultado["rapl"]["enabled"] is True
    assert resultado["environment_overrides"]["rapl_forced_disabled"] is False


def test_env_t05_sin_control_de_frecuencia_no_es_elegible():
    perfil = environment.EnvironmentProfile("local", True, [], False, "", [], 0, {}, False, False)
    resultado = environment.validate_environment_vs_manifest(perfil, {"rapl": {"enabled": False}})
    assert resultado["not_eligible_for_training_dataset"] is True


def test_env_t07_politica_smt_se_conserva_en_metadata():
    perfil = environment.EnvironmentProfile("local", True, [], True, "intel_pstate", [], 0, {}, False, False)
    resultado = environment.validate_environment_vs_manifest(
        perfil, {"rapl": {"enabled": False}, "smt_policy": "all_threads"}
    )
    assert resultado["environment_metadata"]["smt_policy"] == "all_threads"


def test_env_t08_topologia_numa_delegada(tmp_path):
    raiz = crear_sysfs(tmp_path)
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.numa_cpu_map == {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    assert perfil.delegated_cpu_numa_nodes == {2: 0, 3: 0, 4: 1, 5: 1}


def test_env_t08_eventos_perf_disponibles(tmp_path):
    raiz = crear_sysfs(tmp_path)
    assert environment.detect_environment("2-5", str(raiz)).perf_events_available == ["cycles"]


def test_rapl_descubre_subdominios_con_identificadores_unicos(tmp_path):
    raiz = crear_sysfs(tmp_path, rapl=10)
    core = raiz / "class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0"
    core.mkdir()
    (core / "name").write_text("core")
    perfil = environment.detect_environment("2-5", str(raiz))
    assert "core-package-0" in perfil.rapl_domains_available
    assert perfil.rapl_domain_paths["core-package-0"].endswith("intel-rapl:0:0")


def test_entorno_separa_niveles_y_permiso_de_escritura(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq")
    for cpu in range(2, 6):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        for name in ("scaling_governor", "scaling_min_freq", "scaling_max_freq"):
            (cpufreq / name).write_text("valor")
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.frequency_levels_supported is True
    assert perfil.frequency_control_strategy == "discrete_bounds"
    assert perfil.frequency_write_capable is True


def test_arc94_bounded_range_no_exige_governor_escribible(tmp_path):
    """ARC-94: P1 solicita escritura solo sobre scaling_min_freq/max_freq
    -- bounded_range (intel_pstate) nunca escribe scaling_governor
    (freqctl._apply_bounded pinea min=max=target bajo el governor que ya
    esté activo), así que exigirlo escribible bloqueaba
    frequency_write_capable incluso con el permiso correcto ya concedido."""
    raiz = crear_sysfs(tmp_path, rapl=10)  # driver=intel_pstate por defecto
    for cpu in range(2, 6):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        (cpufreq / "scaling_min_freq").write_text("1200000")
        (cpufreq / "scaling_max_freq").write_text("3600000")
        (cpufreq / "scaling_governor").write_text("powersave")
        os.chmod(cpufreq / "scaling_governor", 0o444)  # sin permiso de escritura, como P1 real
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.frequency_control_strategy == "bounded_range"
    assert perfil.frequency_write_capable is True


def test_arc94_discrete_bounds_si_exige_governor_escribible(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq")
    for cpu in range(2, 6):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        for name in ("scaling_governor", "scaling_min_freq", "scaling_max_freq"):
            (cpufreq / name).write_text("valor")
        os.chmod(cpufreq / "scaling_governor", 0o444)
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.frequency_control_strategy == "discrete_bounds"
    assert perfil.frequency_write_capable is False


def test_env_t09_deteccion_no_escribe_archivos(tmp_path, monkeypatch):
    raiz = crear_sysfs(tmp_path)
    escrituras = []

    def prohibir_escritura(*args, **kwargs):
        escrituras.append((args, kwargs))
        raise AssertionError("detect_environment no debe escribir")

    monkeypatch.setattr(Path, "write_text", prohibir_escritura)
    environment.detect_environment("2-5", str(raiz))
    environment.detect_environment("2-5", str(raiz))
    assert escrituras == []


def test_env_t10_reporte_de_entorno(tmp_path):
    raiz = crear_sysfs(tmp_path)
    perfil = environment.detect_environment("2-5", str(raiz))
    salida = tmp_path / "salida"
    salida.mkdir()
    reporte = environment.write_environment_report(perfil, salida)
    contenido = reporte.read_text(encoding="utf-8")
    datos = json.loads(contenido)
    assert reporte.name == "environment_report.json"
    assert set(environment.asdict(perfil)).issubset(datos)
    assert datos["rapl_capable"] is False
    assert datos["rapl_domains_available"] == []
    assert datos["smt_siblings"]["2"] == [2, 3]
    assert datos["gpu_present"] is False
    assert datos["tier"] == "local"
    assert datos["numa_cpu_map"]["0"] == [0, 1, 2, 3]


def test_env_t11_dominio_de_frecuencia_por_socket(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq")
    for cpu in (2, 3, 4, 5):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        (cpufreq / "freqdomain_cpus").write_text("0-7")
    perfil = environment.detect_environment("2-5", str(raiz))
    assert perfil.frequency_domain_cpus == {
        2: [0, 1, 2, 3, 4, 5, 6, 7],
        3: [0, 1, 2, 3, 4, 5, 6, 7],
        4: [0, 1, 2, 3, 4, 5, 6, 7],
        5: [0, 1, 2, 3, 4, 5, 6, 7],
    }


def test_env_t11b_dominio_de_frecuencia_formato_separado_por_espacios(tmp_path):
    """F4.2 (2026-08-01): freqdomain_cpus en felix real usa una lista plana
    separada por espacios ("0 1 2 ... 39"), no el formato con rangos y comas
    ("0-7,32-39") que asumian los tests anteriores. Sin el fix a
    _parse_cpu_list, este formato parseaba a una lista vacia y E10 nunca
    tenia datos con que bloquear."""
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq")
    for cpu in (2, 3, 4, 5):
        cpufreq = raiz / f"devices/system/cpu/cpu{cpu}/cpufreq"
        (cpufreq / "freqdomain_cpus").write_text("0 1 2 3 4 5 6 7 32 33 34 35 36 37 38 39")
    perfil = environment.detect_environment("2-5", str(raiz))
    dominio_esperado = [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
    assert perfil.frequency_domain_cpus == {
        2: dominio_esperado, 3: dominio_esperado, 4: dominio_esperado, 5: dominio_esperado,
    }


def test_env_t12_dominio_de_frecuencia_usa_related_cpus_si_falta_freqdomain(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq")
    (raiz / "devices/system/cpu/cpu2/cpufreq/related_cpus").write_text("2-3")
    perfil = environment.detect_environment("2-3", str(raiz))
    assert perfil.frequency_domain_cpus == {2: [2, 3]}
    assert 3 not in perfil.frequency_domain_cpus


def test_env_t13_reporte_incluye_dominio_de_frecuencia(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq")
    (raiz / "devices/system/cpu/cpu2/cpufreq/freqdomain_cpus").write_text("2-3")
    perfil = environment.detect_environment("2-3", str(raiz))
    salida = tmp_path / "salida"
    salida.mkdir()
    reporte = environment.write_environment_report(perfil, salida)
    datos = json.loads(reporte.read_text(encoding="utf-8"))
    assert datos["frequency_domain_cpus"]["2"] == [2, 3]


def test_config_inyectada_define_rutas_y_tier_cloud(tmp_path, monkeypatch):
    raiz = crear_sysfs(tmp_path)
    configuracion_toml = tmp_path / "orchestrator.toml"
    configuracion_toml.write_text(
        f'''[harness]
exec_flag = "--programa"
exec_args_flag = "--argumentos"
binary_path = "bin/lanzador"
[sysfs]
cpu_root = "{raiz / "devices/system/cpu"}"
rapl_root = "{raiz / "class/powercap/intel-rapl"}"
numa_root = "{raiz / "devices/system/node"}"
perf_events_root = "{raiz / "bus/event_source/devices/cpu/events"}"
drm_root = "{raiz / "class/drm"}"
[detection]
slurm_env_var = "MI_SLURM"
tier_override_env_var = "MI_TIER"
tier_hpc = "cluster"
tier_local = "equipo"
tier_cloud = "nube"
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("MI_TIER", "nube")
    perfil = environment.detect_environment("2-5", config=load_config(configuracion_toml))
    assert perfil.tier == "nube"
    assert perfil.scaling_driver == "intel_pstate"


def _perf_stat_hasta_n_eventos(limite: int):
    """Fake de run_perf_stat: sin multiplexado hasta `limite` eventos,
    reproduce la anotacion real de perf stat (`<not counted>`/`(NN.NN%)`)
    a partir de ahi -- mismo patron observado en felix (N=5 limpio, N=6
    con `branch-misses <not counted>` y porcentajes bajo 100%)."""
    def run_perf_stat(events):
        if len(events) <= limite:
            return ""
        return "     <not counted>      branch-misses                                    (0.00%)\n"
    return run_perf_stat


def test_d05_probe_pmc_count_para_en_el_primer_n_con_multiplexado():
    assert environment.probe_pmc_count(run_perf_stat=_perf_stat_hasta_n_eventos(5)) == 5


def test_d05_probe_pmc_count_sin_multiplexado_hasta_el_maximo():
    assert environment.probe_pmc_count(max_events=4, run_perf_stat=_perf_stat_hasta_n_eventos(99)) == 4


def test_d05_probe_pmc_count_cero_si_perf_no_esta_disponible():
    def run_perf_stat(events):
        raise FileNotFoundError("perf: command not found")
    assert environment.probe_pmc_count(run_perf_stat=run_perf_stat) == 0


def test_d05_probe_pmc_count_detecta_porcentaje_sin_not_counted():
    def run_perf_stat(events):
        if len(events) <= 2:
            return ""
        return "  1,234  cycles                                            (61.38%)\n"
    assert environment.probe_pmc_count(run_perf_stat=run_perf_stat) == 2


# ARC-87: deteccion de capacidades de frecuencia de GPU (probe_gpu_clocks,
# _gpu_frequency_write_capable) y su wiring en detect_environment().

_SUPPORTED_CLOCKS_STDOUT = """GPU 00000000:41:00.0
    Supported Clocks
        Memory                            : 1215 MHz
            Graphics                      : 1410 MHz
            Graphics                      : 1395 MHz
            Graphics                      : 765 MHz
"""


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def test_arc87_probe_gpu_clocks_parsea_relojes_soportados():
    clocks, strategy = environment.probe_gpu_clocks(
        run_nvidia_smi=lambda gpu_index: _FakeCompletedProcess(stdout=_SUPPORTED_CLOCKS_STDOUT)
    )
    assert clocks == [765, 1395, 1410]
    assert strategy == "locked_clocks"


def test_arc87_probe_gpu_clocks_sin_nvidia_smi_es_unavailable():
    def run_nvidia_smi(gpu_index):
        raise FileNotFoundError("nvidia-smi: command not found")
    assert environment.probe_gpu_clocks(run_nvidia_smi=run_nvidia_smi) == ([], "unavailable")


def test_arc87_probe_gpu_clocks_returncode_no_cero_es_unavailable():
    clocks, strategy = environment.probe_gpu_clocks(
        run_nvidia_smi=lambda gpu_index: _FakeCompletedProcess(returncode=1)
    )
    assert clocks == []
    assert strategy == "unavailable"


def test_arc87_probe_gpu_clocks_salida_vacia_es_unavailable():
    clocks, strategy = environment.probe_gpu_clocks(
        run_nvidia_smi=lambda gpu_index: _FakeCompletedProcess(stdout="no supported clocks here")
    )
    assert clocks == []
    assert strategy == "unavailable"


def test_arc87_write_capable_falso_si_strategy_unavailable(monkeypatch):
    monkeypatch.setattr(environment.os, "geteuid", lambda: 0)
    assert environment._gpu_frequency_write_capable("unavailable") is False


def test_arc87_write_capable_usa_euid_por_defecto(monkeypatch):
    monkeypatch.delenv("HYPERION_GPU_FREQ_WRITE_CAPABLE", raising=False)
    monkeypatch.setattr(environment.os, "geteuid", lambda: 0)
    assert environment._gpu_frequency_write_capable("locked_clocks") is True
    monkeypatch.setattr(environment.os, "geteuid", lambda: 1000)
    assert environment._gpu_frequency_write_capable("locked_clocks") is False


def test_arc87_write_capable_override_por_variable_de_entorno(monkeypatch):
    monkeypatch.setattr(environment.os, "geteuid", lambda: 1000)
    monkeypatch.setenv("HYPERION_GPU_FREQ_WRITE_CAPABLE", "1")
    assert environment._gpu_frequency_write_capable("locked_clocks") is True
    monkeypatch.setenv("HYPERION_GPU_FREQ_WRITE_CAPABLE", "0")
    assert environment._gpu_frequency_write_capable("locked_clocks") is False


def test_arc87_detect_environment_sin_gpu_no_llama_nvidia_smi(tmp_path, monkeypatch):
    raiz = crear_sysfs(tmp_path, rapl=10)
    llamado = []
    monkeypatch.setattr(environment, "probe_gpu_clocks", lambda: llamado.append(1) or ([], "unavailable"))

    perfil = environment.detect_environment("2-5", str(raiz))

    assert llamado == []  # gpu_present=False (sin class/drm/card*) -> nunca se gasta el subprocess
    assert perfil.gpu_available_clocks_mhz == []
    assert perfil.gpu_frequency_control_strategy == "unavailable"
    assert perfil.gpu_frequency_write_capable is False


def test_arc87_detect_environment_con_gpu_presente_consulta_relojes(tmp_path, monkeypatch):
    raiz = crear_sysfs(tmp_path, rapl=10)
    tarjeta = raiz / "class/drm/card0/device"
    tarjeta.mkdir(parents=True)
    monkeypatch.setattr(
        environment, "probe_gpu_clocks", lambda: ([765, 1410], "locked_clocks")
    )
    monkeypatch.setattr(environment.os, "geteuid", lambda: 0)

    perfil = environment.detect_environment("2-5", str(raiz))

    assert perfil.gpu_present is True
    assert perfil.gpu_available_clocks_mhz == [765, 1410]
    assert perfil.gpu_frequency_control_strategy == "locked_clocks"
    assert perfil.gpu_frequency_write_capable is True
