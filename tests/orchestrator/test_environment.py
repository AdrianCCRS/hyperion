from pathlib import Path
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


def test_env_t02_una_frecuencia_no_es_controlable(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="acpi-cpufreq", frecuencias="2400000")
    assert environment.detect_environment("2-5", str(raiz)).freq_control_capable is False


def test_env_t03_driver_desconocido_no_es_controlable(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="hypervisor-virtual")
    assert environment.detect_environment("2-5", str(raiz)).freq_control_capable is False


def test_env_t02_amd_pstate_es_controlable(tmp_path):
    raiz = crear_sysfs(tmp_path, driver="amd-pstate")
    assert environment.detect_environment("2-5", str(raiz)).freq_control_capable is True


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


def test_config_inyectada_define_rutas_y_tier_cloud(tmp_path, monkeypatch):
    raiz = crear_sysfs(tmp_path)
    configuracion_toml = tmp_path / "orchestrator.toml"
    configuracion_toml.write_text(
        f'''[harness]
exec_flag = "--programa"
exec_args_flag = "--argumentos"
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
