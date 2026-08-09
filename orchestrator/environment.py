from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable

from .config import OrchestratorConfig, SysfsPaths, load_config

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentProfile:
    tier: str                           # local | cloud_own | hpc_sc3
    rapl_capable: bool
    rapl_domains_available: list[str]
    freq_control_capable: bool
    scaling_driver: str
    available_frequencies_khz: list[int]
    numa_nodes: int
    smt_siblings: dict[int, list[int]]  # Mapeo de CPU a sus hermanos SMT
    gpu_present: bool
    gpu_exclusive_hint: bool            # Heurística: local=True, hpc_sc3=Nunca True
    frequency_levels_supported: bool = False
    frequency_write_capable: bool = False
    frequency_control_strategy: str = "unavailable"
    frequency_control_paths: dict[int, dict[str, str]] = field(default_factory=dict)
    rapl_domain_paths: dict[str, str] = field(default_factory=dict)
    # ARC-87: eje de frecuencia de GPU, análogo a available_frequencies_khz/
    # frequency_control_strategy/frequency_write_capable de CPU pero para
    # nvmlDeviceSetGpuLockedClocks (ver gpu_freqctl.py). gpu_available_clocks_mhz
    # queda vacío y gpu_frequency_control_strategy="unavailable" en cualquier
    # nodo sin GPU NVIDIA o sin `nvidia-smi` -- nunca se asume un rango.
    gpu_available_clocks_mhz: list[int] = field(default_factory=list)
    gpu_frequency_control_strategy: str = "unavailable"
    gpu_frequency_write_capable: bool = False


def _read_text(path: Path) -> str | None:
    """Lee un archivo; centralizarlo facilita sustituirlo en las pruebas."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_cpu_list(cpu_list: str) -> list[int]:
    """Acepta tanto listas con rangos separadas por coma ("0-3,5,7-9", el
    formato de cpuset.cpus/CLI) como listas planas separadas por espacio
    ("0 1 2 3", el formato real de freqdomain_cpus/related_cpus en felix —
    confirmado por auditoría F4.2 el 2026-08-01; sin este manejo, esos dos
    archivos siempre parseaban a una lista vacía y el check E10 nunca tenía
    datos con qué bloquear)."""
    cpus: set[int] = set()
    for token in cpu_list.replace(",", " ").split():
        try:
            first, last = token.split("-", 1) if "-" in token else (token, token)
            cpus.update(range(int(first), int(last) + 1))
        except ValueError:
            continue
    return sorted(cpus)


def _available_cpus(sysfs: SysfsPaths, delegated: list[int]) -> list[int]:
    if delegated:
        return delegated
    return sorted(
        int(path.name.removeprefix("cpu"))
        for path in sysfs.cpu_root.glob("cpu[0-9]*")
        if path.name.removeprefix("cpu").isdigit()
    )


def _cpufreq_directory(sysfs: SysfsPaths, cpu: int) -> Path:
    """Obtiene la interfaz por CPU o policyN, según lo que exponga el kernel."""
    per_cpu = sysfs.cpu_root / f"cpu{cpu}" / "cpufreq"
    policy = sysfs.cpu_root / "cpufreq" / f"policy{cpu}"
    return per_cpu if per_cpu.exists() else policy


def _frequency_data(sysfs: SysfsPaths, cpus: list[int]) -> tuple[str, list[int], dict[int, dict[str, str]]]:
    driver = ""
    frequencies: set[int] = set()
    for cpu in cpus:
        cpu_path = _cpufreq_directory(sysfs, cpu)
        if not driver:
            driver = _read_text(cpu_path / "scaling_driver") or ""
        values = _read_text(cpu_path / "scaling_available_frequencies")
        if values:
            for value in values.split():
                try:
                    frequencies.add(int(value))
                except ValueError:
                    continue
        else:
            # ARC-94: intel_pstate en modo activo (confirmado en vivo en
            # paccaA100) nunca expone scaling_available_frequencies --
            # antes de este fallback, freq_capable quedaba en False para
            # SIEMPRE en ese driver, sin importar si min/max eran
            # escribibles. cpuinfo_min_freq/cpuinfo_max_freq son los
            # límites físicos reales del CPU (no una lista inventada de
            # pasos): alcanzan para que freq_capable detecte un rango
            # válido, y freqctl._target_khz ya deriva fracciones continuas
            # sobre min/max para la estrategia bounded_range, así que no
            # hace falta ninguna lista de frecuencias intermedias.
            cpu_min = _read_text(cpu_path / "cpuinfo_min_freq")
            cpu_max = _read_text(cpu_path / "cpuinfo_max_freq")
            if cpu_min and cpu_max:
                try:
                    frequencies.add(int(cpu_min))
                    frequencies.add(int(cpu_max))
                except ValueError:
                    pass
    controls: dict[int, dict[str, str]] = {}
    for cpu in cpus:
        cpu_path = _cpufreq_directory(sysfs, cpu)
        paths = {
            name: str(cpu_path / name)
            for name in ("scaling_governor", "scaling_min_freq", "scaling_max_freq")
            if (cpu_path / name).exists()
        }
        if paths:
            controls[cpu] = paths
    return driver, sorted(frequencies), controls


# D05: los 10 eventos genéricos de PERF_TYPE_HARDWARE definidos por el
# kernel (linux/perf_event.h, PERF_COUNT_HW_*). Nombres tal como los acepta
# `perf stat -e` -- portables entre Intel/AMD, no configs crudos.
_GENERIC_HARDWARE_EVENTS = (
    "instructions", "cycles", "cache-references", "cache-misses",
    "branch-instructions", "branch-misses", "bus-cycles", "ref-cycles",
    "stalled-cycles-frontend", "stalled-cycles-backend",
)

_MULTIPLEXING_MARKER = re.compile(r"not counted|\(\s*\d+\.\d+%\)")


_NATIVE_PROBE_SOURCE = Path(__file__).resolve().parent / "native" / "pmc_multiplex_probe.c"
_native_probe_binary_cache: Path | None = None


def _compiled_native_probe() -> Path | None:
    """ARC-53: compila (una sola vez, resultado cacheado en disco) el
    helper C (native/pmc_multiplex_probe.c) que reproduce el mismo
    criterio de multiplexado de D05 sin depender del CLI `perf` -- ausente
    en paccaA100, confirmado en la auditoría (Auditoria_PaccaA100_Unicartagena.md).
    Reusa el mismo mecanismo perf_event_open ya validado en
    telemetry/src/perf_reader.cpp en vez de reimplementar el struct
    perf_event_attr en Python/ctypes: un ABI de kernel mal replicado de
    memoria fallaría distinto y en silencio, el mismo riesgo que ya
    confirmamos con la codificación cruda de stalled cycles (ARC-52).
    Retorna None si gcc no está disponible o la compilación falla -- ese
    caso se trata igual que "perf ausente", nunca se aprueba D05 por
    default."""
    global _native_probe_binary_cache
    if _native_probe_binary_cache is not None and _native_probe_binary_cache.exists():
        return _native_probe_binary_cache
    if not _NATIVE_PROBE_SOURCE.exists():
        return None
    binary_path = Path(tempfile.gettempdir()) / "hyperion_pmc_multiplex_probe"
    if not binary_path.exists():
        try:
            result = subprocess.run(
                ["gcc", "-O2", "-o", str(binary_path), str(_NATIVE_PROBE_SOURCE)],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0 or not binary_path.exists():
            return None
    _native_probe_binary_cache = binary_path
    return binary_path


def _run_perf_stat_probe(events: tuple[str, ...]) -> str:
    """Corre `perf stat` sobre un `sleep` corto solo para ver si el kernel
    logra programar todos los eventos pedidos sin multiplexar -- no mide
    ningún workload real, D05 no necesita valores, solo saber cuántos
    contadores caben a la vez.

    ARC-53: si el CLI `perf` no está instalado, cae a un helper C propio
    (_compiled_native_probe) que hace exactamente la misma pregunta por
    syscall directo. El formato de retorno se mantiene compatible con
    _MULTIPLEXING_MARKER en ambos casos, así que probe_pmc_count() no
    necesita saber cuál de los dos caminos se usó.
    """
    try:
        result = subprocess.run(
            ["perf", "stat", "-e", ",".join(events), "--", "sleep", "0.2"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return result.stderr
    except FileNotFoundError:
        binary = _compiled_native_probe()
        if binary is None:
            return ""  # ni perf ni gcc disponibles: D05 bloquea con datos ausentes
        native_result = subprocess.run(
            [str(binary), *events],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if native_result.returncode != 0:
            # Un evento no abrió (permiso o no soportado) -- tratar como
            # evidencia de que este N no cabe, igual que "not counted".
            return "not counted"
        return native_result.stdout


def probe_pmc_count(
    max_events: int = len(_GENERIC_HARDWARE_EVENTS),
    *,
    run_perf_stat: Callable[[tuple[str, ...]], str] = _run_perf_stat_probe,
) -> int:
    """D05: número real de contadores de hardware que este nodo puede
    programar SIMULTÁNEAMENTE sin multiplexar, medido empíricamente.

    Nunca asumido por modelo de CPU ni por una tabla fija -- la regla del
    Handoff en Registro_Cambios_Fuera_Plan_Original.md es explícita: "PMCs
    no pueden aprobarse por omisión, deben obtenerse de una fuente real o
    bloquear". Se van pidiendo más eventos genéricos de PERF_TYPE_HARDWARE
    a `perf stat` hasta que aparece evidencia de multiplexado (`<not
    counted>` o la anotación de porcentaje que perf imprime cuando
    time_running < time_enabled); el último N sin esa evidencia es el
    conteo real. Si `perf` no está disponible o falla en la primera
    corrida, retorna 0 (D05 bloquea con datos ausentes, no aprueba por
    omisión).
    """
    best = 0
    for n in range(1, max_events + 1):
        subset = _GENERIC_HARDWARE_EVENTS[:n]
        try:
            output = run_perf_stat(subset)
        except (OSError, subprocess.SubprocessError):
            break
        if _MULTIPLEXING_MARKER.search(output):
            break
        best = n
    return best


_GPU_SUPPORTED_CLOCK_MARKER = re.compile(r"Graphics\s*:\s*(\d+)\s*MHz")


def _run_nvidia_smi_supported_clocks(gpu_index: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "-q", "-d", "SUPPORTED_CLOCKS"],
        capture_output=True, text=True, timeout=10, check=False,
    )


def probe_gpu_clocks(
    gpu_index: int = 0,
    *,
    run_nvidia_smi: Callable[[int], subprocess.CompletedProcess] = _run_nvidia_smi_supported_clocks,
) -> tuple[list[int], str]:
    """ARC-87: relojes de SM soportados por la GPU, vía `nvidia-smi -q -d
    SUPPORTED_CLOCKS` -- confirmado en ARC-62 que esta consulta funciona sin
    privilegios especiales en paccaA100, a diferencia de fijar el reloj
    (`-lgc`, ver gpu_freqctl.py), que sí los exige. Puramente de lectura
    (ENV-01): nunca intenta fijar un reloj durante la detección, solo lista
    los valores discretos que el driver anuncia como soportados.

    Retorna ([], "unavailable") si `nvidia-smi` no existe, falla, o no
    reporta ningún reloj -- nunca se asume un rango a partir del modelo de
    GPU."""
    try:
        result = run_nvidia_smi(gpu_index)
    except (OSError, subprocess.SubprocessError):
        return [], "unavailable"
    if result.returncode != 0:
        return [], "unavailable"
    clocks = sorted({int(value) for value in _GPU_SUPPORTED_CLOCK_MARKER.findall(result.stdout)})
    if not clocks:
        return [], "unavailable"
    return clocks, "locked_clocks"


def _gpu_frequency_write_capable(strategy: str) -> bool:
    """ARC-87: heurística de solo lectura, no una verificación por
    os.access() como E09 en CPU -- GPU no tiene una ruta de archivo sysfs
    que fijar el reloj sea cuestión de permisos de archivo. ARC-62 confirmó
    que `nvmlDeviceSetGpuLockedClocks` (`nvidia-smi -lgc`) exige privilegios
    de root en este driver (Applications Clocks, la vía histórica sin
    privilegios, está deprecada) -- el EUID efectivo es la señal de solo
    lectura más cercana disponible hoy. Sobreescribible por variable de
    entorno para cuando el permiso P4 real se conceda por un mecanismo
    distinto de root literal (ej. un wrapper con sudo específico, o un
    grupo del sistema que futuras versiones del driver puedan habilitar),
    sin tener que volver a tocar este archivo cuando eso se confirme.
    """
    if strategy == "unavailable":
        return False
    override = os.environ.get("HYPERION_GPU_FREQ_WRITE_CAPABLE")
    if override is not None:
        return override.strip().lower() not in ("", "0", "false")
    return os.geteuid() == 0


def _frequency_domain_data(sysfs: SysfsPaths, cpus: list[int]) -> dict[int, list[int]]:
    """Lee la agrupación real de control de frecuencia por CPU (E10).

    En hardware antiguo (ej. acpi-cpufreq por socket) ``freqdomain_cpus``
    puede abarcar más CPUs que los delegados a la campaña, exponiendo el
    riesgo de que fijar la frecuencia afecte cores de otro job. Se prueban
    los tres nombres de archivo que el kernel usa según el driver.
    """
    domains: dict[int, list[int]] = {}
    for cpu in cpus:
        cpu_path = _cpufreq_directory(sysfs, cpu)
        for filename in ("freqdomain_cpus", "related_cpus", "affected_cpus"):
            value = _read_text(cpu_path / filename)
            if value:
                domains[cpu] = _parse_cpu_list(value)
                break
    return domains


def _rapl_data(sysfs: SysfsPaths) -> tuple[list[str], dict[str, str], bool]:
    rapl_root = sysfs.rapl_root
    domain_paths: dict[str, str] = {}
    for domain_path in sorted(rapl_root.rglob("intel-rapl:*")):
        if not domain_path.is_dir():
            continue
        name = _read_text(domain_path / "name") or domain_path.name
        parent_name = _read_text(domain_path.parent / "name")
        alias = f"{name}-{parent_name}" if parent_name else name
        if alias in domain_paths:
            alias = f"{alias}@{domain_path.name}"
        domain_paths[alias] = str(domain_path)

    root_energy = rapl_root / "intel-rapl:0" / "energy_uj"
    first = _read_text(root_energy)
    if first is None:
        return list(domain_paths), domain_paths, False
    # ENV-03: dos lecturas separadas por 100 ms, sin escribir en sysfs.
    time.sleep(0.1)
    second = _read_text(root_energy)
    return list(domain_paths), domain_paths, second is not None and first != second


def _numa_data(sysfs: SysfsPaths, delegated: list[int]) -> tuple[dict[int, list[int]], dict[int, int]]:
    topology: dict[int, list[int]] = {}
    delegated_nodes: dict[int, int] = {}
    for node_path in sysfs.numa_root.glob("node[0-9]*"):
        try:
            node_id = int(node_path.name.removeprefix("node"))
        except ValueError:
            continue
        cpus = _parse_cpu_list(_read_text(node_path / "cpulist") or "")
        topology[node_id] = cpus
        for cpu in delegated:
            if cpu in cpus:
                delegated_nodes[cpu] = node_id
    return topology, delegated_nodes


def detect_environment(
    delegated_cpus: str,
    base_sys_path: str = "/sys",
    *,
    config: OrchestratorConfig | None = None,
) -> EnvironmentProfile:
    """
    Lee las capacidades del sistema mediante operaciones de solo lectura.

    ``base_sys_path`` permite usar un sysfs virtual en pruebas; no se escribe
    ningún archivo durante la detección (ENV-01).
    """
    platform = config or load_config()
    # Un sysfs alternativo siempre prevalece para aislar las pruebas del host.
    sysfs = platform.sysfs if base_sys_path == "/sys" else SysfsPaths.from_base(base_sys_path)
    delegated = _parse_cpu_list(delegated_cpus)
    cpus = _available_cpus(sysfs, delegated)
    scaling_driver, frequencies, control_paths = _frequency_data(sysfs, cpus)
    # ENV-02: solo drivers físicos conocidos y más de una frecuencia son válidos.
    freq_capable = (
        scaling_driver in {"intel_pstate", "acpi-cpufreq", "amd-pstate"}
        and len(frequencies) > 1
    )
    rapl_domains, rapl_domain_paths, rapl_capable = _rapl_data(sysfs)
    strategy = "discrete_bounds" if scaling_driver == "acpi-cpufreq" and freq_capable else (
        "bounded_range" if freq_capable else "unavailable"
    )
    # ARC-94: bounded_range (intel_pstate/amd-pstate) nunca escribe
    # scaling_governor -- pinea min=max=target bajo el governor que ya esté
    # activo (freqctl._apply_bounded). Exigir ese archivo como escribible
    # para esta estrategia hacía que frequency_write_capable dependiera de
    # un permiso que P1 (solo scaling_min_freq/max_freq) nunca solicitó ni
    # iba a conceder, bloqueando el nodo real (paccaA100) incluso con el
    # permiso correcto ya otorgado. discrete_bounds sí necesita el governor
    # escribible (scaling_setspeed solo tiene efecto bajo governor=userspace).
    required_attrs = (
        {"scaling_min_freq", "scaling_max_freq"} if strategy == "bounded_range"
        else {"scaling_governor", "scaling_min_freq", "scaling_max_freq"}
    )
    frequency_write_capable = bool(control_paths) and all(
        os.access(path, os.W_OK)
        for controls in control_paths.values()
        for name, path in controls.items()
        if name in required_attrs
    )

    smt_siblings: dict[int, list[int]] = {}
    for cpu in delegated:
        siblings = _read_text(
            sysfs.cpu_root / f"cpu{cpu}" / "topology/thread_siblings_list"
        )
        if siblings is not None:
            smt_siblings[cpu] = _parse_cpu_list(siblings)
    numa_cpu_map, delegated_cpu_numa_nodes = _numa_data(sysfs, delegated)
    frequency_domain_cpus = _frequency_domain_data(sysfs, delegated)
    perf_events = sorted(
        path.name
        for path in sysfs.perf_events_root.glob("*")
        if path.is_file()
    )
    gpu_present = any(
        (card / "device").exists()
        for card in sysfs.drm_root.glob("card[0-9]*")
    )
    # ARC-87: solo se consulta nvidia-smi si hay una GPU presente -- en un
    # nodo sin GPU esta rama nunca dispara un subprocess innecesario, y
    # probe_gpu_clocks() ya degrada solo a ([], "unavailable") si
    # nvidia-smi no existe o falla, así que el resultado es el mismo de
    # cualquier forma; la guarda es solo para no gastar el subprocess.
    gpu_clocks_mhz, gpu_strategy = probe_gpu_clocks() if gpu_present else ([], "unavailable")
    gpu_frequency_write_capable = _gpu_frequency_write_capable(gpu_strategy)
    detected_tiers = {
        platform.detection.tier_local,
        platform.detection.tier_cloud,
        platform.detection.tier_hpc,
    }
    configured_tier = os.environ.get(platform.detection.tier_override_env_var)
    if configured_tier in detected_tiers:
        tier = configured_tier
    elif os.environ.get(platform.detection.slurm_env_var):
        tier = platform.detection.tier_hpc
    else:
        tier = platform.detection.tier_local
    profile = EnvironmentProfile(
        tier=tier,
        rapl_capable=rapl_capable,
        rapl_domains_available=rapl_domains,
        freq_control_capable=freq_capable,
        scaling_driver=scaling_driver,
        available_frequencies_khz=frequencies,
        numa_nodes=len(numa_cpu_map),
        smt_siblings=smt_siblings,
        gpu_present=gpu_present,
        gpu_exclusive_hint=gpu_present and tier == "local",
        frequency_levels_supported=freq_capable,
        frequency_write_capable=frequency_write_capable,
        frequency_control_strategy=strategy,
        frequency_control_paths=control_paths,
        rapl_domain_paths=rapl_domain_paths,
        gpu_available_clocks_mhz=gpu_clocks_mhz,
        gpu_frequency_control_strategy=gpu_strategy,
        gpu_frequency_write_capable=gpu_frequency_write_capable,
    )
    # Datos complementarios requeridos por ENV-06 y ENV-08, conservando la API pública.
    profile.delegated_cpus = delegated
    profile.numa_cpu_map = numa_cpu_map
    profile.delegated_cpu_numa_nodes = delegated_cpu_numa_nodes
    profile.perf_events_available = perf_events
    profile.frequency_domain_cpus = frequency_domain_cpus
    return profile


def validate_environment_vs_manifest(profile: EnvironmentProfile, manifest: dict) -> dict:
    """Aplica restricciones detectadas y registra las anulaciones en el manifest."""
    rapl = manifest.get("rapl")
    if not isinstance(rapl, dict):
        rapl = {}
        manifest["rapl"] = rapl
    overrides = manifest.setdefault("environment_overrides", {})
    if rapl.get("enabled") is True and not profile.rapl_capable:
        rapl["enabled"] = False
        overrides["rapl_forced_disabled"] = True
        logger.warning("RAPL fue deshabilitado: environment.rapl_capable es false")
    else:
        overrides["rapl_forced_disabled"] = False
    # ENV-05: el perfil determina la elegibilidad sin intentar controlar frecuencia.
    manifest["not_eligible_for_training_dataset"] = not (
        profile.freq_control_capable and profile.frequency_write_capable
    )
    # ENV-07: la política declarada queda disponible para la metadata de campaña.
    metadata = manifest.setdefault("environment_metadata", {})
    metadata["smt_policy"] = manifest.get("smt_policy")
    return manifest


def write_environment_report(profile: EnvironmentProfile, output_dir: str | Path) -> Path:
    """Serializa environment_report.json en el directorio de la campaña (ENV-09)."""
    report = asdict(profile)
    report["delegated_cpus"] = getattr(profile, "delegated_cpus", [])
    report["numa_cpu_map"] = getattr(profile, "numa_cpu_map", {})
    report["delegated_cpu_numa_nodes"] = getattr(profile, "delegated_cpu_numa_nodes", {})
    report["perf_events_available"] = getattr(profile, "perf_events_available", [])
    report["frequency_domain_cpus"] = getattr(profile, "frequency_domain_cpus", {})
    path = Path(output_dir) / "environment_report.json"
    with path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    return path
