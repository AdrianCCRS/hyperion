from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

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


def _read_text(path: Path) -> str | None:
    """Lee un archivo; centralizarlo facilita sustituirlo en las pruebas."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: set[int] = set()
    for token in cpu_list.split(","):
        try:
            first, last = token.split("-", 1) if "-" in token else (token, token)
            cpus.update(range(int(first), int(last) + 1))
        except ValueError:
            continue
    return sorted(cpus)


def _available_cpus(sys_path: Path, delegated: list[int]) -> list[int]:
    if delegated:
        return delegated
    return sorted(
        int(path.name.removeprefix("cpu"))
        for path in (sys_path / "devices/system/cpu").glob("cpu[0-9]*")
        if path.name.removeprefix("cpu").isdigit()
    )


def _frequency_data(sys_path: Path, cpus: list[int]) -> tuple[str, list[int]]:
    driver = ""
    frequencies: set[int] = set()
    for cpu in cpus:
        cpu_path = sys_path / "devices/system/cpu" / f"cpu{cpu}" / "cpufreq"
        if not driver:
            driver = _read_text(cpu_path / "scaling_driver") or ""
        values = _read_text(cpu_path / "scaling_available_frequencies")
        if values:
            for value in values.split():
                try:
                    frequencies.add(int(value))
                except ValueError:
                    continue
    return driver, sorted(frequencies)


def _rapl_data(sys_path: Path) -> tuple[list[str], bool]:
    rapl_root = sys_path / "class/powercap/intel-rapl"
    domains: list[str] = []
    for domain_path in rapl_root.glob("intel-rapl:*"):
        if not domain_path.is_dir():
            continue
        domains.append(_read_text(domain_path / "name") or domain_path.name)

    root_energy = rapl_root / "intel-rapl:0" / "energy_uj"
    first = _read_text(root_energy)
    if first is None:
        return domains, False
    # ENV-03: dos lecturas separadas por 100 ms, sin escribir en sysfs.
    time.sleep(0.1)
    second = _read_text(root_energy)
    return domains, second is not None and first != second


def _numa_data(sys_path: Path, delegated: list[int]) -> tuple[dict[int, list[int]], dict[int, int]]:
    topology: dict[int, list[int]] = {}
    delegated_nodes: dict[int, int] = {}
    for node_path in (sys_path / "devices/system/node").glob("node[0-9]*"):
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


def detect_environment(delegated_cpus: str, base_sys_path: str = "/sys") -> EnvironmentProfile:
    """
    Lee las capacidades del sistema mediante operaciones de solo lectura.

    ``base_sys_path`` permite usar un sysfs virtual en pruebas; no se escribe
    ningún archivo durante la detección (ENV-01).
    """
    sys_path = Path(base_sys_path)
    delegated = _parse_cpu_list(delegated_cpus)
    cpus = _available_cpus(sys_path, delegated)
    scaling_driver, frequencies = _frequency_data(sys_path, cpus)
    # ENV-02: solo drivers físicos conocidos y más de una frecuencia son válidos.
    freq_capable = (
        scaling_driver in {"intel_pstate", "acpi-cpufreq", "amd-pstate"}
        and len(frequencies) > 1
    )
    rapl_domains, rapl_capable = _rapl_data(sys_path)

    smt_siblings: dict[int, list[int]] = {}
    for cpu in delegated:
        siblings = _read_text(
            sys_path / "devices/system/cpu" / f"cpu{cpu}" / "topology/thread_siblings_list"
        )
        if siblings is not None:
            smt_siblings[cpu] = _parse_cpu_list(siblings)
    numa_cpu_map, delegated_cpu_numa_nodes = _numa_data(sys_path, delegated)
    perf_events = sorted(
        path.name
        for path in (sys_path / "bus/event_source/devices/cpu/events").glob("*")
        if path.is_file()
    )
    gpu_present = any(
        (card / "device").exists()
        for card in (sys_path / "class/drm").glob("card[0-9]*")
    )
    tier = "hpc_sc3" if os.environ.get("SLURM_JOB_ID") else "local"
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
    )
    # Datos complementarios requeridos por ENV-06 y ENV-08, conservando la API pública.
    profile.delegated_cpus = delegated
    profile.numa_cpu_map = numa_cpu_map
    profile.delegated_cpu_numa_nodes = delegated_cpu_numa_nodes
    profile.perf_events_available = perf_events
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
    manifest["not_eligible_for_training_dataset"] = not profile.freq_control_capable
    return manifest


def write_environment_report(profile: EnvironmentProfile, output_dir: str | Path) -> Path:
    """Serializa environment_report.json en el directorio de la campaña (ENV-09)."""
    report = asdict(profile)
    report["delegated_cpus"] = getattr(profile, "delegated_cpus", [])
    report["numa_cpu_map"] = getattr(profile, "numa_cpu_map", {})
    report["delegated_cpu_numa_nodes"] = getattr(profile, "delegated_cpu_numa_nodes", {})
    report["perf_events_available"] = getattr(profile, "perf_events_available", [])
    path = Path(output_dir) / "environment_report.json"
    with path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    return path
