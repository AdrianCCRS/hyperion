from dataclasses import dataclass
import glob
import os
from pathlib import Path
import time

@dataclass
class EnvironmentProfile:
    """
    Represents an environment profile in the catalog.
    """
    tier: str #local | hpc_sc3
    rapl_capable: bool
    rapl_domains_available: list[str] | None
    freq_control_capable: bool
    scaling_driver: str | None #intel_pstate | acpi_cpufreq | none
    numa_nodes: int | None
    smt_siblings: dict[int, list[int]] | None # {0: [0, 1], 1: [2, 3]} #Solo si smt_siblings != None
    gpu_present: bool
    gpu_exclusive_hint: bool | None # heuristica; en local suele ser True, en hpc_sc3 nunca asumir True

def detect_environment(delegated_cpus: str) -> EnvironmentProfile:
    """
    Detects the environment profile based on the system's characteristics.
    """
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    cpus: set[int] = set()
    for part in delegated_cpus.split(","):
        try:
            start, end = (part.split("-", 1) + [part])[:2] if "-" in part else (part, part)
            cpus.update(range(int(start), int(end) + 1))
        except ValueError:
            continue

    cpu_paths = [f"/sys/devices/system/cpu/cpu{cpu}" for cpu in sorted(cpus)]
    if not cpu_paths:
        cpu_paths = glob.glob("/sys/devices/system/cpu/cpu[0-9]*")

    drivers = [read(f"{cpu}/cpufreq/scaling_driver") for cpu in cpu_paths]
    scaling_driver = next((driver for driver in drivers if driver), None)
    frequencies: set[str] = set()
    for cpu in cpu_paths:
        values = read(f"{cpu}/cpufreq/scaling_available_frequencies")
        if values:
            frequencies.update(values.split())
    freq_control_capable = (
        scaling_driver in {"intel_pstate", "acpi-cpufreq", "amd-pstate"}
        and len(frequencies) > 1
    )

    rapl_domains: list[str] = []
    energy_paths = glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj")
    for energy_path in energy_paths:
        name = read(str(Path(energy_path).with_name("name")))
        rapl_domains.append(name or Path(energy_path).parent.name)
    first_energy = read(energy_paths[0]) if energy_paths else None
    if first_energy is not None:
        time.sleep(0.1)
    second_energy = read(energy_paths[0]) if energy_paths else None
    rapl_capable = (
        first_energy is not None
        and second_energy is not None
        and first_energy != second_energy
    )

    smt_siblings: dict[int, list[int]] = {}
    for cpu in cpus:
        siblings = read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if siblings:
            sibling_ids: list[int] = []
            for part in siblings.split(","):
                start, end = (part.split("-", 1) + [part])[:2] if "-" in part else (part, part)
                sibling_ids.extend(range(int(start), int(end) + 1))
            smt_siblings[cpu] = sibling_ids

    numa_nodes = len(glob.glob("/sys/devices/system/node/node[0-9]*")) or None
    gpu_present = any(
        Path(card, "device").exists()
        for card in glob.glob("/sys/class/drm/card[0-9]*")
    )
    tier = "hpc_sc3" if os.environ.get("SLURM_JOB_ID") else "local"

    return EnvironmentProfile(
        tier=tier,
        rapl_capable=rapl_capable,
        rapl_domains_available=rapl_domains or None,
        freq_control_capable=freq_control_capable,
        scaling_driver=scaling_driver,
        numa_nodes=numa_nodes,
        smt_siblings=smt_siblings or None,
        gpu_present=gpu_present,
        gpu_exclusive_hint=True if gpu_present and tier == "local" else None,
    )
