from dataclasses import asdict, dataclass, field
import glob
import json
import os
from pathlib import Path
import time


@dataclass
class EnvironmentProfile:
    """Read-only snapshot of the node capabilities used by a campaign."""

    tier: str
    rapl_capable: bool
    rapl_domains_available: list[str] | None
    freq_control_capable: bool
    scaling_driver: str | None
    numa_nodes: int | None
    smt_siblings: dict[int, list[int]] | None
    gpu_present: bool
    gpu_exclusive_hint: bool | None
    delegated_cpus: list[int] = field(default_factory=list)
    numa_cpu_map: dict[int, list[int]] = field(default_factory=dict)
    delegated_cpu_numa_nodes: dict[int, int] = field(default_factory=dict)
    smt_policy: str = "all_threads"
    perf_events_available: list[str] = field(default_factory=list)


def _parse_cpu_list(cpu_list: str) -> list[int]:
    cpus: set[int] = set()
    for part in cpu_list.split(","):
        try:
            start, end = (part.split("-", 1) + [part])[:2] if "-" in part else (part, part)
            cpus.update(range(int(start), int(end) + 1))
        except ValueError:
            continue
    return sorted(cpus)


def detect_environment(
    delegated_cpus: str, *, smt_policy: str = "all_threads"
) -> EnvironmentProfile:
    """Detect capabilities with read-only sysfs/procfs access only (ENV-01)."""
    if smt_policy not in {"all_threads", "one_thread_per_physical_core"}:
        raise ValueError("ENV-07: política SMT no soportada")

    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    cpus = _parse_cpu_list(delegated_cpus)
    cpu_paths = [f"/sys/devices/system/cpu/cpu{cpu}" for cpu in cpus]
    if not cpu_paths:
        cpu_paths = glob.glob("/sys/devices/system/cpu/cpu[0-9]*")

    drivers = [read(f"{cpu}/cpufreq/scaling_driver") for cpu in cpu_paths]
    scaling_driver = next((driver for driver in drivers if driver), None)
    frequencies: set[str] = set()
    for cpu in cpu_paths:
        values = read(f"{cpu}/cpufreq/scaling_available_frequencies")
        if values:
            frequencies.update(values.split())
    # ENV-02: only known hardware drivers with multiple frequencies qualify.
    freq_control_capable = (
        scaling_driver in {"intel_pstate", "acpi-cpufreq", "amd-pstate"}
        and len(frequencies) > 1
    )

    rapl_domains: list[str] = []
    energy_paths = glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj")
    for energy_path in energy_paths:
        name = read(str(Path(energy_path).with_name("name")))
        rapl_domains.append(name or Path(energy_path).parent.name)
    rapl_root_energy = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    first_energy = read(rapl_root_energy)
    if first_energy is not None:
        # ENV-03: minimal synthetic CPU load followed by the required 100 ms delay.
        deadline = time.perf_counter() + 0.005
        while time.perf_counter() < deadline:
            pass
        time.sleep(0.1)
    second_energy = read(rapl_root_energy)
    rapl_capable = (
        first_energy is not None
        and second_energy is not None
        and first_energy != second_energy
    )

    # ENV-07: record every delegated logical CPU's sibling set and policy.
    smt_siblings: dict[int, list[int]] = {}
    for cpu in cpus:
        siblings = read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if siblings:
            smt_siblings[cpu] = _parse_cpu_list(siblings)

    # ENV-06: preserve full NUMA topology and the placement of delegated CPUs.
    numa_cpu_map: dict[int, list[int]] = {}
    delegated_cpu_numa_nodes: dict[int, int] = {}
    for node_path in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            node = int(Path(node_path).name.removeprefix("node"))
        except ValueError:
            continue
        node_cpus = _parse_cpu_list(read(f"{node_path}/cpulist") or "")
        numa_cpu_map[node] = node_cpus
        for cpu in cpus:
            if cpu in node_cpus:
                delegated_cpu_numa_nodes[cpu] = node

    # ENV-08: PMU aliases from sysfs are the real supported perf event subset.
    perf_events_available = sorted(
        Path(event_path).name
        for event_path in glob.glob("/sys/bus/event_source/devices/cpu/events/*")
        if Path(event_path).is_file()
    )
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
        numa_nodes=len(numa_cpu_map) or None,
        smt_siblings=smt_siblings or None,
        gpu_present=gpu_present,
        gpu_exclusive_hint=True if gpu_present and tier == "local" else None,
        delegated_cpus=cpus,
        numa_cpu_map=numa_cpu_map,
        delegated_cpu_numa_nodes=delegated_cpu_numa_nodes,
        smt_policy=smt_policy,
        perf_events_available=perf_events_available,
    )


def campaign_environment_metadata(
    profile: EnvironmentProfile, *, rapl_enabled: bool
) -> dict[str, object]:
    """Apply the environment constraints that campaign metadata must retain."""
    # ENV-04: an unsupported manifest request is explicitly overridden.
    effective_rapl_enabled = rapl_enabled and profile.rapl_capable
    # ENV-05: no frequency control means exclusion from the training dataset.
    return {
        "rapl_enabled": effective_rapl_enabled,
        "rapl_forced_disabled": rapl_enabled and not profile.rapl_capable,
        "not_eligible_for_training_dataset": not profile.freq_control_capable,
        "smt_policy": profile.smt_policy,
    }


def write_environment_report(profile: EnvironmentProfile, output_dir: str | Path) -> Path:
    """Serialize the ENV-09 campaign artifact in its already-created output dir."""
    report_path = Path(output_dir) / "environment_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(asdict(profile), report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    return report_path
