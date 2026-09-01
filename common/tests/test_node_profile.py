from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hpc import node_profile
from common.hpc.config import SysfsPaths


CPUINFO_2S_2C_2T = "\n\n".join(
    f"processor\t: {logical}\nmodel name\t: Intel(R) Xeon(R) CPU X7560 @ 2.27GHz\n"
    f"physical id\t: {socket}\ncore id\t: {core}"
    for logical, (socket, core) in enumerate(
        [(s, c) for s in range(2) for c in range(2) for _ in range(2)]
    )
) + "\n"


def _write_cache(sys_root: Path, cpu: int, *, l1_kb=32, l2_kb=256, llc_kb=24576, shared: str,
                  llc_line_size=64) -> None:
    cache_root = sys_root / f"devices/system/cpu/cpu{cpu}/cache"
    entries = [
        ("index0", "1", "Data", f"{l1_kb}K", str(cpu), 64),
        ("index1", "1", "Instruction", f"{l1_kb}K", str(cpu), 64),
        ("index2", "2", "Unified", f"{l2_kb}K", str(cpu), 64),
        ("index3", "3", "Unified", f"{llc_kb}K", shared, llc_line_size),
    ]
    for name, level, kind, size, shared_list, line_size in entries:
        index_dir = cache_root / name
        index_dir.mkdir(parents=True)
        (index_dir / "level").write_text(level)
        (index_dir / "type").write_text(kind)
        (index_dir / "size").write_text(size)
        (index_dir / "shared_cpu_list").write_text(shared_list)
        (index_dir / "coherency_line_size").write_text(str(line_size))


def test_cal07_build_node_profile_es_solo_lectura_y_agrega_datos_existentes(tmp_path):
    sys_root = tmp_path / "sys"
    for cpu in (2, 3):
        _write_cache(sys_root, cpu, shared="2-3")
    cpuinfo_path = tmp_path / "cpuinfo"
    cpuinfo_path.write_text(CPUINFO_2S_2C_2T)

    env = SimpleNamespace(
        available_frequencies_khz=[1064000, 1330000, 2261000],
        numa_nodes=2,
        scaling_driver="acpi-cpufreq",
        perf_events_available=["cycles", "instructions"],
        rapl_domains_available=[],
        pmc_count=4,
    )

    profile = node_profile.build_node_profile(
        env, [2, 3],
        node_id="felix-sc3",
        hostname="felix",
        proc_cpuinfo_path=cpuinfo_path,
        sysfs=SysfsPaths.from_base(sys_root),
    )

    assert profile.node_id == "felix-sc3"
    assert profile.cpu_model == "Intel(R) Xeon(R) CPU X7560 @ 2.27GHz"
    assert profile.sockets == 2
    assert profile.cores_total == 4  # 2 sockets x 2 cores
    assert profile.threads_per_core == 2  # 8 logicos / 4 fisicos
    assert profile.numa_nodes == 2
    assert profile.cache_l1_kb == 32
    assert profile.cache_l2_kb == 256
    assert profile.cache_llc_kb == 24576
    assert profile.cache_llc_shared is True  # shared_cpu_list "2-3" -> 2 cpus
    assert profile.cache_line_size_bytes == 64
    assert profile.freq_min_khz == 1064000
    assert profile.freq_max_khz == 2261000
    assert profile.scaling_driver == "acpi-cpufreq"
    assert profile.perf_events_supported == ("cycles", "instructions")
    assert profile.pmc_count == 4
    # Nada se escribio: los archivos fuente conservan su contenido original.
    assert (sys_root / "devices/system/cpu/cpu2/cache/index3/size").read_text() == "24576K"


def test_cal07_llc_no_compartida(tmp_path):
    sys_root = tmp_path / "sys"
    _write_cache(sys_root, 0, shared="0")
    cpuinfo_path = tmp_path / "cpuinfo"
    cpuinfo_path.write_text(CPUINFO_2S_2C_2T)
    env = SimpleNamespace(available_frequencies_khz=[], numa_nodes=1, scaling_driver="")

    profile = node_profile.build_node_profile(
        env, [0], node_id="n1", proc_cpuinfo_path=cpuinfo_path, sysfs=SysfsPaths.from_base(sys_root)
    )

    assert profile.cache_llc_shared is False
    assert profile.freq_min_khz == 0
    assert profile.freq_max_khz == 0


def test_cal08_cal11_roundtrip_json(tmp_path):
    profile = node_profile.NodeProfile(
        node_id="felix-sc3", hostname="felix", cpu_model="X7560", sockets=4, cores_total=32,
        threads_per_core=2, numa_nodes=4, cache_l1_kb=32, cache_l2_kb=256, cache_llc_kb=24576,
        cache_llc_shared=True, cache_line_size_bytes=64, freq_min_khz=1064000, freq_max_khz=2261000,
        scaling_driver="acpi-cpufreq", perf_events_supported=("cycles",), rapl_domains_available=(), pmc_count=4,
    )
    path = node_profile.write_node_profile(profile, tmp_path)
    assert path.name == "node_profile.json"

    loaded = node_profile.load_node_profile(tmp_path)
    assert loaded == profile
