from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .config import SysfsPaths


@dataclass(frozen=True)
class NodeProfile:
    """CAL-08: hardware/topology profile, reorganized from data environment.py
    and /proc/cpuinfo already read (CAL-07: build_node_profile does not touch
    hardware beyond reads already performed elsewhere)."""

    node_id: str
    hostname: str
    cpu_model: str
    sockets: int
    cores_total: int
    threads_per_core: int
    numa_nodes: int
    cache_l1_kb: int
    cache_l2_kb: int
    cache_llc_kb: int
    cache_llc_shared: bool
    # POST-10: fuente real de LLC_LINE_SIZE_BYTES para bytes_moved_window en
    # postprocess.py. Nunca se asume 64 bytes sin leerlo de sysfs.
    cache_line_size_bytes: int
    freq_min_khz: int
    freq_max_khz: int
    scaling_driver: str
    perf_events_supported: tuple[str, ...] = ()
    rapl_domains_available: tuple[str, ...] = ()
    # Consumido por preflight.check_perf_counter_capacity (D05); no forma
    # parte del NodeProfile de la guía técnica pero ya era parte del
    # contrato existente con preflight.py, así que se conserva.
    pmc_count: int = 0


def _parse_cpuinfo(text: str) -> tuple[str, int, int, int]:
    """Devuelve (cpu_model, sockets, cores_total, threads_per_core).

    cores_total cuenta pares (physical_id, core_id) únicos (núcleos físicos);
    threads_per_core = logical / cores_total. Si el /proc/cpuinfo no declara
    physical id/core id (p. ej. contenedores restringidos), se degrada a
    cores_total=logical y threads_per_core=1 en vez de fallar.
    """
    model = ""
    sockets: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    logical = 0
    physical_id: int | None = None
    core_id: int | None = None

    def _flush() -> None:
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))

    for line in text.splitlines():
        if not line.strip():
            _flush()
            physical_id = None
            core_id = None
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            logical += 1
        elif key == "model name" and not model:
            model = value
        elif key == "physical id":
            try:
                physical_id = int(value)
                sockets.add(physical_id)
            except ValueError:
                pass
        elif key == "core id":
            try:
                core_id = int(value)
            except ValueError:
                pass
    _flush()

    cores_total = len(pairs) if pairs else logical
    sockets_total = len(sockets) if sockets else 1
    threads_per_core = (logical // cores_total) if cores_total else 1
    return model, sockets_total, cores_total, max(threads_per_core, 1)


def _parse_cache_size_kb(text: str) -> int | None:
    text = text.strip().upper()
    if not text:
        return None
    try:
        if text.endswith("K"):
            return int(text[:-1])
        if text.endswith("M"):
            return int(text[:-1]) * 1024
        return int(text) // 1024
    except ValueError:
        return None


def _cache_data(sysfs: SysfsPaths, cpus: Iterable[int]) -> tuple[int, int, int, bool, int]:
    """Lee /sys/devices/system/cpu/cpu*/cache/index*/ (solo lectura, CAL-07).

    Devuelve (l1_kb, l2_kb, llc_kb, llc_shared, llc_line_size_bytes): el
    tamaño de cada nivel es el declarado por una sola instancia
    representativa (todas las instancias del mismo nivel reportan el mismo
    tamaño por diseño de la cache); llc es el nivel más alto observado;
    llc_shared es True si su shared_cpu_list abarca más de un CPU lógico;
    llc_line_size_bytes viene de coherency_line_size del mismo índice LLC
    (POST-10: nunca se asume 64 bytes sin leerlo).
    """
    l1_kb = l2_kb = llc_kb = 0
    llc_level = -1
    llc_shared = False
    llc_line_size_bytes = 0
    seen_levels: set[int] = set()
    for cpu in cpus:
        cache_root = sysfs.cpu_root / f"cpu{cpu}" / "cache"
        if not cache_root.exists():
            continue
        for index_dir in sorted(cache_root.glob("index*")):
            level_text = _read_text(index_dir / "level")
            type_text = _read_text(index_dir / "type") or ""
            if level_text is None or type_text == "Instruction":
                continue
            try:
                level = int(level_text)
            except ValueError:
                continue
            size_kb = _parse_cache_size_kb(_read_text(index_dir / "size") or "")
            if size_kb is None:
                continue
            if level == 1 and level not in seen_levels:
                l1_kb = size_kb
            elif level == 2 and level not in seen_levels:
                l2_kb = size_kb
            if level >= max(llc_level, 0):
                shared_list = _read_text(index_dir / "shared_cpu_list") or ""
                shared_cpus = _parse_cpu_range(shared_list)
                if level > llc_level or size_kb > llc_kb:
                    llc_level = level
                    llc_kb = size_kb
                    llc_shared = len(shared_cpus) > 1
                    line_size_text = _read_text(index_dir / "coherency_line_size")
                    try:
                        llc_line_size_bytes = int(line_size_text) if line_size_text else 0
                    except ValueError:
                        llc_line_size_bytes = 0
            seen_levels.add(level)
    return l1_kb, l2_kb, llc_kb, llc_shared, llc_line_size_bytes


def _parse_cpu_range(text: str) -> list[int]:
    cpus: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            first, last = token.split("-", 1) if "-" in token else (token, token)
            cpus.update(range(int(first), int(last) + 1))
        except ValueError:
            continue
    return sorted(cpus)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def build_node_profile(
    env: Any,
    delegated_cpus: Iterable[int],
    *,
    node_id: str,
    hostname: str = "",
    proc_cpuinfo_path: str | Path = "/proc/cpuinfo",
    sysfs: SysfsPaths | None = None,
) -> NodeProfile:
    """CAL-07: solo lectura. Reorganiza /proc/cpuinfo, la jerarquía de cache
    de sysfs, y el EnvironmentProfile ya calculado por environment.py; no
    ejecuta nada nuevo sobre el hardware.
    """
    cpus = list(delegated_cpus)
    cpuinfo_text = _read_text(Path(proc_cpuinfo_path)) or ""
    cpu_model, sockets, cores_total, threads_per_core = _parse_cpuinfo(cpuinfo_text)

    resolved_sysfs = sysfs or SysfsPaths.from_base("/sys")
    l1_kb, l2_kb, llc_kb, llc_shared, llc_line_size_bytes = _cache_data(resolved_sysfs, cpus)

    available = list(getattr(env, "available_frequencies_khz", []) or [])
    freq_min_khz = min(available) if available else 0
    freq_max_khz = max(available) if available else 0

    return NodeProfile(
        node_id=node_id,
        hostname=hostname,
        cpu_model=cpu_model,
        sockets=sockets,
        cores_total=cores_total,
        threads_per_core=threads_per_core,
        numa_nodes=int(getattr(env, "numa_nodes", 0) or 0),
        cache_l1_kb=l1_kb,
        cache_l2_kb=l2_kb,
        cache_llc_kb=llc_kb,
        cache_llc_shared=llc_shared,
        cache_line_size_bytes=llc_line_size_bytes,
        freq_min_khz=freq_min_khz,
        freq_max_khz=freq_max_khz,
        scaling_driver=str(getattr(env, "scaling_driver", "")),
        perf_events_supported=tuple(getattr(env, "perf_events_available", ()) or ()),
        rapl_domains_available=tuple(getattr(env, "rapl_domains_available", ()) or ()),
        pmc_count=int(getattr(env, "pmc_count", 0) or 0),
    )


def write_node_profile(profile: NodeProfile, output_dir: str | Path) -> Path:
    """CAL-08/CAL-11: node_profile.json con todos los campos del dataclass."""
    path = Path(output_dir) / "node_profile.json"
    with path.open("w", encoding="utf-8") as profile_file:
        json.dump(asdict(profile), profile_file, indent=2, sort_keys=True)
        profile_file.write("\n")
    return path


def load_node_profile(output_dir: str | Path) -> NodeProfile:
    path = Path(output_dir) / "node_profile.json"
    with path.open(encoding="utf-8") as profile_file:
        data = json.load(profile_file)
    data["perf_events_supported"] = tuple(data.get("perf_events_supported", ()))
    data["rapl_domains_available"] = tuple(data.get("rapl_domains_available", ()))
    return NodeProfile(**data)
