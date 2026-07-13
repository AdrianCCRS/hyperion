from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping, Protocol

from .config import SysfsPaths, load_config
from .catalog import verify_binary


@dataclass(frozen=True)
class CheckResult:
    factor_id: str
    name: str
    passed: bool
    blocking: bool
    observed: dict[str, Any]
    message: str


class GpuInspector(Protocol):
    """Adaptador NVML sustituible por un mock en pruebas."""

    def active_processes(self) -> list[int]: ...
    def persistence_mode(self) -> bool | None: ...
    def mig_configuration(self) -> str | None: ...


def _result(
    factor_id: str, name: str, passed: bool, blocking: bool, observed: dict[str, Any], message: str
) -> CheckResult:
    return CheckResult(factor_id, name, passed, blocking, observed, message)


def _value(source: Any, name: str, default: Any = None) -> Any:
    return source.get(name, default) if isinstance(source, Mapping) else getattr(source, name, default)


def _cores(manifest: Any) -> list[int]:
    cores = _value(manifest, "cores", {})
    delegated = _value(cores, "delegated_cpus", ())
    return list(delegated)


def check_numa(delegated_cpus: Iterable[int], numa_cpu_map: Mapping[int, Iterable[int]]) -> CheckResult:
    delegated = set(delegated_cpus)
    nodes = {
        node for node, cpus in numa_cpu_map.items() if delegated.intersection(cpus)
    }
    complete = set().union(*(set(cpus) for cpus in numa_cpu_map.values())) if numa_cpu_map else set()
    passed = len(nodes) == 1 and delegated.issubset(complete)
    return _result("E04", "Afinidad NUMA", passed, True, {"nodes": sorted(nodes)}, "Los cores delegados deben pertenecer a un único nodo NUMA")


def check_smt(env: Any, manifest: Any) -> CheckResult:
    policy = _value(manifest, "smt_policy")
    passed = policy in {"all_threads", "one_thread_per_physical_core"}
    return _result("E05", "Política SMT", passed, True, {"policy": policy, "siblings": _value(env, "smt_siblings", {})}, "Declare smt_policy como all_threads o one_thread_per_physical_core")


def check_turbo_hwp(cpu_root: str | Path | None = None) -> CheckResult:
    root = Path(cpu_root) if cpu_root is not None else load_config().sysfs.cpu_root
    intel_pstate = root / "intel_pstate"
    observed = {
        "no_turbo": _read(intel_pstate / "no_turbo"),
        "status": _read(intel_pstate / "status"),
    }
    # En plataformas sin intel_pstate no existe una interfaz Turbo/HWP equivalente.
    return _result("E01", "Estado Turbo/HWP", True, True, observed, "Estado Turbo/HWP registrado para compararlo durante la campaña")


def check_turbo_hwp_unchanged(snapshot: Mapping[str, Any], cpu_root: str | Path | None = None) -> CheckResult:
    current = check_turbo_hwp(cpu_root).observed
    passed = dict(snapshot) == current
    return _result("E01", "Deriva Turbo/HWP", passed, True, {"expected": dict(snapshot), "current": current}, "El estado Turbo/HWP cambió durante la campaña")


def check_cgroup_clean(cgroup_path: str | Path, *, factor_id: str = "E03") -> CheckResult:
    contents = _read(Path(cgroup_path) / "cgroup.procs")
    pids = contents.split() if contents is not None else []
    return _result(factor_id, "cgroup sin procesos", not pids, True, {"pids": pids}, "El cgroup delegado debe estar vacío")


def check_governor(delegated_cpus: Iterable[int], expected: str, cpu_root: str | Path | None = None) -> CheckResult:
    root = Path(cpu_root) if cpu_root is not None else load_config().sysfs.cpu_root
    observed = {
        cpu: _read(root / f"cpu{cpu}" / "cpufreq/scaling_governor")
        for cpu in delegated_cpus
    }
    passed = bool(observed) and all(value == expected for value in observed.values())
    return _result("E07", "Governor efectivo", passed, True, {"governors": observed, "expected": expected}, "El governor efectivo no coincide con el esperado")


def check_frequency_write_permission(delegated_cpus: Iterable[int], cpu_root: str | Path | None = None) -> CheckResult:
    root = Path(cpu_root) if cpu_root is not None else load_config().sysfs.cpu_root
    paths = [
        root / f"cpu{cpu}" / "cpufreq" / filename
        for cpu in delegated_cpus
        for filename in ("scaling_governor", "scaling_setspeed")
    ]
    writable = {str(path): os.access(path, os.W_OK) for path in paths}
    return _result("E09", "Permisos de control de frecuencia", all(writable.values()), True, {"writable": writable}, "No hay permiso de escritura en todos los controles de frecuencia")


def check_external_load(threshold: float, load_reader: Callable[[], tuple[float, float, float]] = os.getloadavg) -> CheckResult:
    load = float(load_reader()[0])
    return _result("E08", "Carga externa", load <= threshold, True, {"load_1m": load, "threshold": threshold}, "La carga externa supera el umbral")


def check_temperature(temperature_c: float | None, minimum_c: float = 0.0, maximum_c: float = 90.0) -> CheckResult:
    if temperature_c is None:
        return _result("E02", "Temperatura de paquete", True, False, {"temperature_c": "unavailable"}, "No hay sensor térmico disponible")
    passed = minimum_c <= temperature_c <= maximum_c
    return _result("E02", "Temperatura de paquete", passed, True, {"temperature_c": temperature_c, "range_c": [minimum_c, maximum_c]}, "La temperatura está fuera del rango permitido")


def check_foreign_processes(foreign_pids: Iterable[int]) -> CheckResult:
    pids = list(foreign_pids)
    return _result("E06", "Procesos ajenos", not pids, True, {"foreign_pids": pids}, "Hay procesos ajenos con afinidad a los cores delegados")


def check_run_id_unique(output_dir: str | Path, run_id: str, overwrite: bool = False) -> CheckResult:
    path = Path(output_dir) / run_id
    passed = overwrite or not path.exists()
    return _result("I07", "run_id único", passed, True, {"path": str(path), "exists": path.exists()}, "El directorio de la corrida ya existe")


def check_rapl_wrap(env: Any, *, rapl_enabled: bool = True, rapl_root: str | Path | None = None) -> CheckResult:
    if not rapl_enabled or not _value(env, "rapl_capable", False):
        return _result("I05", "Rango de RAPL", True, False, {"rapl_wrap_correction": "not_applicable"}, "RAPL no está habilitado")
    root = Path(rapl_root) if rapl_root is not None else load_config().sysfs.rapl_root
    value = _read(root / "intel-rapl:0/max_energy_range_uj")
    status = "available" if value is not None else "unavailable"
    return _result("I05", "Rango de RAPL", True, False, {"rapl_wrap_correction": status, "max_energy_range_uj": value}, "La corrección de wrap se registró")


def check_rapl_domains(requested: Iterable[str], available: Iterable[str], rapl_enabled: bool) -> CheckResult:
    requested_set, available_set = set(requested), set(available)
    passed = not rapl_enabled or requested_set.issubset(available_set)
    return _result("I08", "Dominios RAPL", passed, True, {"requested": sorted(requested_set), "available": sorted(available_set)}, "Hay dominios RAPL solicitados no disponibles")


def check_disk_space(output_dir: str | Path, projected_bytes: int) -> CheckResult:
    free = shutil.disk_usage(Path(output_dir).parent).free
    passed = projected_bytes >= 0 and free >= projected_bytes
    return _result("I09", "Espacio libre", passed, True, {"free_bytes": free, "projected_bytes": projected_bytes}, "El espacio libre es menor que el tamaño proyectado")


def check_binary_exists(entry: Any) -> CheckResult:
    path = Path(entry.exec_path)
    passed = path.is_file() and os.access(path, os.X_OK)
    return _result("C01", "Binario ejecutable", passed, True, {"exec_path": str(path)}, "El binario no existe o no es ejecutable")


def check_binary_checksum(entry: Any) -> CheckResult:
    # C02 delega el hash real al catálogo; preflight solo lo presenta como CheckResult.
    passed = verify_binary(entry)
    return _result("C02", "Checksum del binario", passed, True, {"expected": entry.binary_checksum}, "El checksum del binario no coincide")


def check_success_check(entry: Any) -> CheckResult:
    check = getattr(entry, "success_check", None)
    if not isinstance(check, Mapping):
        return _result("C03", "success_check", False, True, {}, "success_check debe ser un objeto")
    if check.get("type") == "exit_code" and isinstance(check.get("expected", 0), int) and not isinstance(check.get("expected", 0), bool):
        return _result("C03", "success_check", True, True, {"type": "exit_code"}, "success_check válido")
    if check.get("type") == "stdout_regex" and isinstance(check.get("pattern"), str):
        try:
            re.compile(check["pattern"])
        except re.error as error:
            return _result("C03", "success_check", False, True, {}, f"Regex inválido: {error}")
        return _result("C03", "success_check", True, True, {"type": "stdout_regex"}, "success_check válido")
    return _result("C03", "success_check", False, True, {}, "Tipo de success_check no soportado")


def check_memory_size(entry: Any, ram_bytes: int | None = None) -> CheckResult:
    estimated = getattr(entry, "estimated_memory_bytes", None)
    if estimated is None:
        return _result("C05", "Memoria del size_variant", False, True, {"estimated_bytes": "not_declared"}, "El catálogo debe declarar estimated_memory_bytes")
    if ram_bytes is None:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        ram_bytes = pages * page_size
    passed = isinstance(estimated, int) and estimated >= 0 and estimated <= ram_bytes
    return _result("C05", "Memoria del size_variant", passed, True, {"estimated_bytes": estimated, "ram_bytes": ram_bytes}, "El size_variant requiere más RAM que la disponible")


def check_toolchain(rebuild: bool) -> CheckResult:
    if not rebuild:
        return _result("D01", "Toolchain", True, False, {"rebuild": False}, "No se recompilarán binarios")
    tools = {tool: shutil.which(tool) for tool in ("gcc", "gfortran", "make")}
    return _result("D01", "Toolchain", all(tools.values()), True, tools, "Falta una herramienta de compilación")


def check_perf_counter_capacity(requested_events: Iterable[str], pmc_count: int | None) -> CheckResult:
    requested = list(requested_events)
    if pmc_count is None:
        return _result("D05", "Capacidad PMC", True, True, {"requested": len(requested), "pmc_count": "not_declared"}, "La capacidad PMC se verificará con node_profile")
    return _result("D05", "Capacidad PMC", len(requested) <= pmc_count, True, {"requested": len(requested), "pmc_count": pmc_count}, "Se solicitaron más eventos que PMCs disponibles")


def check_core_hour_budget(remaining: float | None, projected: float | None) -> CheckResult:
    if remaining is None or projected is None:
        return _result("OPS-01", "Presupuesto hora-núcleo", True, True, {"status": "not_declared"}, "El presupuesto se verificará cuando esté declarado")
    return _result("OPS-01", "Presupuesto hora-núcleo", remaining >= projected, True, {"remaining": remaining, "projected": projected}, "El presupuesto restante es insuficiente")


def check_gpu(inspector: GpuInspector | None) -> list[CheckResult]:
    if inspector is None:
        return [
            _result("G01", "GPU sin actividad ajena", False, True, {}, "Se requiere un inspector NVML"),
            _result("G02", "Persistence mode", False, True, {}, "Se requiere un inspector NVML"),
            _result("G03", "Configuración MIG", False, True, {}, "Se requiere un inspector NVML"),
        ]
    pids = inspector.active_processes()
    persistence, mig = inspector.persistence_mode(), inspector.mig_configuration()
    return [
        _result("G01", "GPU sin actividad ajena", not pids, True, {"pids": pids}, "Hay procesos CUDA ajenos"),
        _result("G02", "Persistence mode", persistence is not None, True, {"persistence_mode": persistence}, "No se pudo leer persistence mode"),
        _result("G03", "Configuración MIG", mig is not None, True, {"mig": mig}, "No se pudo leer la configuración MIG"),
    ]


def check_calibration_output(stream_stdout: str, ert_stdout: str) -> CheckResult:
    bw = re.search(r"(?:Best Rate MB/s|Bandwidth)\s*[:=]\s*([0-9.]+)", stream_stdout, re.I)
    flops = re.search(r"(?:Peak GFLOPS|GFLOPS)\s*[:=]\s*([0-9.]+)", ert_stdout, re.I)
    passed = bw is not None and flops is not None
    return _result("D02", "Salida de calibración", passed, True, {"stream_parseable": bw is not None, "ert_parseable": flops is not None}, "STREAM y ERT deben reportar BW y FLOP/s parseables")


def check_calibration_plausibility(bw: float, p: float, spec_bw: float, spec_p: float, tolerance: float = 0.4) -> CheckResult:
    def plausible(value: float, reference: float) -> bool:
        return reference > 0 and reference * (1 - tolerance) <= value <= reference * (1 + tolerance)
    passed = plausible(bw, spec_bw) and plausible(p, spec_p)
    return _result("D03", "Plausibilidad Roofline", passed, True, {"bw": bw, "p": p, "spec_bw": spec_bw, "spec_p": spec_p}, "Los picos medidos están fuera de la tolerancia declarada")


def check_calibration_stability(cv_pct: float, threshold: float = 5.0) -> CheckResult:
    passed = cv_pct <= threshold
    return _result("D04", "Estabilidad de calibración", passed, False, {"cv_pct": cv_pct, "threshold": threshold}, "El CV de referencias P95 supera el umbral")


def run_post_calibration_preflight(calibration: Any, references: Any, hardware_spec: Mapping[str, float]) -> list[CheckResult]:
    """Valida D02–D04 tras STREAM/ERT y antes de generar la matriz dataset."""
    stream_stdout, ert_stdout = _value(calibration, "stream_stdout", ""), _value(calibration, "ert_stdout", "")
    bw, p = _value(calibration, "bw_pico_bytes_per_s"), _value(calibration, "p_pico_flops_per_s")
    cv_pct = _value(references, "cv_pct")
    results = [check_calibration_output(stream_stdout, ert_stdout)]
    if isinstance(bw, (int, float)) and isinstance(p, (int, float)):
        results.append(check_calibration_plausibility(bw, p, hardware_spec["bw_pico_bytes_per_s"], hardware_spec["p_pico_flops_per_s"]))
    else:
        results.append(_result("D03", "Plausibilidad Roofline", False, True, {}, "La calibración no declaró BW_pico y P_pico"))
    if isinstance(cv_pct, (int, float)):
        results.append(check_calibration_stability(float(cv_pct)))
    else:
        results.append(_result("D04", "Estabilidad de calibración", False, False, {}, "No hay referencias P95 para calcular CV"))
    return results


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def run_campaign_preflight(
    manifest: Any, env: Any, catalog: Mapping[str, Any], *, sysfs: SysfsPaths | None = None, node_profile: Any = None, gpu_inspector: GpuInspector | None = None
) -> list[CheckResult]:
    """Ejecuta todos los checks de campaña que disponen de datos antes de la matriz."""
    cores = _cores(manifest)
    sysfs = sysfs or load_config().sysfs
    rapl = _value(manifest, "rapl", {})
    gpu = _value(manifest, "gpu", {})
    results = [
        check_turbo_hwp(sysfs.cpu_root),
        check_numa(cores, getattr(env, "numa_cpu_map", {})),
        check_smt(env, manifest),
        check_rapl_wrap(env, rapl_enabled=bool(_value(rapl, "enabled", False)), rapl_root=sysfs.rapl_root),
    ]
    cgroup_path = _value(manifest, "cgroup_path")
    if cgroup_path:
        results.append(check_cgroup_clean(cgroup_path))
    if _value(env, "freq_control_capable", False):
        results.append(check_frequency_write_permission(cores, sysfs.cpu_root))
        results.append(check_governor(cores, "userspace", sysfs.cpu_root))
    results.append(check_rapl_domains(_value(rapl, "domains", []), _value(env, "rapl_domains_available", []), bool(_value(rapl, "enabled", False))))
    output_dir, overwrite = _value(manifest, "output_dir"), bool(_value(manifest, "overwrite", False))
    results.append(_result("I07", "Directorio de campaña", overwrite or not Path(output_dir).exists(), True, {"output_dir": str(output_dir)}, "output_dir ya existe"))
    results.append(check_disk_space(output_dir, int(_value(manifest, "projected_campaign_bytes", 0))))
    if _value(gpu, "enabled", False):
        results.extend(check_gpu(gpu_inspector))
    refs = tuple(_value(manifest, "calibration", ())) + tuple(_value(manifest, "kernels", ()))
    for reference in refs:
        entry = catalog[reference]
        results.extend([check_binary_exists(entry), check_binary_checksum(entry), check_success_check(entry), check_memory_size(entry)])
    results.append(check_toolchain(bool(_value(manifest, "rebuild", False))))
    results.append(check_perf_counter_capacity(_value(manifest, "perf_events", ()), _value(node_profile, "pmc_count", _value(env, "pmc_count"))))
    results.append(check_core_hour_budget(_value(manifest, "remaining_core_hours"), _value(manifest, "projected_core_hours")))
    return results


def run_reduced_preflight(manifest: Any, env: Any, entry: Any, run_id: str, *, expected_governor: str = "userspace", load_threshold: float = 1.0, turbo_snapshot: Mapping[str, Any] | None = None, cpu_root: str | Path | None = None, load_reader: Callable[[], tuple[float, float, float]] = os.getloadavg) -> list[CheckResult]:
    """Ejecuta checks por corrida, incluidos C01/C02 para detectar binarios cambiados."""
    results = [
        check_temperature(_value(manifest, "package_temperature_c"), _value(manifest, "temperature_min_c", 0.0), _value(manifest, "temperature_max_c", 90.0)),
        check_foreign_processes(_value(manifest, "foreign_affinity_pids", ())),
        check_governor(_cores(manifest), expected_governor, cpu_root),
        check_external_load(load_threshold, load_reader),
        check_run_id_unique(_value(manifest, "output_dir"), run_id, bool(_value(manifest, "overwrite", False))),
        check_binary_exists(entry),
        check_binary_checksum(entry),
        check_success_check(entry),
    ]
    if turbo_snapshot is not None:
        results.append(check_turbo_hwp_unchanged(turbo_snapshot, cpu_root))
    return results
