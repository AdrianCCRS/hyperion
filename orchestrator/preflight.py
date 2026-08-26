from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping, Protocol

from .config import SysfsPaths, load_config
from .catalog import verify_binary

# ARC-101: el conjunto real y fijo de eventos de PMU que
# telemetry_kernel_launcher pide por muestra de CPU
# (telemetry/include/telemetry/perf_reader.hpp, kEventCount=10) -- no es
# configurable desde el manifiesto, el harness siempre pide los 10 (o
# degrada evento por evento, nunca menos de golpe). D05 debe compararse
# contra este número real, no contra un campo del manifiesto que nunca
# existió: antes de este fix, check_perf_counter_capacity() se llamaba con
# manifest.perf_events, un atributo que Manifest jamás declaró, así que
# _value() siempre caía en su default () y D05 pasaba trivialmente sin
# comparar nada contra el presupuesto real de contadores del nodo. Si
# perf_reader.hpp cambia su conjunto de eventos, esta tupla debe
# actualizarse a mano junto con él.
#
# ARC-132: "l2_lines_in_all" retirado de este conteo -- perf_reader.cpp ya
# no intenta abrirlo en absoluto (índice kL2LinesInAll saltado antes de
# perf_event_open, mismo camino de degradación elegante que ya existía para
# hardware sin soporte, ARC-63) desde que se confirmó que nmi_watchdog=1
# reserva un PMC físico por núcleo que el presupuesto pmc_count=10 (ARC-53)
# nunca descontó -- pedir el décimo contador forzaba multiplexación real
# (deltas negativos en FP_ARITH_INST_RETIRED). El presupuesto real que el
# harness consume hoy es 9, no 10; D05 seguía comparando contra 10 y podía
# bloquear una campaña innecesariamente en un nodo con pmc_count=9.
_HARNESS_PERF_EVENTS = (
    "instructions", "cycles", "cache_references", "cache_misses",
    "stalled_cycles_backend",
    "fp_scalar_double", "fp_128b_packed_double", "fp_256b_packed_double", "fp_512b_packed_double",
)


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


def _requires_frequency_control(manifest: Any) -> bool:
    """Solo los niveles fixed requieren permisos de escritura y userspace."""
    for level in _value(manifest, "frequency_levels", ()):
        mode = _value(level, "mode")
        if mode == "fixed":
            return True
    return False


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


def check_turbo_hwp(
    cpu_root: str | Path | None = None, *, require_disabled: bool = False
) -> CheckResult:
    root = Path(cpu_root) if cpu_root is not None else load_config().sysfs.cpu_root
    intel_pstate = root / "intel_pstate"
    amd_pstate = root / "amd_pstate"
    cpufreq = root / "cpufreq"
    observed = {
        "scaling_driver": _read(root / "cpu0" / "cpufreq" / "scaling_driver"),
        "no_turbo": _read(intel_pstate / "no_turbo"),
        "status": _read(intel_pstate / "status"),
        "amd_pstate_status": _read(amd_pstate / "status"),
        "cpufreq_boost": _read(cpufreq / "boost"),
    }
    passed = not require_disabled or observed["no_turbo"] == "1"
    message = (
        "Estado Turbo/HWP/CPB registrado para compararlo durante la campaña"
        if passed else
        "La campaña exige turbo deshabilitado (intel_pstate/no_turbo=1)"
    )
    return _result("E01", "Estado Turbo/HWP/CPB", passed, True, observed, message)


def check_turbo_hwp_unchanged(snapshot: Mapping[str, Any], cpu_root: str | Path | None = None) -> CheckResult:
    current = check_turbo_hwp(cpu_root).observed
    passed = dict(snapshot) == current
    return _result("E01", "Deriva Turbo/HWP", passed, True, {"expected": dict(snapshot), "current": current}, "El estado Turbo/HWP cambió durante la campaña")


def check_cgroup_clean(cgroup_path: str | Path, *, factor_id: str = "E03") -> CheckResult:
    contents = _read(Path(cgroup_path) / "cgroup.procs")
    pids = contents.split() if contents is not None else []
    return _result(factor_id, "cgroup sin procesos", not pids, True, {"pids": pids}, "El cgroup delegado debe estar vacío")


def check_governor(delegated_cpus: Iterable[int], expected: str | None, cpu_root: str | Path | None = None, control_paths: Mapping[int, Mapping[str, str]] | None = None) -> CheckResult:
    """E07: verifica el governor efectivo.

    ARC-94: ``expected`` puede ser ``None`` para la estrategia
    ``bounded_range`` (intel_pstate/amd-pstate) -- a diferencia de
    ``discrete_bounds`` (acpi-cpufreq), donde ``scaling_setspeed`` solo
    tiene efecto bajo el governor ``userspace`` (requisito real del
    kernel), fijar el rango vía ``scaling_min_freq``/``scaling_max_freq``
    no depende de ningún governor específico -- intel_pstate solo ofrece
    ``performance``/``powersave`` (ninguno es ``userspace``, confirmado en
    paccaA100) y ambos respetan el rango como límite. Exigir "userspace"
    incondicionalmente bloqueaba el preflight en cualquier nodo con este
    driver, sin importar qué permiso concediera el administrador.
    """
    root = Path(cpu_root) if cpu_root is not None else load_config().sysfs.cpu_root
    observed = {
        cpu: _read(Path(control_paths[cpu]["scaling_governor"]))
        if control_paths and cpu in control_paths and "scaling_governor" in control_paths[cpu]
        else _read(root / f"cpu{cpu}" / "cpufreq/scaling_governor")
        for cpu in delegated_cpus
    }
    if expected is None:
        passed = bool(observed) and all(value is not None for value in observed.values())
        return _result(
            "E07", "Governor efectivo", passed, True,
            {"governors": observed, "expected": "cualquiera (bounded_range: min/max aplica sin importar el governor)"},
            "No se pudo leer el governor efectivo en ninguno de los cores delegados",
        )
    passed = bool(observed) and all(value == expected for value in observed.values())
    return _result("E07", "Governor efectivo", passed, True, {"governors": observed, "expected": expected}, "El governor efectivo no coincide con el esperado")


def check_frequency_write_permission(
    delegated_cpus: Iterable[int],
    cpu_root: str | Path | None = None,
    control_paths: Mapping[int, Mapping[str, str]] | None = None,
    required_attrs: Iterable[str] | None = None,
) -> CheckResult:
    """E09.

    ARC-95: ``required_attrs`` es, igual que ``check_governor(expected=None)``,
    la corrección para ``bounded_range`` (intel_pstate/amd-pstate) -- exigir
    ``scaling_governor`` escribible aquí hacía que E09 bloqueara con
    exactamente el permiso P1 solicitado (solo scaling_min_freq/max_freq),
    porque este chequeo nunca se enteró de la misma distinción de
    estrategia que ya se aplicó en environment.py/check_governor. Por
    defecto sigue exigiendo los tres atributos (compatibilidad con
    llamadores que no declaran estrategia).
    """
    attrs = tuple(required_attrs) if required_attrs is not None else ("scaling_governor", "scaling_min_freq", "scaling_max_freq")
    root = Path(cpu_root) if cpu_root is not None else load_config().sysfs.cpu_root
    paths = (
        [Path(path) for cpu in delegated_cpus for name, path in (control_paths or {}).get(cpu, {}).items() if name in attrs]
        if control_paths is not None
        else [
            root / f"cpu{cpu}" / "cpufreq" / filename
            for cpu in delegated_cpus
            for filename in attrs
        ]
    )
    writable = {str(path): os.access(path, os.W_OK) for path in paths}
    return _result("E09", "Permisos de control de frecuencia", bool(writable) and all(writable.values()), True, {"writable": writable}, "No hay permiso de escritura en todos los controles de frecuencia")


def check_frequency_domain(delegated_cpus: Iterable[int], frequency_domain_cpus: Mapping[int, Iterable[int]] | None) -> CheckResult:
    """E10: el dominio real de control de frecuencia no debe exceder los cores delegados.

    En hardware donde el control es por socket (ej. acpi-cpufreq en Nehalem-EX),
    fijar la frecuencia de un core delegado también cambia la de cores ajenos que
    compartan el dominio. Si el kernel no expone datos de dominio (driver moderno
    sin ese archivo), no se bloquea por falta de evidencia.
    """
    delegated = set(delegated_cpus)
    domains = frequency_domain_cpus or {}
    leaking = {
        cpu: sorted(set(cpus) - delegated)
        for cpu, cpus in domains.items()
        if cpu in delegated and not set(cpus).issubset(delegated)
    }
    passed = not leaking
    return _result(
        "E10",
        "Dominio de frecuencia compartido",
        passed,
        True,
        {"domains": {cpu: sorted(cpus) for cpu, cpus in domains.items()}, "leaking_cpus": leaking},
        "El dominio de control de frecuencia incluye CPUs fuera de los cores delegados (riesgo de afectar a otro job)",
    )


def check_exclusive_node_allocation(
    uncore_enabled: bool, cpus_allowed: Iterable[int], total_cpu_count: int
) -> CheckResult:
    """E11: uncore_imc (CAS_COUNT_READ/WRITE) son contadores de ámbito
    sistema/socket (pid=-1, ver telemetry/include/telemetry/uncore_reader.hpp)
    -- miden TODO el tráfico de memoria del nodo, no solo el de los cores
    delegados. Si el job no reserva el nodo completo (`--exclusive`), otro
    job ajeno corriendo en el resto del nodo contamina esa lectura sin
    ninguna forma de separarla, exactamente el mismo riesgo de fuga entre
    usuarios que E10 ya vigila para el dominio de frecuencia. Requisito
    explícito del usuario: bloquea SIEMPRE que uncore esté habilitado, no es
    opcional.

    Mismo mecanismo ya validado empíricamente en pacca
    (memoria de proyecto, `pacca-cluster-unicartagena-facts`): con
    `--exclusive --ntasks=1`, `/proc/self/status` reporta
    `Cpus_allowed_list` cubriendo TODAS las CPUs lógicas del nodo (el
    aislamiento real lo da `--exclusive` a nivel Slurm, no un cpuset fino) --
    así que "el job tiene acceso a todas las CPUs lógicas" es la señal
    disponible de que el nodo es exclusivo, no una prueba directa contra
    Slurm.
    """
    if not uncore_enabled:
        return _result(
            "E11", "Reserva exclusiva de nodo (uncore)", True, False,
            {"uncore_enabled": False},
            "uncore no está habilitado en este manifiesto, no aplica",
        )
    allowed = set(cpus_allowed)
    expected = set(range(total_cpu_count))
    passed = total_cpu_count > 0 and allowed == expected
    return _result(
        "E11", "Reserva exclusiva de nodo (uncore)", passed, True,
        {"cpus_allowed": sorted(allowed), "total_cpu_count": total_cpu_count},
        "uncore_imc mide tráfico de memoria de TODO el nodo -- se requiere reserva exclusiva "
        "(#SBATCH --exclusive) cuando está habilitado, y el job no parece tener el nodo completo",
    )


def check_uncore_required_for_cpu_dataset(entries: Iterable[Any], uncore_enabled: bool) -> CheckResult:
    """E12 (ARC-123): sin `manifest.uncore.enabled=True`, ninguna ventana de
    CPU puede llegar a `quality_status="ok"` -- `_finalize_operational_intensity()`
    (`postprocess.py`) deja `operational_intensity`/`phase_label_train`
    indefinidos en toda ventana que ningún intervalo real de `uncore_imc`
    cubrió, y sin uncore habilitado eso es SIEMPRE. `validate_windows()`
    (VAL-09/I10) exige al menos `target_windows_per_repetition` ventanas
    `"ok"` para aceptar una corrida -- sin este chequeo, una campaña de CPU
    completa correría durante horas antes de descubrir, recién al terminar
    la primera corrida, que el 100% de las corridas de CPU van a rechazarse.
    Bloquea temprano en cambio, mismo principio que E08/E09/I09.

    No aplica a una campaña puramente de GPU: las filas GPU nunca dependen
    de esta señal (`usable_status="gpu_telemetry"` en `validate_windows()`),
    así que un catálogo sin ningún kernel de CPU nunca dispara este chequeo.

    ARC-191: `entries` debe ser SOLO `manifest.kernels` (role=="dataset"),
    nunca incluir `manifest.calibration`. Las corridas de calibración
    (STREAM/ERT) nunca pasan por `postprocess_run()`/`validate_windows()`
    -- `calibration.py` las ejecuta con `run_single()` directo y lee su
    ancho de banda/FLOPs del propio stdout, nunca de `uncore_imc` -- así
    que no están sujetas al riesgo que este chequeo previene. Como MAN-07
    exige declarar `stream_official`/`ert_probe` (`device=="cpu"`) en
    `calibration:` en TODA campaña, incluirlas aquí habría bloqueado
    incluso una campaña 100% GPU.
    """
    has_cpu_kernel = any(_value(entry, "device", "cpu") != "gpu" for entry in entries)
    passed = not has_cpu_kernel or uncore_enabled
    return _result(
        "E12", "uncore requerido para clasificar CPU", passed, True,
        {"has_cpu_kernel": has_cpu_kernel, "uncore_enabled": uncore_enabled},
        "La campaña incluye kernels de CPU pero manifest.uncore.enabled no está activo -- "
        "ninguna ventana de CPU podría llegar a quality_status=\"ok\" (ARC-123), "
        "toda corrida de CPU sería rechazada por VAL-09/I10",
    )


def check_uncore_readable(
    uncore_enabled: bool,
    *,
    probe: Callable[[], tuple[bool, str]] | None = None,
    paranoid_path: str | Path = "/proc/sys/kernel/perf_event_paranoid",
) -> CheckResult:
    """E13 (ARC-184): comprueba que los contadores de uncore se pueden LEER
    de verdad, no solo que el manifiesto los pida.

    E12 verifica `manifest.uncore.enabled`, que es una declaración de
    intención. Este chequeo verifica la capacidad real, que es otra cosa y
    puede desaparecer sin que nada en el repositorio cambie: `uncore_imc`
    son contadores de ámbito sistema y `perf` solo los abre con
    `perf_event_paranoid <= 0` o con CAP_PERFMON en el binario. Con
    `paranoid = 2` --el valor por defecto de la mayoría de distribuciones--
    perf degrada los eventos a espacio de usuario, donde el IMC no existe,
    y devuelve "<not supported>" por cada término.

    Lo peor de ese modo de fallo es que NO es ruidoso: `perf stat` sigue
    corriendo y emitiendo intervalos a su cadencia normal, solo que con los
    contadores vacíos (ARC-120 ya blindó al lector para distinguirlo). La
    campaña corre entera, produce miles de ventanas, y solo al escribir
    cada `verdict.json` se descubre que el 100 % quedó
    `intensity_undefined` y toda corrida se rechaza por VAL-09/I10.

    Ocurrió de verdad el 2026-08-22: el pre-vuelo de fases gastó 27
    corridas y ~20 min para terminar en 0 aceptadas / 27 rechazadas, y la
    campaña de tamaño se canceló a mano tras dos minutos al detectarse lo
    mismo. La campaña del 2026-08-20 sí había leído uncore en el MISMO nodo
    y sin reinicio de por medio, así que el permiso puede irse y volver:
    hay que comprobarlo en cada campaña, no una vez.
    """
    if not uncore_enabled:
        return _result(
            "E13", "Contadores de uncore legibles", True, False,
            {"uncore_enabled": False},
            "uncore no está habilitado en este manifiesto, no aplica",
        )

    paranoid: int | None = None
    try:
        paranoid = int(Path(paranoid_path).read_text().strip())
    except (OSError, ValueError):
        paranoid = None

    if probe is None:
        probe = _probe_uncore_counters
    readable, detail = probe()

    return _result(
        "E13", "Contadores de uncore legibles", readable, True,
        {"perf_event_paranoid": paranoid, "probe": detail},
        "No se pueden LEER los contadores uncore_imc pese a que el manifiesto los exige. "
        "Sin ellos toda ventana de CPU queda intensity_undefined y la campaña completa "
        "se rechazaría por VAL-09/I10 (ARC-184). "
        "Requiere que el administrador ponga kernel.perf_event_paranoid <= 0 o conceda "
        "CAP_PERFMON a /usr/bin/perf; no es corregible desde la cuenta de usuario",
    )


def _probe_uncore_counters(timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Lee un intervalo real de `uncore_imc_0/cas_count_read/`.

    Se mide contra el PMU de verdad y no contra `perf_event_paranoid` a
    secas porque CAP_PERFMON en el binario de perf también habilita la
    lectura con paranoid alto: mirar solo el sysctl daría un falso negativo
    en un nodo correctamente configurado por capacidades.

    Bug encontrado el 2026-08-25: esta sonda invocaba `perf stat` SIN
    `-I` (modo intervalo) y con separador ',' -- un modo de invocación
    distinto al que usa el lector de producción (`uncore_reader.cpp`,
    que siempre pasa `-a -I <ms> -x ';'`). Sin `-I`, perf emite el
    formato de RESUMEN final, que antepone un valor YA ESCALADO con
    unidad ("0.23,MiB,uncore_imc_0/cas_count_read/,102712278,100.00,,":
    el campo 0 es la magnitud legible, no un conteo entero) -- el chequeo
    original solo miraba el campo 0 y esperaba un entero puro, así que
    fallaba SIEMPRE con este formato pese a que el conteo crudo (campo 3)
    era real. Eso produjo falsos negativos de E13 indistinguibles de una
    pérdida real de permiso (jobs 6431/6484), cuando en ambos casos
    CAP_PERFMON seguía presente (`getcap /usr/bin/perf` lo confirmó).
    Corregido invocando perf EXACTAMENTE como producción (`-I`/`;`) y
    parseando el mismo campo (índice 1, "interval-time;value;unit;...")
    que `parse_perf_stat_csv_line` en `uncore_reader.cpp`.
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["perf", "stat", "-a", "-I", "100", "-x", ";",
             "-e", "uncore_imc_0/cas_count_read/", "sleep", "0.3"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"no se pudo ejecutar perf: {error}"

    # perf escribe el informe en stderr; formato por línea de intervalo:
    # "interval-time;value;unit;event;time-running;percent-running[,...]".
    output = (completed.stderr or "") + (completed.stdout or "")
    for line in output.strip().splitlines():
        fields = line.split(";")
        if len(fields) < 2:
            continue
        value_field = fields[1].strip()
        if not value_field or value_field.startswith("<"):
            continue
        if value_field.isdigit():
            return True, f"cas_count_read={value_field}"
    return False, output.strip().splitlines()[-1][:200] if output.strip() else "sin salida"


def check_external_load(threshold: float, load_reader: Callable[[], tuple[float, float, float]] = os.getloadavg, cpu_count: int = 1) -> CheckResult:
    """E08.

    ARC-109: la versión original comparaba ``load_1m / cpu_count`` contra
    ``threshold`` directamente -- pero ``load_1m`` (promedio de carga del
    sistema) incluye el trabajo LEGÍTIMO y ESPERADO de la propia campaña
    sobre los CPUs delegados, no solo contaminación externa. En una
    campaña real, densa y de larga duración (corridas casi consecutivas
    en `cpu_count` núcleos dedicados durante horas), la media exponencial
    de 1 minuto de Linux converge y se mantiene estable cerca de
    `cpu_count` mientras el trabajo propio sigue -- confirmado en pacca:
    ``load_1m`` se mantuvo en 6.04 (cpu_count=6) de forma prácticamente
    constante durante toda una campaña real, rechazando 105 de 126
    combinaciones por "carga externa" que en realidad era el propio
    harness. El chequeo original no podía distinguir "mis 6 núcleos
    delegados están ocupados, como se espera" de "hay 6 núcleos de más
    ocupados por un proceso ajeno".

    Fix: en vez de comparar el promedio total, se compara el EXCESO sobre
    `cpu_count` -- la parte de `load_1m` que NO se explica por el trabajo
    esperado de la propia campaña sobre sus núcleos delegados. `threshold`
    ahora expresa cuánto exceso, como fracción de `cpu_count`, se tolera
    antes de sospechar contaminación real (p. ej. threshold=1.0 tolera
    hasta el doble de `cpu_count` de carga total antes de bloquear).
    """
    load = float(load_reader()[0])
    excess = max(load - cpu_count, 0.0)
    normalized_excess = excess / max(cpu_count, 1)
    return _result(
        "E08", "Carga externa", normalized_excess <= threshold, True,
        {"load_1m": load, "excess_load_1m": excess, "normalized_excess_load_1m": normalized_excess, "cpu_count": max(cpu_count, 1), "threshold": threshold},
        "La carga externa (más allá de lo que explica la propia campaña) supera el umbral",
    )


def check_temperature(temperature_c: float | None, minimum_c: float = 0.0, maximum_c: float = 90.0) -> CheckResult:
    if temperature_c is None:
        return _result("E02", "Temperatura de paquete", True, False, {"temperature_c": "unavailable"}, "No hay sensor térmico disponible")
    passed = minimum_c <= temperature_c <= maximum_c
    return _result("E02", "Temperatura de paquete", passed, True, {"temperature_c": temperature_c, "range_c": [minimum_c, maximum_c]}, "La temperatura está fuera del rango permitido")


def read_package_temperature_c(hwmon_root: str | Path = "/sys/class/hwmon") -> float | None:
    """Lee la mayor temperatura de paquete expuesta por coretemp.

    Los índices hwmon no son estables entre arranques, por lo que se
    descubre por `name=coretemp` y por etiquetas `Package id N`. Los valores
    sysfs están en miligrados Celsius. Devuelve None si no existe una
    lectura válida; nunca sustituye un sensor ausente por cero.
    """
    temperatures: list[float] = []
    for hwmon_dir in sorted(Path(hwmon_root).glob("hwmon*")):
        if _read(hwmon_dir / "name") != "coretemp":
            continue
        for label_path in sorted(hwmon_dir.glob("temp*_label")):
            label = _read(label_path)
            if label is None or not label.startswith("Package id "):
                continue
            input_path = label_path.with_name(label_path.name.replace("_label", "_input"))
            raw = _read(input_path)
            try:
                temperatures.append(float(raw) / 1000.0)
            except (TypeError, ValueError):
                continue
    return max(temperatures) if temperatures else None


def check_foreign_processes(foreign_pids: Iterable[int]) -> CheckResult:
    pids = list(foreign_pids)
    return _result("E06", "Procesos ajenos", not pids, True, {"foreign_pids": pids}, "Hay procesos ajenos con afinidad a los cores delegados")


def _parse_stat_state_and_processor(stat_text: str) -> tuple[str, int] | None:
    """/proc/<pid>/stat: campo 2 (comm) puede tener espacios/parentesis, asi
    que se ubica por el ULTIMO ')' antes de partir el resto por espacios.
    Desde ahi: campo[0]=state (field 3), campo[36]=processor (field 39) --
    ver `man proc`."""
    close = stat_text.rfind(")")
    if close < 0:
        return None
    rest = stat_text[close + 1:].split()
    if len(rest) <= 36:
        return None
    return rest[0], int(rest[36])


def detect_foreign_affinity_pids(
    delegated_cpus: Iterable[int],
    *,
    proc_root: str | Path = "/proc",
    own_pids: Iterable[int] = (),
) -> list[int]:
    """E06: escanea /proc/*/stat buscando procesos que están CORRIENDO
    ACTIVAMENTE en este instante (state='R') sobre uno de delegated_cpus
    (campo "processor" de /proc/<pid>/stat) -- nunca por membresía de
    cgroup, que no detecta contención real de caché/ancho de banda de
    memoria entre procesos que comparten los mismos cores físicos (un
    efecto físico independiente de que perf_event_open con PID+inherit
    atribuya bien las muestras al proceso correcto -- eso resuelve
    atribución, no contención).

    Deliberadamente NO usa Cpus_allowed (aunque un proceso lo tenga
    solapado, no está causando contención si no está corriendo ahí AHORA):
    en la práctica casi todo proceso del sistema en reposo tiene
    Cpus_allowed sin restringir, así que filtrar solo por esa máscara
    marca como "ajeno" a decenas de daemons inactivos en cualquier
    máquina real -- confirmado en el primer piloto real contra felix
    (F4.4), donde eso hizo que las 6 combinaciones se rechazaran sin
    ninguna contención real. `processor` + state='R' es una foto del
    scheduler en este instante: solo atrapa lo que de verdad está
    ejecutando ciclos ahí ahora mismo.

    Excluye hilos de kernel (sin /proc/<pid>/cmdline, no son carga de
    trabajo real) y `own_pids` (el propio proceso del orquestador).
    """
    delegated = set(delegated_cpus)
    excluded = set(own_pids)
    root = Path(proc_root)
    foreign: list[int] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return foreign
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            if not (entry / "cmdline").read_bytes():
                continue  # hilo de kernel, sin argv
        except OSError:
            continue  # el proceso murió entre el listado y la lectura
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed = _parse_stat_state_and_processor(stat_text)
        if parsed is None:
            continue
        state, processor = parsed
        if state == "R" and processor in delegated:
            foreign.append(pid)
    return foreign


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


def check_binary_checksum(entry: Any, node_id: str | None = None) -> CheckResult:
    # C02 delega el hash real al catálogo; preflight solo lo presenta como CheckResult.
    passed = verify_binary(entry, node_id)
    return _result("C02", "Checksum del binario", passed, True, {"expected": entry.binary_checksum, "node_id": node_id}, "El checksum del binario no coincide")


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
        return _result("D05", "Capacidad PMC", False, True, {"requested": len(requested), "pmc_count": "not_declared"}, "node_profile debe declarar la capacidad PMC")
    return _result("D05", "Capacidad PMC", len(requested) <= pmc_count, True, {"requested": len(requested), "pmc_count": pmc_count}, "Se solicitaron más eventos que PMCs disponibles")


def check_core_hour_budget(remaining: float | None, projected: float | None) -> CheckResult:
    if remaining is None or projected is None:
        return _result("OPS-01", "Presupuesto hora-núcleo", False, True, {"status": "not_declared"}, "El presupuesto y la proyección deben estar declarados")
    return _result("OPS-01", "Presupuesto hora-núcleo", remaining >= projected, True, {"remaining": remaining, "projected": projected}, "El presupuesto restante es insuficiente")


def check_gpu_foreign_activity(inspector: GpuInspector | None) -> CheckResult:
    """G01, factorizado de check_gpu() (ARC-129) para poder correrlo TAMBIÉN
    por combinación, no solo una vez al inicio de la campaña -- a diferencia
    de G02/G03 (persistence mode, configuración MIG: administrativos,
    estáticos, poco probable que cambien a mitad de una campaña de horas en
    un nodo exclusivo), otro job del clúster compartido puede empezar a usar
    la GPU en cualquier momento, exactamente el mismo riesgo que
    check_foreign_processes() (E06) ya vigila para CPU."""
    if inspector is None:
        return _result("G01", "GPU sin actividad ajena", False, True, {}, "Se requiere un inspector NVML")
    pids = inspector.active_processes()
    return _result("G01", "GPU sin actividad ajena", not pids, True, {"pids": pids}, "Hay procesos CUDA ajenos")


def check_gpu(inspector: GpuInspector | None) -> list[CheckResult]:
    if inspector is None:
        return [
            check_gpu_foreign_activity(inspector),
            _result("G02", "Persistence mode", False, True, {}, "Se requiere un inspector NVML"),
            _result("G03", "Configuración MIG", False, True, {}, "Se requiere un inspector NVML"),
        ]
    persistence, mig = inspector.persistence_mode(), inspector.mig_configuration()
    return [
        check_gpu_foreign_activity(inspector),
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
    manifest: Any, env: Any, catalog: Mapping[str, Any], *, sysfs: SysfsPaths | None = None,
    node_profile: Any = None, gpu_inspector: GpuInspector | None = None,
    uncore_probe: Callable[[], tuple[bool, str]] | None = None,
) -> list[CheckResult]:
    """Ejecuta todos los checks de campaña que disponen de datos antes de la matriz."""
    cores = _cores(manifest)
    sysfs = sysfs or load_config().sysfs
    node_id = _value(node_profile, "node_id", None)
    rapl = _value(manifest, "rapl", {})
    gpu = _value(manifest, "gpu", {})
    turbo = _value(manifest, "turbo", {})
    results = [
        check_turbo_hwp(
            sysfs.cpu_root,
            require_disabled=bool(_value(turbo, "require_disabled", False)),
        ),
        check_numa(cores, getattr(env, "numa_cpu_map", {})),
        check_smt(env, manifest),
        check_rapl_wrap(env, rapl_enabled=bool(_value(rapl, "enabled", False)), rapl_root=sysfs.rapl_root),
    ]
    cgroup_path = _value(manifest, "cgroup_path")
    if cgroup_path:
        results.append(check_cgroup_clean(cgroup_path))
    if _requires_frequency_control(manifest):
        control_paths = _value(env, "frequency_control_paths", None)
        # ARC-94/95: "userspace"/scaling_governor solo son requisito real del
        # kernel para la estrategia discrete_bounds (scaling_setspeed,
        # acpi-cpufreq); para bounded_range (intel_pstate/amd-pstate, fija
        # min=max=target) no hay governor obligatorio -- ver check_governor().
        # ARC-95: E09 tenía el mismo bug que E07 ya corrigió -- exigía
        # scaling_governor escribible sin importar la estrategia, bloqueando
        # con exactamente el permiso P1 solicitado (solo min/max).
        strategy = _value(env, "frequency_control_strategy", None)
        required_attrs = (
            ("scaling_min_freq", "scaling_max_freq") if strategy == "bounded_range"
            else ("scaling_governor", "scaling_min_freq", "scaling_max_freq")
        )
        results.append(check_frequency_write_permission(cores, sysfs.cpu_root, control_paths, required_attrs))
        expected_governor = "userspace" if strategy == "discrete_bounds" else None
        results.append(check_governor(cores, expected_governor, sysfs.cpu_root, control_paths))
        results.append(check_frequency_domain(cores, _value(env, "frequency_domain_cpus", None)))
    results.append(check_rapl_domains(_value(rapl, "domains", []), _value(env, "rapl_domains_available", []), bool(_value(rapl, "enabled", False))))
    uncore = _value(manifest, "uncore", {})
    uncore_enabled = bool(_value(uncore, "enabled", False))
    results.append(check_exclusive_node_allocation(
        uncore_enabled,
        os.sched_getaffinity(0),
        os.cpu_count() or 0,
    ))
    refs = tuple(_value(manifest, "calibration", ())) + tuple(_value(manifest, "kernels", ()))
    entries = [catalog[reference] for reference in refs]
    # ARC-191: E12 debe mirar solo `kernels:` (role=="dataset"), NO
    # `calibration:`. Las corridas de calibración (STREAM/ERT) nunca pasan
    # por postprocess_run()/validate_windows() -- calibration.py las
    # ejecuta con run_single() directo y lee su ancho de banda/FLOPs del
    # propio stdout (bandwidth_stdout_pattern/flops_stdout_pattern), nunca
    # de uncore_imc. E12 existe para blindar contra "corrida de CPU en la
    # MATRIZ que nunca llega a quality_status=ok" (ARC-123); una entrada de
    # calibración jamás pasa por esa matriz, así que incluirla en
    # has_cpu_kernel exigía uncore.enabled=True incluso en una campaña sin
    # NINGÚN kernel de CPU en el dataset -- MAN-07 obliga a declarar
    # stream_official/ert_probe en `calibration:` SIEMPRE (son la única
    # fuente de ancho de banda/FLOPs para I_ridge de CPU), así que antes de
    # esta corrección ninguna campaña, ni siquiera una 100% GPU, podía
    # desactivar uncore. `entries` (arriba, combinado) se sigue usando tal
    # cual para los demás chequeos (binario/checksum/success_check/memoria),
    # que sí aplican por igual a calibración y a dataset.
    dataset_refs = tuple(_value(manifest, "kernels", ()))
    dataset_entries = [catalog[reference] for reference in dataset_refs]
    results.append(check_uncore_required_for_cpu_dataset(dataset_entries, uncore_enabled))
    # E13 va junto a E12 a propósito: E12 comprueba que el manifiesto PIDA
    # uncore, E13 que el nodo pueda DARLO. Los dos fallan igual aguas abajo
    # (toda ventana de CPU en intensity_undefined) pero por causas
    # independientes, y el segundo puede aparecer sin que cambie nada del
    # repositorio.
    results.append(check_uncore_readable(uncore_enabled, probe=uncore_probe))
    output_dir, overwrite = _value(manifest, "output_dir"), bool(_value(manifest, "overwrite", False))
    results.append(_result("I07", "Directorio de campaña", overwrite or not Path(output_dir).exists(), True, {"output_dir": str(output_dir)}, "output_dir ya existe"))
    projected = _value(manifest, "projected_campaign_bytes")
    if isinstance(projected, int) and projected >= 0:
        results.append(check_disk_space(output_dir, projected))
    else:
        results.append(_result("I09", "Espacio libre", False, True, {"projected_bytes": "not_declared"}, "Debe declararse projected_campaign_bytes"))
    if _value(gpu, "enabled", False):
        results.extend(check_gpu(gpu_inspector))
    for entry in entries:
        results.extend([check_binary_exists(entry), check_binary_checksum(entry, node_id), check_success_check(entry), check_memory_size(entry)])
    results.append(check_toolchain(bool(_value(manifest, "rebuild", False))))
    results.append(check_perf_counter_capacity(_HARNESS_PERF_EVENTS, _value(node_profile, "pmc_count", _value(env, "pmc_count"))))
    results.append(check_core_hour_budget(_value(manifest, "remaining_core_hours"), _value(manifest, "projected_core_hours")))
    return results


def run_reduced_preflight(manifest: Any, env: Any, entry: Any, run_id: str, *, expected_governor: str | None = None, load_threshold: float = 1.0, turbo_snapshot: Mapping[str, Any] | None = None, cpu_root: str | Path | None = None, load_reader: Callable[[], tuple[float, float, float]] = os.getloadavg, node_id: str | None = None) -> list[CheckResult]:
    """Ejecuta checks por corrida, incluidos C01/C02 para detectar binarios cambiados.

    ARC-94: ``expected_governor=None`` (el default) deriva el valor correcto
    de ``env.frequency_control_strategy`` -- "userspace" solo para
    discrete_bounds, ninguno para bounded_range (ver check_governor()). Un
    valor explícito sigue anulando la derivación, para pruebas o casos
    excepcionales.
    """
    results = [
        check_temperature(_value(manifest, "package_temperature_c"), _value(manifest, "temperature_min_c", 0.0), _value(manifest, "temperature_max_c", 90.0)),
        check_foreign_processes(_value(manifest, "foreign_affinity_pids", ())),
        check_external_load(load_threshold, load_reader, max(len(_cores(manifest)), 1)),
        check_run_id_unique(_value(manifest, "output_dir"), run_id, bool(_value(manifest, "overwrite", False))),
        check_binary_exists(entry),
        check_binary_checksum(entry, node_id),
        check_success_check(entry),
    ]
    if _requires_frequency_control(manifest):
        strategy = _value(env, "frequency_control_strategy", None)
        governor = expected_governor if expected_governor is not None else ("userspace" if strategy == "discrete_bounds" else None)
        results.append(check_governor(_cores(manifest), governor, cpu_root, _value(env, "frequency_control_paths", None)))
    if turbo_snapshot is not None:
        results.append(check_turbo_hwp_unchanged(turbo_snapshot, cpu_root))
    return results
