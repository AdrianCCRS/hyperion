from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from .catalog import KernelEntry, load_catalog

logger = logging.getLogger(__name__)


class ManifestValidationError(ValueError):
    """Error de un campo del manifest, identificado por su regla o factor."""

    def __init__(self, rule_id: str, field: str, message: str) -> None:
        self.rule_id = rule_id
        self.field = field
        super().__init__(f"{rule_id}: {field}: {message}")


@dataclass(frozen=True)
class FrequencyLevel:
    id: str
    mode: str
    fraction: float | None = None


@dataclass(frozen=True)
class Cores:
    delegated_cpus: tuple[int, ...]
    collector_cpu: int
    consumer_cpu: int
    numa_node_pin: int | None


@dataclass(frozen=True)
class Timeouts:
    ready: int
    run: int
    shutdown: int


@dataclass(frozen=True)
class Combination:
    kernel_ref: str
    frequency_level: FrequencyLevel
    repetition_index: int
    # ARC-129: nivel de GPU de esta combinación cuando manifest.gpu_frequency_levels
    # existe y el kernel es device=="gpu" (producto cartesiano CPU x GPU real,
    # ver campaign.build_matrix). None en todo otro caso -- kernels de CPU, o
    # kernels de GPU en una campaña que no declaró gpu_frequency_levels
    # (acoplado al eje de CPU, comportamiento anterior sin cambios).
    gpu_frequency_level: FrequencyLevel | None = None


@dataclass(frozen=True)
class Manifest:
    campaign_id: str
    environment_tier: str
    seed: int
    output_dir: Path
    overwrite: bool
    catalog_path: Path
    calibration: tuple[str, ...]
    kernels: tuple[str, ...]
    frequency_levels: tuple[FrequencyLevel, ...]
    repetitions_per_combination: int
    target_windows_per_repetition: int
    interval_ns: int
    running_ratio_min: float
    cores: Cores
    smt_policy: str
    cgroup_path: str | None
    perf_enabled: bool
    rapl: Mapping[str, Any]
    gpu: Mapping[str, Any]
    timeouts_seconds: Timeouts
    # D03/CAL-04: valores de ficha técnica declarados para el chequeo de
    # plausibilidad de la calibración Roofline (±40%). Ausente = el chequeo
    # D03 no puede aprobarse (ver calibration.py); nunca se infiere.
    hardware_datasheet: Mapping[str, float] | None = None
    # ARC-73: I09/OPS-01 (preflight.py) exigen estos tres campos -- ausentes
    # aquí desde siempre, así que ningún manifiesto podía pasar
    # run_campaign_preflight() automático (ARC-58) hasta esta corrección.
    # Opcionales a nivel de schema a propósito (None es un valor de manifest
    # válido) -- preflight.py, no manifest.py, decide que "no declarado"
    # bloquea (I09/OPS-01 fallan cerrado, nunca aprueban por omisión).
    projected_campaign_bytes: int | None = None
    remaining_core_hours: float | None = None
    projected_core_hours: float | None = None
    # ARC-102: umbral de carga externa normalizada (E08, load_1m/cpu_count)
    # por encima del cual una combinación o la calibración se rechaza antes
    # de medir -- opcional, ausente usa el default conservador de
    # campaign.py (1.0) para no romper manifiestos existentes que no lo
    # declaraban (no existía como campo real hasta este cambio, aunque
    # run_campaign() ya lo aceptaba como parámetro).
    load_threshold: float | None = None
    # ARC-88: runner.build_command() ya lee esto con getattr(manifest,
    # "gpu_interval_ns", None), pero nunca estaba declarado como campo real
    # del manifiesto -- ningún YAML podía fijarlo, así que toda campaña GPU
    # de producción caía siempre en el default de 100ms del launcher
    # (collector.hpp), el mismo valor que ARC-83/84/86 ya habían encontrado
    # insuficiente para varios kernels GPU cortos. None = usar el default
    # del launcher (nunca se infiere ni se fuerza un valor aquí).
    gpu_interval_ns: int | None = None
    # ARC-116: contadores uncore_imc (CAS_COUNT_READ/WRITE, DRAM real en vez
    # del proxy cache_misses*line_size). Opcional, {} = deshabilitado --
    # ausente en todo manifiesto de campañas anteriores, nunca se infiere
    # habilitado. Cuando enabled=True, preflight.py (E11) exige que el job
    # tenga el nodo completo (--exclusive): son contadores de ámbito
    # sistema/socket, no por-PID.
    uncore: Mapping[str, Any] = field(default_factory=dict)
    # ARC-129: eje de frecuencia de GPU independiente del de CPU. None (el
    # default, y el único valor de todo manifiesto anterior a este cambio)
    # preserva el comportamiento acoplado de siempre -- runner.py reusa
    # frequency_levels para el eje GPU cuando esto está ausente. Cuando SÍ
    # se declara, campaign.build_matrix() arma el producto cartesiano
    # completo frequency_levels x gpu_frequency_levels para cada kernel
    # device=="gpu" (decisión explícita del usuario, no la opción acotada
    # que se había recomendado -- multiplica el tamaño de la matriz GPU).
    gpu_frequency_levels: tuple[FrequencyLevel, ...] | None = None
    # Control global de turbo y validación del reloj efectivo. El helper que
    # cambia no_turbo vive en el wrapper operacional de pacca; el manifiesto
    # solo declara el estado requerido y cómo validar la traza ya medida.
    turbo: Mapping[str, Any] = field(default_factory=dict)
    frequency_validation: Mapping[str, Any] = field(default_factory=dict)
    temperature: Mapping[str, Any] = field(default_factory=dict)
    # ARC-161: en paccaA100, energy_performance_preference=performance bajo
    # HWP hace que el hardware decaiga lentamente hacia un techo de
    # frecuencia mas bajo tras venir de un nivel mas alto -- confirmado con
    # scaling_cur_freq muestreado en vivo, en una escala de segundos, no
    # milisegundos (docs/retoma/pacca/Diagnostico_CAL07_Dispersion_Frecuencia_STREAM_20260819.md).
    # Ausente/{} = deshabilitado, mismo criterio que turbo/uncore/gpu --
    # nunca se infiere habilitado. Cuando enabled=True, apply_frequency() se
    # sigue de una espera activa (relee scaling_cur_freq hasta que los CPUs
    # delegados caigan dentro de tolerance_fraction del objetivo, o falla
    # con FrequencyControlError tras timeout_seconds) en vez de una pausa
    # fija -- un barrido real (0.5s a 12s) mostro que el asentamiento NO es
    # monotono con el tiempo esperado (8s asento limpio, 12s inmediatamente
    # despues fallo peor que 8s), asi que una pausa ciega no es confiable.
    frequency_settle: Mapping[str, Any] = field(default_factory=dict)
    # 2026-08-25: CAM-04 medía overhead de instrumentación (baseline sin
    # perf vs. telemetry) en TODAS las repeticiones de TODAS las
    # combinaciones -- duplica el número de procesos lanzados en cada
    # campaña. Con 540 pares ya medidos (arc174), el overhead está
    # caracterizado con potencia de sobra (media 1.95%, estable entre
    # kernels, varía sobre todo por nivel de frecuencia -- ver
    # docs/general/Estrategia_CPU_Fase2.md). None (default) preserva el
    # comportamiento de siempre -- ningún manifiesto existente cambia de
    # costo sin declararlo explícitamente. Un valor no vacío restringe el
    # baseline a las repeticiones listadas (p.ej. (1,) = solo la primera de
    # cada combinación), como vigilancia de que el overhead no se desvió,
    # no como medición completa de nuevo.
    baseline_repetition_indices: tuple[int, ...] | None = None


def _error(rule_id: str, field: str, message: str) -> None:
    raise ManifestValidationError(rule_id, field, message)


def _required(document: Mapping[str, Any], field: str) -> Any:
    if field not in document:
        _error("MAN-00", field, "campo obligatorio ausente")
    return document[field]


def _parse_cpu_list(value: Any, field: str) -> tuple[int, ...]:
    if isinstance(value, str):
        cpus: set[int] = set()
        for token in value.split(","):
            try:
                first, last = token.split("-", 1) if "-" in token else (token, token)
                cpus.update(range(int(first), int(last) + 1))
            except ValueError:
                _error("MAN-06", field, "rango de CPU inválido")
        return tuple(sorted(cpus))
    if isinstance(value, list) and all(isinstance(cpu, int) and not isinstance(cpu, bool) for cpu in value):
        return tuple(sorted(set(value)))
    _error("MAN-06", field, "debe ser una lista de enteros o un rango")


def _parse_references(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _error("MAN-09", field, "debe ser una lista")
    refs: list[str] = []
    for item in value:
        reference = item.get("kernel_ref") if isinstance(item, Mapping) else item
        if not isinstance(reference, str) or not reference:
            _error("MAN-09", field, "cada entrada debe declarar kernel_ref")
        refs.append(reference)
    return tuple(refs)


def _parse_frequency_levels(value: Any, *, field_name: str = "frequency_levels") -> tuple[FrequencyLevel, ...]:
    if not isinstance(value, list) or not value:
        _error("MAN-10", field_name, "debe ser una lista no vacía")
    levels: list[FrequencyLevel] = []
    native_levels = 0
    for index, item in enumerate(value):
        field = f"{field_name}[{index}]"
        if not isinstance(item, Mapping):
            _error("MAN-10", field, "debe ser un objeto")
        level_id, mode = item.get("id"), item.get("mode")
        if not isinstance(level_id, str) or not level_id:
            _error("MAN-10", f"{field}.id", "debe ser un texto no vacío")
        if mode == "native_governor":
            native_levels += 1
            levels.append(FrequencyLevel(level_id, mode))
            continue
        if mode != "fixed":
            _error("MAN-10", f"{field}.mode", "debe ser fixed o native_governor")
        fraction = item.get("fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            _error("MAN-10", f"{field}.fraction", "debe ser numérica")
        if not 0.0 <= float(fraction) <= 1.0:
            _error("MAN-10", f"{field}.fraction", "debe estar en [0.0, 1.0]")
        levels.append(FrequencyLevel(level_id, mode, float(fraction)))
    if native_levels != 1:
        _error("MAN-10", field_name, "debe contener exactamente un nivel native_governor")
    return tuple(levels)


def _parse_gpu_frequency_levels(value: Any) -> tuple[FrequencyLevel, ...] | None:
    # ARC-129: eje de GPU independiente del de CPU -- ausente (None) es el
    # caso normal (toda campaña sin producto cartesiano CPU x GPU, incluida
    # cualquier campaña sin kernels de GPU) y preserva el comportamiento
    # acoplado anterior (runner.py reusa frequency_levels para el eje GPU
    # cuando esto es None). Mismas reglas de validación que frequency_levels
    # (MAN-10: lista no vacía, exactamente un nivel native_governor) cuando
    # SÍ se declara -- no tiene sentido barrer GPU sin un REF propio.
    if value is None:
        return None
    return _parse_frequency_levels(value, field_name="gpu_frequency_levels")


_DATASHEET_KEYS = ("bw_pico_bytes_per_s", "p_pico_flops_per_s")


def _parse_hardware_datasheet(value: Any) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        _error("MAN-00", "hardware_datasheet", "debe ser un objeto o estar ausente")
    parsed: dict[str, float] = {}
    for key in _DATASHEET_KEYS:
        raw = value.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
            _error("MAN-00", f"hardware_datasheet.{key}", "debe ser numérico y positivo")
        parsed[key] = float(raw)
    return parsed or None


def _parse_optional_non_negative_number(document: Mapping[str, Any], field: str) -> float | None:
    # ARC-73: ausente es válido (preflight.py decide si eso bloquea, nunca
    # manifest.py) -- solo se valida el tipo/rango cuando SÍ está declarado.
    value = document.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        _error("MAN-00", field, "debe ser numérico y no negativo, o estar ausente")
    return float(value)


def compute_matrix_size(manifest: Manifest) -> int:
    """Devuelve las combinaciones telemetry: kernel × frecuencia × repetición."""
    return (
        len(manifest.kernels)
        * len(manifest.frequency_levels)
        * manifest.repetitions_per_combination
    )


def _validate_catalog_references(
    calibration: tuple[str, ...], kernels: tuple[str, ...], catalog: Mapping[str, KernelEntry]
) -> None:
    # MAN-09 se resuelve antes de detectar o modificar el nodo.
    unknown = (set(calibration) | set(kernels)) - set(catalog)
    if unknown:
        _error("MAN-09", "calibration/kernels", f"kernel_ref inexistente: {', '.join(sorted(unknown))}")
    overlap = set(calibration) & set(kernels)
    if overlap:
        _error("MAN-08", "calibration/kernels", f"kernel_ref repetido entre roles: {', '.join(sorted(overlap))}")
    calibration_entries = [catalog[reference] for reference in calibration]
    if not any(entry.reports_bandwidth_stdout for entry in calibration_entries) or not any(
        entry.reports_flops_stdout for entry in calibration_entries
    ):
        _error("MAN-07", "calibration", "I_ridge no es calculable: faltan referencias de ancho de banda o FLOP/s")
    # MAN-12 (ARC-94): nada validaba que `kernels:` solo tuviera entradas
    # role=="dataset" -- un manifiesto real (campaign_pacca_gpu_ref.yaml)
    # metía 4 kernels role=="calibration" directamente en `kernels:`,
    # generando filas de windows.csv sin phase_label_hint/etiqueta posible
    # y multiplicando la matriz de combinaciones sin necesidad. Simétrico
    # para `calibration:`, que tampoco debería aceptar un kernel de
    # dataset.
    wrong_role_kernels = sorted(ref for ref in kernels if getattr(catalog[ref], "role", "dataset") != "dataset")
    if wrong_role_kernels:
        _error(
            "MAN-12", "kernels",
            f"kernel_ref con role != 'dataset' no puede declararse en kernels: {', '.join(wrong_role_kernels)}",
        )
    wrong_role_calibration = sorted(
        ref for ref in calibration if getattr(catalog[ref], "role", "calibration") != "calibration"
    )
    if wrong_role_calibration:
        _error(
            "MAN-12", "calibration",
            f"kernel_ref con role != 'calibration' no puede declararse en calibration: {', '.join(wrong_role_calibration)}",
        )


def load(path: str | Path) -> Manifest:
    """Carga y valida un campaign.yaml antes de ejecutar cualquier operación de nodo."""
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as source_file:
        document = yaml.safe_load(source_file) or {}
    if not isinstance(document, Mapping):
        _error("MAN-00", "manifest", "la raíz YAML debe ser un objeto")

    # No se aceptan valores por defecto silenciosos para estos campos reproducibles.
    seed = _required(document, "seed")
    overwrite = _required(document, "overwrite")
    cgroup_path = _required(document, "cgroup_path")
    if isinstance(seed, bool) or not isinstance(seed, int):
        _error("MAN-05", "seed", "debe ser un entero declarado explícitamente")
    if not isinstance(overwrite, bool):
        _error("MAN-04", "overwrite", "debe ser booleano y estar declarado explícitamente")
    if cgroup_path is not None and not isinstance(cgroup_path, str):
        _error("MAN-01", "cgroup_path", "debe ser texto o null")

    tier = _required(document, "environment_tier")
    if not isinstance(tier, str):
        _error("MAN-01", "environment_tier", "debe ser texto")
    # MAN-01: cgroup_path es opcional en TODOS los tiers, incluido hpc_sc3.
    # Antes de ARC-40 aquí se exigía un cgroup_path no nulo para hpc_sc3 --
    # un resabio de cuando perf se adjuntaba por cgroup (previo a la
    # migración PID+inherit de Fase 1, CPP-01..08). check_foreign_processes
    # (E06) ya no depende de cgroups: escanea Cpus_allowed real de procesos
    # vivos, un mecanismo estrictamente más fuerte (detecta contención de
    # caché/ancho de banda por afinidad real, no por membresía de cgroup,
    # y no requiere delegación de cgroup del clúster). Ver ARC-41.

    repetitions = _required(document, "repetitions_per_combination")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 3:
        _error("MAN-02", "repetitions_per_combination", "el mínimo es 3")

    output_dir_value = _required(document, "output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        _error("MAN-04", "output_dir", "debe ser una ruta no vacía")
    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = source_path.parent / output_dir
    if output_dir.exists() and not overwrite:
        _error("I07", "output_dir", "ya existe y overwrite es false")

    cores_value = _required(document, "cores")
    if not isinstance(cores_value, Mapping):
        _error("MAN-06", "cores", "debe ser un objeto")
    delegated = _parse_cpu_list(cores_value.get("delegated_cpus"), "cores.delegated_cpus")
    collector, consumer = cores_value.get("collector_cpu"), cores_value.get("consumer_cpu")
    if any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in (collector, consumer)):
        _error("MAN-06", "cores.collector_cpu/cores.consumer_cpu", "deben ser enteros")
    if set(delegated) & {collector, consumer} or collector == consumer:
        _error("MAN-06", "cores", "delegated_cpus, collector_cpu y consumer_cpu no pueden solaparse")
    numa_node_pin = cores_value.get("numa_node_pin")
    if numa_node_pin is not None and (isinstance(numa_node_pin, bool) or not isinstance(numa_node_pin, int)):
        _error("MAN-06", "cores.numa_node_pin", "debe ser un entero o null")
    cores = Cores(delegated, collector, consumer, numa_node_pin)
    smt_policy = _required(document, "smt_policy")
    if smt_policy not in {"all_threads", "one_thread_per_physical_core"}:
        _error("MAN-00", "smt_policy", "debe ser all_threads o one_thread_per_physical_core")

    frequency_levels = _parse_frequency_levels(_required(document, "frequency_levels"))
    gpu_frequency_levels = _parse_gpu_frequency_levels(document.get("gpu_frequency_levels"))
    interval_ns = _required(document, "interval_ns")
    if isinstance(interval_ns, bool) or not isinstance(interval_ns, int) or interval_ns <= 0:
        _error("MAN-11", "interval_ns", "debe ser un entero mayor que cero")
    running_ratio = _required(document, "running_ratio_min")
    if isinstance(running_ratio, bool) or not isinstance(running_ratio, (int, float)) or not 0.0 < float(running_ratio) <= 1.0:
        _error("MAN-11", "running_ratio_min", "debe estar en (0.0, 1.0]")

    calibration = _parse_references(_required(document, "calibration"), "calibration")
    kernels = _parse_references(_required(document, "kernels"), "kernels")
    catalog_path_value = _required(document, "catalog_path")
    if not isinstance(catalog_path_value, str) or not catalog_path_value:
        _error("MAN-09", "catalog_path", "debe ser una ruta no vacía")
    catalog_path = Path(catalog_path_value)
    if not catalog_path.is_absolute():
        catalog_path = source_path.parent / catalog_path
    catalog = load_catalog(str(catalog_path))
    _validate_catalog_references(calibration, kernels, catalog)

    campaign_id = _required(document, "campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        _error("MAN-00", "campaign_id", "debe ser texto no vacío")
    target_windows = _required(document, "target_windows_per_repetition")
    if isinstance(target_windows, bool) or not isinstance(target_windows, int) or target_windows <= 0:
        _error("MAN-00", "target_windows_per_repetition", "debe ser un entero mayor que cero")
    perf_enabled = _required(document, "perf_enabled")
    rapl, gpu = _required(document, "rapl"), _required(document, "gpu")
    timeout_data = _required(document, "timeouts_seconds")
    if not isinstance(perf_enabled, bool) or not isinstance(rapl, Mapping) or not isinstance(gpu, Mapping):
        _error("MAN-00", "perf_enabled/rapl/gpu", "campos con tipo inválido")
    if not isinstance(timeout_data, Mapping):
        _error("MAN-00", "timeouts_seconds", "debe ser un objeto")
    try:
        timeouts = Timeouts(*(timeout_data[name] for name in ("ready", "run", "shutdown")))
    except KeyError as error:
        _error("MAN-00", f"timeouts_seconds.{error.args[0]}", "campo obligatorio ausente")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (timeouts.ready, timeouts.run, timeouts.shutdown)):
        _error("MAN-00", "timeouts_seconds", "todos los valores deben ser enteros positivos")

    hardware_datasheet = _parse_hardware_datasheet(document.get("hardware_datasheet"))
    projected_campaign_bytes_raw = _parse_optional_non_negative_number(document, "projected_campaign_bytes")
    projected_campaign_bytes = (
        int(projected_campaign_bytes_raw) if projected_campaign_bytes_raw is not None else None
    )
    remaining_core_hours = _parse_optional_non_negative_number(document, "remaining_core_hours")
    projected_core_hours = _parse_optional_non_negative_number(document, "projected_core_hours")
    load_threshold = _parse_optional_non_negative_number(document, "load_threshold")

    # ARC-88: mismo criterio de validación que interval_ns (entero positivo),
    # pero opcional -- ausente significa "usar el default del launcher"
    # (collector.hpp, 100ms), nunca se infiere ni se fuerza aquí.
    gpu_interval_ns_raw = document.get("gpu_interval_ns")
    if gpu_interval_ns_raw is not None and (
        isinstance(gpu_interval_ns_raw, bool) or not isinstance(gpu_interval_ns_raw, int) or gpu_interval_ns_raw <= 0
    ):
        _error("MAN-11", "gpu_interval_ns", "debe ser un entero mayor que cero, o estar ausente")

    # ARC-116: ausente = {} (deshabilitado), mismo criterio que rapl/gpu --
    # nunca se infiere habilitado.
    uncore_raw = document.get("uncore", {})
    if not isinstance(uncore_raw, Mapping):
        _error("MAN-00", "uncore", "debe ser un objeto")

    turbo_raw = document.get("turbo", {})
    if not isinstance(turbo_raw, Mapping):
        _error("MAN-00", "turbo", "debe ser un objeto")
    require_disabled = turbo_raw.get("require_disabled", False)
    if not isinstance(require_disabled, bool):
        _error("MAN-00", "turbo.require_disabled", "debe ser booleano")

    frequency_validation_raw = document.get("frequency_validation", {})
    if not isinstance(frequency_validation_raw, Mapping):
        _error("MAN-00", "frequency_validation", "debe ser un objeto")
    require_per_window = frequency_validation_raw.get("require_per_window", False)
    tolerance_fraction = frequency_validation_raw.get("tolerance_fraction")
    if not isinstance(require_per_window, bool):
        _error("MAN-00", "frequency_validation.require_per_window", "debe ser booleano")
    if tolerance_fraction is not None and (
        isinstance(tolerance_fraction, bool)
        or not isinstance(tolerance_fraction, (int, float))
        or not 0 <= float(tolerance_fraction) < 1
    ):
        _error(
            "MAN-00", "frequency_validation.tolerance_fraction",
            "debe ser un número en el intervalo [0, 1)",
        )
    if require_per_window and tolerance_fraction is None:
        _error(
            "MAN-00", "frequency_validation.tolerance_fraction",
            "es obligatorio cuando require_per_window=true",
        )
    grace_seconds = frequency_validation_raw.get("grace_seconds", 0.0)
    if isinstance(grace_seconds, bool) or not isinstance(grace_seconds, (int, float)) or grace_seconds < 0:
        _error(
            "MAN-00", "frequency_validation.grace_seconds",
            "debe ser un número mayor o igual que cero",
        )
    tail_grace_seconds = frequency_validation_raw.get("tail_grace_seconds", 0.0)
    if (
        isinstance(tail_grace_seconds, bool)
        or not isinstance(tail_grace_seconds, (int, float))
        or tail_grace_seconds < 0
    ):
        _error(
            "MAN-00", "frequency_validation.tail_grace_seconds",
            "debe ser un número mayor o igual que cero",
        )

    frequency_settle_raw = document.get("frequency_settle", {})
    if not isinstance(frequency_settle_raw, Mapping):
        _error("MAN-00", "frequency_settle", "debe ser un objeto")
    settle_enabled = frequency_settle_raw.get("enabled", False)
    if not isinstance(settle_enabled, bool):
        _error("MAN-00", "frequency_settle.enabled", "debe ser booleano")
    if settle_enabled:
        settle_timeout = frequency_settle_raw.get("timeout_seconds")
        settle_tolerance = frequency_settle_raw.get("tolerance_fraction")
        settle_poll = frequency_settle_raw.get("poll_interval_seconds", 0.2)
        if (
            isinstance(settle_timeout, bool)
            or not isinstance(settle_timeout, (int, float))
            or float(settle_timeout) <= 0
        ):
            _error(
                "MAN-00", "frequency_settle.timeout_seconds",
                "es obligatorio y debe ser numérico mayor que cero cuando enabled=true",
            )
        if (
            isinstance(settle_tolerance, bool)
            or not isinstance(settle_tolerance, (int, float))
            or not 0 <= float(settle_tolerance) < 1
        ):
            _error(
                "MAN-00", "frequency_settle.tolerance_fraction",
                "es obligatorio y debe ser un número en [0, 1) cuando enabled=true",
            )
        if (
            isinstance(settle_poll, bool)
            or not isinstance(settle_poll, (int, float))
            or float(settle_poll) <= 0
            or float(settle_poll) >= float(settle_timeout)
        ):
            _error(
                "MAN-00", "frequency_settle.poll_interval_seconds",
                "debe ser numérico mayor que cero y menor que timeout_seconds",
            )

    temperature_raw = document.get("temperature", {})
    if not isinstance(temperature_raw, Mapping):
        _error("MAN-00", "temperature", "debe ser un objeto")
    require_package_sensor = temperature_raw.get("require_package_sensor", False)
    temperature_min_c = temperature_raw.get("minimum_c", 0.0)
    temperature_max_c = temperature_raw.get("maximum_c", 90.0)
    if not isinstance(require_package_sensor, bool):
        _error("MAN-00", "temperature.require_package_sensor", "debe ser booleano")
    for field_name, value in (
        ("temperature.minimum_c", temperature_min_c),
        ("temperature.maximum_c", temperature_max_c),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _error("MAN-00", field_name, "debe ser numérico")
    if float(temperature_min_c) >= float(temperature_max_c):
        _error("MAN-00", "temperature", "minimum_c debe ser menor que maximum_c")

    baseline_reps_raw = document.get("baseline_repetition_indices")
    baseline_repetition_indices: tuple[int, ...] | None = None
    if baseline_reps_raw is not None:
        if (
            not isinstance(baseline_reps_raw, list)
            or not baseline_reps_raw
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in baseline_reps_raw)
        ):
            _error(
                "MAN-00", "baseline_repetition_indices",
                "debe ser una lista no vacía de enteros >= 1, o ausente para medir siempre",
            )
        if any(v > repetitions for v in baseline_reps_raw):
            _error(
                "MAN-00", "baseline_repetition_indices",
                f"no puede referenciar una repetición > repetitions_per_combination ({repetitions})",
            )
        baseline_repetition_indices = tuple(sorted(set(baseline_reps_raw)))

    manifest = Manifest(
        campaign_id, tier, seed, output_dir, overwrite, catalog_path, calibration, kernels,
        frequency_levels, repetitions, target_windows, interval_ns, float(running_ratio),
        cores, smt_policy, cgroup_path, perf_enabled, dict(rapl), dict(gpu), timeouts,
        hardware_datasheet, projected_campaign_bytes, remaining_core_hours, projected_core_hours,
        load_threshold, gpu_interval_ns_raw, dict(uncore_raw), gpu_frequency_levels,
        dict(turbo_raw), dict(frequency_validation_raw), dict(temperature_raw),
        dict(frequency_settle_raw), baseline_repetition_indices,
    )
    matrix_size = compute_matrix_size(manifest)
    if baseline_repetition_indices is None:
        # MAN-03: cada combinación tiene baseline y telemetry, por eso se duplica.
        n_baseline = matrix_size
    else:
        # Solo las repeticiones listadas emparejan baseline -- ver
        # schedule_runs()/CAM-04. matrix_size ya incluye todas las
        # repeticiones, así que se prorratea por cuántas de ellas están
        # en baseline_repetition_indices.
        n_baseline = matrix_size * len(baseline_repetition_indices) // repetitions
    logger.info(
        "Matriz de campaña: %d combinaciones (%d con baseline: %d corridas)",
        matrix_size,
        n_baseline,
        matrix_size + n_baseline,
    )
    return manifest
