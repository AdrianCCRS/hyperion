from __future__ import annotations

from dataclasses import dataclass
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


def _parse_frequency_levels(value: Any) -> tuple[FrequencyLevel, ...]:
    if not isinstance(value, list) or not value:
        _error("MAN-10", "frequency_levels", "debe ser una lista no vacía")
    levels: list[FrequencyLevel] = []
    native_levels = 0
    for index, item in enumerate(value):
        field = f"frequency_levels[{index}]"
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
        _error("MAN-10", "frequency_levels", "debe contener exactamente un nivel native_governor")
    return tuple(levels)


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
    if tier == "hpc_sc3" and not cgroup_path:
        _error("MAN-01", "cgroup_path", "es obligatorio cuando environment_tier es hpc_sc3")

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

    manifest = Manifest(
        campaign_id, tier, seed, output_dir, overwrite, catalog_path, calibration, kernels,
        frequency_levels, repetitions, target_windows, interval_ns, float(running_ratio),
        cores, smt_policy, cgroup_path, perf_enabled, dict(rapl), dict(gpu), timeouts,
        hardware_datasheet,
    )
    matrix_size = compute_matrix_size(manifest)
    # MAN-03: cada combinación tiene baseline y telemetry, por eso se duplica.
    logger.info(
        "Matriz de campaña: %d combinaciones (×2 por baseline: %d corridas)",
        matrix_size,
        matrix_size * 2,
    )
    return manifest
