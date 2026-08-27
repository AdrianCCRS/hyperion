from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re

import yaml

from .config import HarnessConfig, load_config

@dataclass
class KernelEntry:
    """
    Represents a kernel entry in the catalog.
    """
    id: str
    suite: str
    role: str
    exec_path: str
    # str: un solo checksum, valido en cualquier nodo (comportamiento
    # historico). dict[node_id, checksum]: el mismo codigo fuente compilado
    # con toolchains distintos por nodo (p.ej. gnu14/14.2.0 en felix vs gcc
    # 12.4.0 en pacca) produce binarios funcionalmente identicos pero con
    # hash distinto -- un solo checksum global rompia C02 en cualquier nodo
    # que no fuera el que lo genero. Ver verify_binary().
    binary_checksum: str | dict[str, str]
    phase_label_hint: str | None #Solo si role == "dataset"
    size_variant: str | None #Opcional
    expected_runtime_seconds: int | None #Opcional
    warmup_seconds: float | None #Opcional
    success_check: dict | None #Opcional
    estimated_memory_bytes: int | None = None
    reports_bandwidth_stdout: bool = False
    reports_flops_stdout: bool = False
    # CAL-02/CAL-03: regex with exactly one capturing group around the numeric
    # value, applied to the suite's own stdout. Required when the matching
    # reports_*_stdout flag is set; never used to read PMU counters.
    bandwidth_stdout_pattern: str | None = None
    flops_stdout_pattern: str | None = None
    # CAL-02/CAL-03/ARC-43: el número que captura el regex está en la unidad
    # nativa que la suite imprime (STREAM reporta MB/s, no B/s; nuestro
    # ert_probe reporta GFLOP/s, no FLOP/s) -- calibration.py multiplica por
    # esto antes de guardarlo como bw_pico_bytes_per_s/p_pico_flops_per_s.
    # Default 1.0 = el regex ya captura la unidad SI base (bytes/s, FLOP/s).
    bandwidth_stdout_unit_multiplier: float = 1.0
    flops_stdout_unit_multiplier: float = 1.0
    # ARC-100: estos tres campos alimentaban el prorrateo de FLOPs por
    # instrucciones (POST-09), retirado del pipeline una vez confirmada la
    # medición directa por hardware (FP_ARITH_INST_RETIRED, ARC-97/98/99) --
    # ver docs/libro/main.tex Marco Conceptual. Ya no los consume ningún
    # código de producción; se conservan declarados en el catálogo como
    # metadato histórico/de auditoría (el propio total/tasa que cada kernel
    # reporta sigue siendo información real, solo que operational_intensity
    # ya no depende de ella).
    flops_total_stdout_pattern: str | None = None
    flops_rate_stdout_pattern: str | None = None
    runtime_seconds_stdout_pattern: str | None = None
    exec_args: str = ""
    # ARC-70: "cpu" (default) o "gpu". Un kernel gpu necesita --enable-gpu en
    # el launcher (telemetria NVML) y el shim LD_PRELOAD que fuerza
    # cudaDeviceScheduleBlockingSync (ver runner.py/gpu_shim.py) -- ningún
    # otro campo del catálogo cambia de significado por esto, es puramente
    # una bandera de qué wiring adicional aplica en runner.py.
    device: str = "cpu"
    # ARC-80: intensidad operacional (FLOPs/byte) medida offline con `ncu`,
    # una sola vez por kernel (Fase 1 -- ver Diseno_Politica_DVFS_CPU_GPU.md
    # sección 3) -- nunca calculada en vivo desde NVML, que no expone FLOPs
    # ni bytes. Solo aplica a kernels dataset con device="gpu"; postprocess.py
    # la compara contra el i_ridge_gpu calibrado (calibration.run_gpu_calibration,
    # por nivel de frecuencia y precisión) para derivar phase_label_train.
    operational_intensity_flops_per_byte: float | None = None
    # ARC-80: "fp32"/"fp64" -- confirmado leyendo el .cu de cada kernel, no
    # asumido (ARC-76: la mayoría de Rodinia es FP32, lavaMD es FP64). Para
    # un kernel dataset determina contra cuál de los dos ridge points de GPU
    # se compara operational_intensity_flops_per_byte; para un kernel de
    # calibración con reports_flops_stdout=True identifica cuál de los dos
    # calibradores (gpu_ert_probe_fp32/fp64) es -- ninguno de los dos se
    # puede inferir del resto del catálogo, así que se declara explícito.
    gpu_precision: str | None = None
    # Selector CPU/GPU (pivote 2026-08-27): identifica la MISMA configuracion
    # logica (operacion + dimensiones) entre su par cpu y su par gpu, p.ej.
    # "gemm_M1024_N1024_K1024" para dual_gemm_M1024_cpu Y dual_gemm_M1024_gpu.
    # No lo consume el orquestador (que ya despacha por device via el campo
    # `device` de arriba); es el join key que usa el analisis posterior para
    # armar el dataset nivel 2 (argmin EDP sobre {device}x{freq_level}).
    config_id: str | None = None

    def __post_init__(self):
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"CAT-09: device de {self.id!r} debe ser 'cpu' o 'gpu', no {self.device!r}")
        self.validate_role_requirements()
        # CAT-03 / C03: validate the check type and compile regexes before runs.
        if not isinstance(self.success_check, dict):
            raise ValueError("C03: success_check debe ser un objeto")

        check_type = self.success_check.get("type")

        if check_type == "exit_code":
            expected = self.success_check.get("expected", 0)
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise ValueError("C03: exit_code.expected debe ser un entero")
            return

        if check_type == "stdout_regex":
            pattern = self.success_check.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(
                    "C03: stdout_regex.pattern debe ser un string no vacío"
                )
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"C03: stdout_regex.pattern inválido: {error}"
                ) from error
            return

        raise ValueError(f"C03: tipo de success_check no soportado: {check_type!r}")

    def validate_role_requirements(self) -> None:
        # CAT-04/C05: los kernels dataset declaran metadatos y memoria estimada.
        if self.role == "dataset":
            required = {
                "phase_label_hint": self.phase_label_hint,
                "size_variant": self.size_variant,
                "expected_runtime_seconds": self.expected_runtime_seconds,
                "warmup_seconds": self.warmup_seconds,
                "estimated_memory_bytes": self.estimated_memory_bytes,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"CAT-04: kernel dataset {self.id!r} requiere {', '.join(missing)}"
                )
            # ARC-80: un kernel dataset de GPU sin esto no puede recibir
            # phase_label_train -- fallar cerrado en vez de dejarlo pasar
            # como "queda en None" (ARC-20: nunca aprobar por falta de datos).
            if self.device == "gpu":
                if self.operational_intensity_flops_per_byte is None or not self.gpu_precision:
                    raise ValueError(
                        f"CAT-10: kernel dataset GPU {self.id!r} requiere "
                        "operational_intensity_flops_per_byte y gpu_precision "
                        "(medidos con ncu, ARC-80) para poder derivar phase_label_train"
                    )
                if self.gpu_precision not in ("fp32", "fp64"):
                    raise ValueError(
                        f"CAT-10: gpu_precision de {self.id!r} debe ser 'fp32' o 'fp64', "
                        f"no {self.gpu_precision!r}"
                    )

        # CAT-05: a calibration entry reports exactly one calibration metric.
        if self.role == "calibration" and (
            self.reports_bandwidth_stdout == self.reports_flops_stdout
        ):
            raise ValueError(
                f"CAT-05: kernel calibration {self.id!r} debe reportar exactamente "
                "una de bandwidth_stdout o flops_stdout"
            )
        # CAL-02/CAL-03: the regex that will extract the peak metric from
        # stdout must be declared and valid at catalog-load time, not
        # discovered as a KeyError mid-calibration.
        if self.reports_bandwidth_stdout:
            self._validate_metric_pattern("bandwidth_stdout_pattern", self.bandwidth_stdout_pattern)
        if self.reports_flops_stdout:
            self._validate_metric_pattern("flops_stdout_pattern", self.flops_stdout_pattern)
        if self.flops_total_stdout_pattern is not None:
            self._validate_metric_pattern("flops_total_stdout_pattern", self.flops_total_stdout_pattern)
        if self.flops_rate_stdout_pattern is not None:
            self._validate_metric_pattern("flops_rate_stdout_pattern", self.flops_rate_stdout_pattern)
        if self.runtime_seconds_stdout_pattern is not None:
            self._validate_metric_pattern("runtime_seconds_stdout_pattern", self.runtime_seconds_stdout_pattern)

    def _validate_metric_pattern(self, field_name: str, pattern: str | None) -> None:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"CAT-05: {field_name} de {self.id!r} debe ser un string no vacío")
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"CAT-05: {field_name} de {self.id!r} inválido: {error}") from error
        if compiled.groups < 1:
            raise ValueError(f"CAT-05: {field_name} de {self.id!r} debe tener un grupo de captura numérico")
        
def load_catalog(catalog_path: str) -> dict[str, KernelEntry]:
    with open(catalog_path, encoding="utf-8") as catalog_file:
        document = yaml.safe_load(catalog_file) or {}
    kernels = document.get("kernels", [])
    if not isinstance(kernels, list):
        raise ValueError("El campo kernels debe ser una lista")

    entries: dict[str, KernelEntry] = {}
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError("Cada entrada de kernels debe ser un objeto")
        kernel_id = kernel.get("id")
        # CAT-08: reject duplicate IDs instead of silently overwriting an entry.
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("CAT-08: cada entrada requiere un id no vacío")
        if kernel_id in entries:
            raise ValueError(f"CAT-08: id de catálogo duplicado: {kernel_id!r}")

        entry = KernelEntry(
            id=kernel["id"],
            suite=kernel["suite"],
            role=kernel["role"],
            exec_path=kernel["exec_path"],
            binary_checksum=kernel["binary_checksum"],
            phase_label_hint=kernel.get("phase_label_hint"),
            size_variant=kernel.get("size_variant"),
            expected_runtime_seconds=kernel.get("expected_runtime_seconds"),
            warmup_seconds=kernel.get("warmup_seconds"),
            success_check=kernel.get("success_check"),
            estimated_memory_bytes=kernel.get("estimated_memory_bytes"),
            reports_bandwidth_stdout=kernel.get("reports_bandwidth_stdout", False),
            reports_flops_stdout=kernel.get("reports_flops_stdout", False),
            bandwidth_stdout_pattern=kernel.get("bandwidth_stdout_pattern"),
            flops_stdout_pattern=kernel.get("flops_stdout_pattern"),
            bandwidth_stdout_unit_multiplier=kernel.get("bandwidth_stdout_unit_multiplier", 1.0),
            flops_stdout_unit_multiplier=kernel.get("flops_stdout_unit_multiplier", 1.0),
            flops_total_stdout_pattern=kernel.get("flops_total_stdout_pattern"),
            flops_rate_stdout_pattern=kernel.get("flops_rate_stdout_pattern"),
            runtime_seconds_stdout_pattern=kernel.get("runtime_seconds_stdout_pattern"),
            exec_args=kernel.get("exec_args", ""),
            device=kernel.get("device", "cpu"),
            operational_intensity_flops_per_byte=kernel.get("operational_intensity_flops_per_byte"),
            gpu_precision=kernel.get("gpu_precision"),
            config_id=kernel.get("config_id"),
        )
        if not isinstance(entry.exec_args, str):
            raise ValueError(f"CAT-06: exec_args de {entry.id!r} debe ser un string")
        entries[entry.id] = entry
    return entries

def verify_binary(entry: KernelEntry, node_id: str | None = None) -> bool:
    """
    C02: sha256(entry.exec_path) == entry.binary_checksum.

    CAT-07: call this same check during campaign preflight and immediately
    before each individual run; it has no cached result.
    Retorna CheckResult con factor_id "C01" o "C02" segun cual falle.

    Si entry.binary_checksum es un dict (checksum por nodo, ver KernelEntry),
    node_id es obligatorio para resolverlo -- sin node_id, o si node_id no
    tiene entrada en el dict, esto falla cerrado (False), nunca aprueba por
    omisión ni cae de vuelta a "cualquier valor sirve".
    """
    # CAT-01 / C01: require a regular executable file before using it.
    if not os.path.isfile(entry.exec_path) or not os.access(entry.exec_path, os.X_OK):
        return False

    try:
        with open(entry.exec_path, "rb") as binary_file:
            checksum = f"sha256:{hashlib.file_digest(binary_file, 'sha256').hexdigest()}"
    except OSError:
        return False

    if isinstance(entry.binary_checksum, dict):
        if node_id is None or node_id not in entry.binary_checksum:
            return False
        expected = entry.binary_checksum[node_id]
    else:
        expected = entry.binary_checksum
    # CAT-02 / C02: reject a binary changed since the catalog was generated.
    return checksum == expected

def resolve_exec_command(
    entry: KernelEntry, harness: HarnessConfig | None = None
) -> list[str]:
    """Traduce una entrada al argv del launcher, sin inferir argumentos."""
    # CAT-06: los flags vienen de la configuración de plataforma; exec_args es la única fuente de argumentos de suite.
    launcher = harness or load_config().harness
    return [launcher.exec_flag, entry.exec_path, launcher.exec_args_flag, entry.exec_args]
