from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re

import yaml

@dataclass
class KernelEntry:
    """
    Represents a kernel entry in the catalog.
    """
    id: str
    suite: str
    role: str
    exec_path: str
    binary_checksum: str
    phase_label_hint: str | None #Solo si role == "dataset"
    size_variant: str | None #Opcional
    expected_runtime_seconds: int | None #Opcional
    warmup_seconds: float | None #Opcional
    success_check: dict | None #Opcional
    reports_bandwidth_stdout: bool = False
    reports_flops_stdout: bool = False
    exec_args: str = ""

    def __post_init__(self):
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
        # CAT-04: dataset kernels require the metadata used to characterize runs.
        if self.role == "dataset":
            required = {
                "phase_label_hint": self.phase_label_hint,
                "size_variant": self.size_variant,
                "expected_runtime_seconds": self.expected_runtime_seconds,
                "warmup_seconds": self.warmup_seconds,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"CAT-04: kernel dataset {self.id!r} requiere {', '.join(missing)}"
                )

        # CAT-05: a calibration entry reports exactly one calibration metric.
        if self.role == "calibration" and (
            self.reports_bandwidth_stdout == self.reports_flops_stdout
        ):
            raise ValueError(
                f"CAT-05: kernel calibration {self.id!r} debe reportar exactamente "
                "una de bandwidth_stdout o flops_stdout"
            )
        
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
            reports_bandwidth_stdout=kernel.get("reports_bandwidth_stdout", False),
            reports_flops_stdout=kernel.get("reports_flops_stdout", False),
            exec_args=kernel.get("exec_args", ""),
        )
        if not isinstance(entry.exec_args, str):
            raise ValueError(f"CAT-06: exec_args de {entry.id!r} debe ser un string")
        entries[entry.id] = entry
    return entries

def verify_binary(entry: KernelEntry) -> bool:
    """
    C02: sha256(entry.exec_path) == entry.binary_checksum.

    CAT-07: call this same check during campaign preflight and immediately
    before each individual run; it has no cached result.
    Retorna CheckResult con factor_id "C01" o "C02" segun cual falle.
    """
    # CAT-01 / C01: require a regular executable file before using it.
    if not os.path.isfile(entry.exec_path) or not os.access(entry.exec_path, os.X_OK):
        return False
    
    try:
        with open(entry.exec_path, "rb") as binary_file:
            checksum = f"sha256:{hashlib.file_digest(binary_file, 'sha256').hexdigest()}"
    except OSError:
        return False
    # CAT-02 / C02: reject a binary changed since the catalog was generated.
    return checksum == entry.binary_checksum

def resolve_exec_command(entry: KernelEntry) -> list[str]:
    """Traduce una entrada al argv del launcher, sin inferir argumentos."""
    # CAT-06: exec_args is the sole source of suite arguments; keep empty values.
    return ["--exec", entry.exec_path, "--exec-args", entry.exec_args]
