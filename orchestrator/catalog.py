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
    warmpup_seconds: int | None #Opcional
    success_check: dict | None #Opcional
    reports_bandwidth_stdout: bool = False
    reports_flops_stdout: bool = False

    def __post_init__(self):
        #VALIDACIONES C03: success_check
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
        #FIN VALIDACIONES C03
        
def load_catalog(catalog_path: str) -> dict[str, KernelEntry]:
    with open(catalog_path, encoding="utf-8") as catalog_file:
        kernels = yaml.safe_load(catalog_file).get("kernels", [])

    return {
        kernel["id"]: KernelEntry(
            id=kernel["id"],
            suite=kernel["suite"],
            role=kernel["role"],
            exec_path=kernel["exec_path"],
            binary_checksum=kernel["binary_checksum"],
            phase_label_hint=kernel.get("phase_label_hint"),
            size_variant=kernel.get("size_variant"),
            expected_runtime_seconds=kernel.get("expected_runtime_seconds"),
            warmpup_seconds=kernel.get("warmup_seconds"),
            success_check=kernel.get("success_check"),
            reports_bandwidth_stdout=kernel.get("reports_bandwidth_stdout", False),
            reports_flops_stdout=kernel.get("reports_flops_stdout", False),
        )
        for kernel in kernels
    }

def verify_binary(entry: KernelEntry) -> bool:
    """
    C02: sha256(entry.exec_path) == entry.binary_checksum
    Retorna CheckResult con factor_id "C01" o "C02" segun cual falle.
    """
    # C01: Check if the file exists and is executable by the current user
    if not os.path.isfile(entry.exec_path) or not os.access(entry.exec_path, os.X_OK):
        return False
    
    try:
        with open(entry.exec_path, "rb") as binary_file:
            # Compute the SHA256 checksum of the binary file
            checksum = f"sha256:{hashlib.file_digest(binary_file, 'sha256').hexdigest()}"
    except OSError:
        return False
    # C02: Check if the binary checksum matches
    return checksum == entry.binary_checksum

def resolve_exec_command(entry: KernelEntry) -> list[str]:
    """Resuelve el comando de ejecución para un kernel."""
    return ["--exec", entry.exec_path, "--exec-args", ""]
