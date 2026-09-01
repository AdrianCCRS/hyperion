from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class HarnessConfig:
    exec_flag: str
    exec_args_flag: str
    binary_path: str


@dataclass(frozen=True)
class SysfsPaths:
    cpu_root: Path
    rapl_root: Path
    numa_root: Path
    perf_events_root: Path
    drm_root: Path

    @classmethod
    def from_base(cls, base_sys_path: str | Path) -> "SysfsPaths":
        """Construye rutas de sysfs virtuales para pruebas sin usar /sys real."""
        base = Path(base_sys_path)
        return cls(
            cpu_root=base / "devices/system/cpu",
            rapl_root=base / "class/powercap/intel-rapl",
            numa_root=base / "devices/system/node",
            perf_events_root=base / "bus/event_source/devices/cpu/events",
            drm_root=base / "class/drm",
        )


@dataclass(frozen=True)
class DetectionConfig:
    slurm_env_var: str
    tier_hpc: str
    tier_local: str
    tier_cloud: str
    tier_override_env_var: str = "HYPERION_ENVIRONMENT_TIER"


@dataclass(frozen=True)
class OrchestratorConfig:
    harness: HarnessConfig
    sysfs: SysfsPaths
    detection: DetectionConfig


def _text(section: dict, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"hpc_config.toml: {key} debe ser un texto no vacío")
    return value


def load_config(path: str | Path | None = None) -> OrchestratorConfig:
    """Carga la configuración de plataforma, separada del manifest de campaña.

    Por defecto busca ``hpc_config.toml`` en la raíz de ``common/`` (dos
    niveles arriba de este archivo: ``common/hpc/config.py`` ->
    ``common/``). No depende del directorio de trabajo del proceso que lo
    invoca -- a diferencia de ``harness.binary_path`` dentro del TOML, que sí
    es relativo a la raíz del repositorio (ver nota en ``hpc_config.toml``).
    """
    config_path = Path(path) if path is not None else Path(__file__).parent.parent / "hpc_config.toml"
    with config_path.open("rb") as config_file:
        document = tomllib.load(config_file)
    try:
        harness_data = document["harness"]
        sysfs_data = document["sysfs"]
        detection_data = document["detection"]
    except KeyError as error:
        raise ValueError(f"hpc_config.toml: sección obligatoria ausente: {error.args[0]}") from error
    if not all(isinstance(section, dict) for section in (harness_data, sysfs_data, detection_data)):
        raise ValueError("hpc_config.toml: cada sección debe ser una tabla")
    # binary_path is relative to the repo root (this file's parent directory)
    # unless it is already absolute, mirroring how manifest.py resolves
    # catalog_path relative to the manifest source file.
    binary_path = _text(harness_data, "binary_path")
    resolved_binary_path = Path(binary_path)
    if not resolved_binary_path.is_absolute():
        resolved_binary_path = config_path.parent / resolved_binary_path
    return OrchestratorConfig(
        harness=HarnessConfig(
            _text(harness_data, "exec_flag"),
            _text(harness_data, "exec_args_flag"),
            str(resolved_binary_path),
        ),
        sysfs=SysfsPaths(
            Path(_text(sysfs_data, "cpu_root")),
            Path(_text(sysfs_data, "rapl_root")),
            Path(_text(sysfs_data, "numa_root")),
            Path(_text(sysfs_data, "perf_events_root")),
            Path(_text(sysfs_data, "drm_root")),
        ),
        detection=DetectionConfig(
            _text(detection_data, "slurm_env_var"),
            _text(detection_data, "tier_hpc"),
            _text(detection_data, "tier_local"),
            _text(detection_data, "tier_cloud"),
            _text(detection_data, "tier_override_env_var"),
        ),
    )
