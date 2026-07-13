from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Sequence

from .catalog import load_catalog
from .config import load_config
from .environment import detect_environment
from .manifest import compute_matrix_size, load


_SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_CPUS_ON_NODE",
    "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_LIST",
    "SLURM_JOB_NODELIST",
)


def _runtime_context() -> dict[str, object]:
    """Recopila el contexto del proceso sin alterar el nodo ni la asignación."""
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        cgroup = []
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "effective_cpus": sorted(os.sched_getaffinity(0)),
        "cgroup": cgroup,
        "slurm": {name: os.environ[name] for name in _SLURM_VARIABLES if name in os.environ},
    }


def create_startup_diagnostic(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    delegated_cpus: str | None = None,
) -> Path:
    """Crea un artefacto de diagnóstico sin ejecutar una campaña ni modificar sysfs."""
    manifest = load(manifest_path)
    catalog = load_catalog(str(manifest.catalog_path))
    configured_cpus = ",".join(str(cpu) for cpu in manifest.cores.delegated_cpus)
    detected_cpus = delegated_cpus or configured_cpus
    environment = detect_environment(detected_cpus, config=load_config(config_path))
    environment_data = asdict(environment)
    for field in ("delegated_cpus", "numa_cpu_map", "delegated_cpu_numa_nodes", "perf_events_available"):
        fallback = {} if field in {"numa_cpu_map", "delegated_cpu_numa_nodes"} else []
        environment_data[field] = getattr(environment, field, fallback)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "startup_diagnostic.json"
    artifact.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "manifest": {
                    "campaign_id": manifest.campaign_id,
                    "environment_tier": manifest.environment_tier,
                    "matrix_size": compute_matrix_size(manifest),
                    "catalog_path": str(manifest.catalog_path),
                    "declared_delegated_cpus": list(manifest.cores.delegated_cpus),
                },
                "catalog": {
                    "loaded": True,
                    "entry_count": len(catalog),
                    "entries": sorted(catalog),
                },
                "environment": environment_data,
                "runtime": _runtime_context(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un diagnóstico de arranque sin ejecutar kernels ni escribir en sysfs."
    )
    parser.add_argument("--manifest", required=True, help="Ruta a campaign.yaml")
    parser.add_argument("--output-dir", required=True, help="Directorio donde se escribirá el JSON")
    parser.add_argument("--config", help="Ruta opcional a orchestrator.toml")
    parser.add_argument(
        "--use-allowed-cpus",
        action="store_true",
        help="Detecta usando los CPUs permitidos al proceso por Slurm/cgroup, no los declarados en el manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada para ejecutar el diagnóstico desde una asignación Slurm."""
    arguments = _parser().parse_args(argv)
    allowed_cpus = None
    if arguments.use_allowed_cpus:
        allowed_cpus = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    artifact = create_startup_diagnostic(
        arguments.manifest,
        arguments.output_dir,
        config_path=arguments.config,
        delegated_cpus=allowed_cpus,
    )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
