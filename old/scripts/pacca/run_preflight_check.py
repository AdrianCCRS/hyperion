"""Preflight real contra paccaA100 (Unicartagena), sin escribir nada en sysfs.

Adaptado de scripts/felix/run_preflight_check.py -- diferencias: node_id,
RAPL habilitado (paccaA100 SI tiene RAPL legible, a diferencia de felix),
y delegated_cpus se mantiene en 0-5 porque cae dentro de los hilos
primarios del socket 0 (0-7) sin tocar sus siblings SMT (16-23), ver
docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md seccion 3.

Correr con el venv ~/hyperion-venv activo y ~/hyperion-kernels como cwd
(catalog.yaml usa exec_path relativo). Resultados de environment/
node_profile en ~/hyperion-results/validation/<run_id>/.
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "hyperion"))

from orchestrator import environment, node_profile, catalog as catalog_module, preflight, config


def main() -> None:
    delegated_cpus = "0-5"
    cfg = config.load_config()
    sysfs = cfg.sysfs

    print("=== environment.detect_environment() ===")
    env = environment.detect_environment(delegated_cpus, config=cfg)
    env.pmc_count = environment.probe_pmc_count()
    print("tier:", env.tier, "| scaling_driver:", env.scaling_driver)
    print("frequency_write_capable:", env.frequency_write_capable, "| pmc_count:", env.pmc_count)
    print("frequency_domain_cpus:", env.frequency_domain_cpus)

    out_dir = Path.home() / "hyperion-results" / "validation" / f"f42_preflight_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    environment.write_environment_report(env, out_dir)

    print()
    print("=== node_profile.build_node_profile() ===")
    profile = node_profile.build_node_profile(
        env, [0, 1, 2, 3, 4, 5], node_id="pacca-a100", hostname=socket.gethostname()
    )
    node_profile.write_node_profile(profile, out_dir)
    print("pmc_count:", getattr(profile, "pmc_count", None))

    print()
    print("=== catalog.load_catalog() ===")
    catalog = catalog_module.load_catalog(
        str(Path.home() / "hyperion" / "orchestrator" / "schemas" / "kernels" / "catalog.yaml")
    )
    kernel_ids = [k for k, entry in catalog.items() if entry.role == "dataset"]
    calibration_ids = [k for k, entry in catalog.items() if entry.role == "calibration"]
    print("kernels:", kernel_ids, "| calibration:", calibration_ids)

    manifest = {
        "cores": {"delegated_cpus": [0, 1, 2, 3, 4, 5]},
        "smt_policy": "one_thread_per_physical_core",
        "rapl": {"enabled": True},
        "gpu": {"enabled": False},
        "output_dir": out_dir,
        "overwrite": True,
        "calibration": calibration_ids,
        "kernels": kernel_ids,
        "frequency_levels": [{"id": "REF", "mode": "native_governor"}],
        "projected_campaign_bytes": 200_000_000,
        "remaining_core_hours": 1000.0,
        "projected_core_hours": 10.0,
        "rebuild": False,
        "perf_events": ["instructions", "cycles", "cache-references", "cache-misses"],
    }

    print()
    print("=== preflight.run_campaign_preflight() ===")
    results = preflight.run_campaign_preflight(
        manifest, env, catalog, sysfs=sysfs, node_profile=profile, gpu_inspector=None,
    )
    for result in results:
        status = "PASA" if result.passed else ("FALLA(bloqueante)" if result.blocking else "FALLA(advertencia)")
        print(f"[{result.factor_id:8s}] {status:20s} {result.name} -- {result.message}")

    failed_blocking = [r.factor_id for r in results if not r.passed and r.blocking]
    print()
    print("=== RESUMEN ===")
    print("total checks:", len(results), "| fallas bloqueantes:", failed_blocking)
    print("TODO EN VERDE:", not failed_blocking)


if __name__ == "__main__":
    main()
