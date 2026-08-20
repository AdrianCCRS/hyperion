#!/usr/bin/env python3
"""ARC-156: segundo control para el hallazgo de ARC-152 (ert_probe con cero
muestras bajo F0). Hipotesis a probar: una condicion de carrera entre la
escritura de scaling_min_freq/scaling_max_freq (6 CPUs delegados) y el
arranque del colector de telemetria (perf_event_open en esos mismos CPUs)
-- inserta una pausa configurable entre apply_frequency() y el lanzamiento
real para ver si recupera las muestras.

Uso:
    python3 diagnose_cal07_control2.py <pausa_segundos>
"""
import json
import sys
import time

sys.path.insert(0, "/home/latorresn/hyperion")

from orchestrator import catalog as catalog_module
from orchestrator import environment as environment_module
from orchestrator import freqctl
from orchestrator import manifest as manifest_module
from orchestrator import runner
from orchestrator import validation as validation_module
from orchestrator.config import load_config

MANIFEST_PATH = "/home/latorresn/hyperion/orchestrator/schemas/campaign_pacca_dvfs_smoke.yaml"
NODE_ID = "pacca-a100"


def main() -> int:
    pause_s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

    manifest = manifest_module.load(MANIFEST_PATH)
    catalog = catalog_module.load_catalog(str(manifest.catalog_path))

    config = load_config()
    delegated = ",".join(str(cpu) for cpu in manifest.cores.delegated_cpus)
    env = environment_module.detect_environment(delegated, config=config)
    try:
        env.pmc_count = environment_module.probe_pmc_count()
    except Exception:
        env.pmc_count = 0

    entry = catalog["ert_probe"]

    # Aplica la frecuencia a mano (mismo mecanismo que run_single usaria
    # internamente) para poder insertar la pausa ANTES de lanzar.
    level = None
    for candidate in manifest.frequency_levels:
        if candidate.id == "F0":
            level = candidate
    assert level is not None
    applied = freqctl.apply_frequency(manifest.cores.delegated_cpus, level, env)
    print(json.dumps({"applied_khz": applied.applied_khz, "pause_s": pause_s}, indent=2))

    if pause_s > 0:
        time.sleep(pause_s)

    result = runner.run_single(
        entry, manifest, "ert_probe", "F0", 1,
        environment_profile=env, node_id=NODE_ID, harness=config.harness,
        apply_frequency=None,  # ya aplicada arriba, run_single no debe repetirla
        run_id=f"cal07_control2_ert_probe_F0_pause{pause_s}",
    )

    print(json.dumps({
        "success": result.success,
        "exit_code": result.exit_code,
        "run_dir": str(result.run_dir),
    }, indent=2))

    if not result.success:
        return 1

    verdict, summary = validation_module.validate_cpu_frequency_trace(
        result.run_dir / "samples.csv",
        require_per_window=True,
        expected_khz=applied.requested_khz,
        tolerance_fraction=manifest.frequency_validation.get("tolerance_fraction"),
        expected_cpu_count=len(manifest.cores.delegated_cpus),
    )
    print(json.dumps({
        "accepted": verdict.accepted,
        "factor_id": verdict.factor_id,
        "message": verdict.message,
        **summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
