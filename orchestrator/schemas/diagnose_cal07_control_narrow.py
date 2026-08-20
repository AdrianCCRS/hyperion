#!/usr/bin/env python3
"""ARC-156: prueba si el bug de "cero muestras" (min==max exacto en
scaling_min_freq/scaling_max_freq, confirmado en F0 Y F4) desaparece con un
rango angosto pero de ancho NO cero (target +/- 1000 kHz) en vez de un
punto exacto -- workaround candidato, no toca produccion todavia.
"""
import json
import sys

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
MARGIN_KHZ = 1000


def main() -> int:
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

    level = next(lvl for lvl in manifest.frequency_levels if lvl.id == "F0")
    target = freqctl._target_khz(level, env.available_frequencies_khz)
    target_min = max(target - MARGIN_KHZ, min(env.available_frequencies_khz))
    target_max = min(target + MARGIN_KHZ, max(env.available_frequencies_khz))

    per_cpu_applied = {}
    for cpu in manifest.cores.delegated_cpus:
        min_path = freqctl._attr_path(env, cpu, freqctl._MIN_ATTR)
        max_path = freqctl._attr_path(env, cpu, freqctl._MAX_ATTR)
        freqctl._write_range_safe(min_path, max_path, target_min, target_max, cpu=cpu)
        per_cpu_applied[cpu] = freqctl._read_int(min_path)

    print(json.dumps({"target_min_khz": target_min, "target_max_khz": target_max, "per_cpu_applied": per_cpu_applied}, indent=2))

    result = runner.run_single(
        entry, manifest, "ert_probe", "F0", 1,
        environment_profile=env, node_id=NODE_ID, harness=config.harness,
        apply_frequency=None, run_id="cal07_control_narrow_ert_probe_F0_rep01",
    )

    print(json.dumps({
        "success": result.success,
        "exit_code": result.exit_code,
        "run_dir": str(result.run_dir),
    }, indent=2))

    if result.success:
        verdict, summary = validation_module.validate_cpu_frequency_trace(
            result.run_dir / "samples.csv",
            require_per_window=True,
            expected_khz=target,
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
