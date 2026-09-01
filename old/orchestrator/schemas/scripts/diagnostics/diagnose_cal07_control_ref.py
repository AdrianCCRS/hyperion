#!/usr/bin/env python3
"""ARC-156: descarta que el bug de "cero muestras" (ARC-152) sea un
artefacto del script de diagnostico en si (en vez de algo especifico de
F0) -- corre ert_probe en REF por el MISMO camino standalone que
diagnose_cal07_control.py, replicando como campaign.py liga el estado
original via functools.partial(freqctl.apply_frequency, original=...)
para que native_governor funcione igual que en produccion.
"""
import functools
import json
import sys

sys.path.insert(0, "/home/latorresn/hyperion")

from orchestrator import catalog as catalog_module
from orchestrator import environment as environment_module
from orchestrator import freqctl
from orchestrator import manifest as manifest_module
from orchestrator import runner
from orchestrator.config import load_config

MANIFEST_PATH = "/home/latorresn/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dvfs_smoke.yaml"
NODE_ID = "pacca-a100"


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

    original_state = freqctl.snapshot_original_state(manifest.cores.delegated_cpus, env)
    bound_apply_frequency = functools.partial(freqctl.apply_frequency, original=original_state)

    result = runner.run_single(
        entry, manifest, "ert_probe", "REF", 1,
        environment_profile=env, node_id=NODE_ID, harness=config.harness,
        apply_frequency=bound_apply_frequency, run_id="cal07_control_ref_ert_probe_REF_rep01",
    )

    print(json.dumps({
        "success": result.success,
        "exit_code": result.exit_code,
        "run_dir": str(result.run_dir),
    }, indent=2))

    if result.success:
        import csv
        with open(result.run_dir / "samples.csv", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("tag") == "CPU"]
        print(json.dumps({"cpu_rows_in_samples_csv": len(rows)}, indent=2))

    freqctl.restore_original_state(original_state, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
