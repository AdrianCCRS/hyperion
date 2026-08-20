#!/usr/bin/env python3
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


def main() -> int:
    level_id = sys.argv[1] if len(sys.argv) > 1 else "F0"
    manifest = manifest_module.load(MANIFEST_PATH)
    catalog = catalog_module.load_catalog(str(manifest.catalog_path))
    config = load_config()
    delegated = ",".join(str(cpu) for cpu in manifest.cores.delegated_cpus)
    env = environment_module.detect_environment(delegated, config=config)
    try:
        env.pmc_count = environment_module.probe_pmc_count()
    except Exception:
        env.pmc_count = 0

    entry = catalog["npb_mg"]
    result = runner.run_single(
        entry, manifest, "npb_mg", level_id, 1,
        environment_profile=env, node_id=NODE_ID, harness=config.harness,
        apply_frequency=freqctl.apply_frequency, run_id=f"cal07_npbmg_{level_id}_rep01",
    )
    print(json.dumps({"success": result.success, "exit_code": result.exit_code, "run_dir": str(result.run_dir)}, indent=2))
    if not result.success:
        return 1
    verdict, summary = validation_module.validate_cpu_frequency_trace(
        result.run_dir / "samples.csv", require_per_window=True,
        expected_khz=getattr(result.applied_frequency, "requested_khz", None),
        tolerance_fraction=manifest.frequency_validation.get("tolerance_fraction"),
        expected_cpu_count=len(manifest.cores.delegated_cpus),
    )
    print(json.dumps({"accepted": verdict.accepted, "factor_id": verdict.factor_id, "message": verdict.message, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
