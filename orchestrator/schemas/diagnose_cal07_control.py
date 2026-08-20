#!/usr/bin/env python3
"""ARC-152: control de diagnostico para CAL-07 (ver
docs/retoma/pacca/Diagnostico_CAL07_Dispersion_Frecuencia_STREAM_20260819.md).

Corre ert_probe (compute-bound) directamente vía runner.run_single() a F0,
con la MISMA aplicacion de frecuencia y el mismo pipeline de telemetria que
usa la calibracion real -- sin pasar por manifest.calibration (que exige
stream_official + ert_probe juntos, MAN-07, y el primero ya crashea CAL-07
antes de llegar al segundo). Aisla si la dispersion de scaling_cur_freq bajo
F0 depende del patron de ejecucion de STREAM (memory-bound) o es un
problema general del candado de frecuencia.

Uso (dentro de with_cpu_turbo_disabled.sh, srun con --exclusive):
    python3 diagnose_cal07_control.py [FREQ_LEVEL_ID]

FREQ_LEVEL_ID (opcional, default F0): cualquier id declarado en
frequency_levels del manifiesto (F0, F0125, F1, ..., F4) -- ARC-156 lo usa
para aislar si el bug de "cero muestras" depende de fijar EXACTAMENTE en el
techo (F0) o de cualquier rango de ancho cero (min==max) sin importar el
punto.
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

    entry = catalog["ert_probe"]

    result = runner.run_single(
        entry, manifest, "ert_probe", level_id, 1,
        environment_profile=env, node_id=NODE_ID, harness=config.harness,
        apply_frequency=freqctl.apply_frequency, run_id=f"cal07_control_ert_probe_{level_id}_rep01",
    )

    print(json.dumps({
        "success": result.success,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "run_dir": str(result.run_dir),
    }, indent=2))

    if not result.success:
        return 1

    verdict, summary = validation_module.validate_cpu_frequency_trace(
        result.run_dir / "samples.csv",
        require_per_window=True,
        expected_khz=getattr(result.applied_frequency, "requested_khz", None),
        tolerance_fraction=manifest.frequency_validation.get("tolerance_fraction"),
        expected_cpu_count=len(manifest.cores.delegated_cpus),
        grace_seconds=float(manifest.frequency_validation.get("grace_seconds", 0.0)),
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
