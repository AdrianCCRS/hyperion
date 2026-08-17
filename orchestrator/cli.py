from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from . import calibration as calibration_module
from . import campaign as campaign_module
from . import catalog as catalog_module
from . import diagnostics as diagnostics_module
from . import environment as environment_module
from . import manifest as manifest_module
from . import node_profile as node_profile_module
from . import postprocess as postprocess_module
from . import preflight as preflight_module
from . import report as report_module
from . import validation as validation_module
from .config import load_config

logger = logging.getLogger(__name__)


def _delegated_cpus_arg(manifest) -> str:
    return ",".join(str(cpu) for cpu in manifest.cores.delegated_cpus)


def _load_manifest_and_catalog(manifest_path: str):
    manifest = manifest_module.load(manifest_path)
    catalog = catalog_module.load_catalog(str(manifest.catalog_path))
    return manifest, catalog


def _detect_environment(manifest, config_path: str | None):
    config = load_config(config_path) if config_path else load_config()
    env = environment_module.detect_environment(_delegated_cpus_arg(manifest), config=config)
    # D05: probado en vivo aquí (no dentro de detect_environment(), que debe
    # seguir siendo una lectura pura de sysfs, ENV-01) porque requiere correr
    # `perf stat` como subproceso. Si perf no está disponible, queda en 0 --
    # D05 bloquea con datos ausentes en vez de aprobar por omisión.
    try:
        env.pmc_count = environment_module.probe_pmc_count()
    except Exception:
        logger.warning("No se pudo medir pmc_count con perf stat; D05 bloqueará hasta resolverlo", exc_info=True)
        env.pmc_count = 0
    return env


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Delegates to diagnostics.py's own CLI (read-only, no kernel runs)."""
    forwarded = ["--manifest", args.manifest, "--output-dir", args.output_dir]
    if args.config:
        forwarded += ["--config", args.config]
    if args.use_allowed_cpus:
        forwarded.append("--use-allowed-cpus")
    return diagnostics_module.main(forwarded)


def cmd_calibrate(args: argparse.Namespace) -> int:
    manifest, catalog = _load_manifest_and_catalog(args.manifest)
    env = _detect_environment(manifest, args.config)

    roofline = calibration_module.run_calibration(
        manifest, catalog, environment_profile=env, node_id=args.node_id,
    )
    profile = node_profile_module.build_node_profile(
        env, manifest.cores.delegated_cpus, node_id=args.node_id, hostname=args.hostname or "",
    )
    node_profile_module.write_node_profile(profile, manifest.output_dir)
    references = calibration_module.run_calibration_references(
        catalog[args.reference_kernel_ref], manifest, args.reference_kernel_ref,
        node_id=args.node_id, environment_profile=env,
    )

    print(json.dumps({
        "roofline_calibration_ref": str(Path(manifest.output_dir) / "roofline_calibration.json"),
        "node_profile_ref": str(Path(manifest.output_dir) / "node_profile.json"),
        "calibration_ref": str(Path(manifest.output_dir) / "calibration_references.json"),
        "plausibility_check_passed": roofline.plausibility_check_passed,
        "calibration_references_accepted": references.accepted,
    }, indent=2))
    return 0


def cmd_run_campaign(args: argparse.Namespace) -> int:
    manifest, catalog = _load_manifest_and_catalog(args.manifest)
    env = _detect_environment(manifest, args.config)

    # ARC-45: run_campaign_preflight() ya no se corre solo a mano por fuera del
    # CLI -- una falla bloqueante aquí detiene la campaña antes de calibrar o
    # tocar sysfs/perf, en vez de descubrirse a mitad de una corrida real.
    profile = node_profile_module.build_node_profile(
        env, manifest.cores.delegated_cpus, node_id=args.node_id, hostname=args.hostname or "",
    )
    preflight_results = preflight_module.run_campaign_preflight(manifest, env, catalog, node_profile=profile)
    blocking_failures = [result for result in preflight_results if not result.passed and result.blocking]
    if blocking_failures:
        print(json.dumps({
            "preflight_passed": False,
            "blocking_failures": [
                {"factor_id": result.factor_id, "name": result.name, "message": result.message}
                for result in blocking_failures
            ],
        }, indent=2))
        return 1

    result = campaign_module.run_campaign(
        manifest, catalog, env, node_id=args.node_id, reference_kernel_ref=args.reference_kernel_ref,
        hostname=args.hostname or "", campaign_timeout_seconds=args.campaign_timeout_seconds,
    )

    # ARC-142: run_campaign() no lanza excepción cuando una combinación es
    # rechazada (E06/E08/G01/D-checks) -- eso es un veredicto normal, no un
    # fallo del proceso. Sin este chequeo, un CI/script que solo mira el
    # exit code no puede distinguir "126/126 aceptadas" de "0/126
    # aceptadas, 126 rechazadas" -- ambos salían con 0. Tampoco distingue
    # una matriz recortada a mitad de camino (p. ej. interrumpida antes de
    # llegar a CampaignTimeoutError) de una matriz completa.
    processed = (
        len(result.progress.accepted_run_ids)
        + len(result.progress.rejected_run_ids)
        + len(result.progress.skipped_run_ids)
    )
    matrix_incomplete = processed < len(result.progress.run_ids_in_order)
    has_rejected = len(result.progress.rejected_run_ids) > 0
    restoration_failed = not result.progress.frequency_restored_verified

    print(json.dumps({
        "accepted": len(result.progress.accepted_run_ids),
        "rejected": len(result.progress.rejected_run_ids),
        "skipped": len(result.progress.skipped_run_ids),
        "total_core_hours": result.progress.total_core_hours,
        "frequency_restored_verified": result.progress.frequency_restored_verified,
        "matrix_incomplete": matrix_incomplete,
    }, indent=2))
    return 1 if (has_rejected or matrix_incomplete or restoration_failed) else 0


def cmd_postprocess(args: argparse.Namespace) -> int:
    manifest, catalog = _load_manifest_and_catalog(args.manifest)
    entry = catalog[args.kernel_ref]

    path = postprocess_module.run_postprocess(
        args.run_dir, run_id=args.run_id, repetition=args.repetition, kernel_ref=args.kernel_ref,
        kernel_entry=entry, node_id=args.node_id, freq_level_id=args.freq_level_id,
        calibration_dir=args.calibration_dir or str(manifest.output_dir),
        warmup_seconds=entry.warmup_seconds or 0.0, running_ratio_min=manifest.running_ratio_min,
        rapl_enabled=bool(manifest.rapl.get("enabled", False)),
    )
    print(path)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    campaign_dir = Path(args.campaign_dir)
    campaign_metadata = json.loads((campaign_dir / "campaign_metadata.json").read_text(encoding="utf-8"))

    # ARC-142: skipped_run_ids (MET-06) son corridas aceptadas en una sesión
    # ANTERIOR a la última invocación de run_campaign() en este output_dir
    # -- accepted_run_ids/rejected_run_ids de campaign_metadata.json solo
    # reflejan lo medido en la última sesión. Sin sumar skipped_run_ids
    # aquí, el reporte de una campaña reanudada subcuenta total_runs/
    # factor_table (silenciosamente, no falla ni avisa).
    all_run_ids = (
        campaign_metadata["accepted_run_ids"]
        + campaign_metadata["rejected_run_ids"]
        + campaign_metadata.get("skipped_run_ids", [])
    )
    verdicts = []
    for run_id in all_run_ids:
        verdict_path = campaign_dir / run_id / "verdict.json"
        if verdict_path.exists():
            verdicts.append(validation_module.load_verdict(campaign_dir / run_id))

    calibration_references = None
    if (campaign_dir / "calibration_references.json").exists():
        calibration_references = calibration_module.load_calibration_references(campaign_dir)

    data = report_module.build_report(
        campaign_id=campaign_metadata["campaign_id"], verdicts=verdicts,
        calibration_references=calibration_references,
        total_core_hours=campaign_metadata.get("total_core_hours", 0.0),
        overhead_pct_values=campaign_metadata.get("overhead_pct_values", []),
    )
    path = report_module.write_report(data, campaign_dir)
    print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.cli",
        description="Orquestador de campañas de telemetría (Fase 1): diagnose, calibrate, run-campaign, postprocess, report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose", help="Diagnóstico de arranque de solo lectura.")
    diagnose.add_argument("--manifest", required=True)
    diagnose.add_argument("--output-dir", required=True)
    diagnose.add_argument("--config")
    diagnose.add_argument("--use-allowed-cpus", action="store_true")
    diagnose.set_defaults(func=cmd_diagnose)

    calibrate = subparsers.add_parser("calibrate", help="Calibración Roofline + node_profile + calibration_references.")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--config")
    calibrate.add_argument("--node-id", required=True)
    calibrate.add_argument("--hostname")
    calibrate.add_argument("--reference-kernel-ref", required=True)
    calibrate.set_defaults(func=cmd_calibrate)

    run_campaign = subparsers.add_parser("run-campaign", help="Corre la campaña completa (calibración + matriz).")
    run_campaign.add_argument("--manifest", required=True)
    run_campaign.add_argument("--config")
    run_campaign.add_argument("--node-id", required=True)
    run_campaign.add_argument("--hostname")
    run_campaign.add_argument("--reference-kernel-ref", required=True)
    run_campaign.add_argument("--campaign-timeout-seconds", type=float, default=None)
    run_campaign.set_defaults(func=cmd_run_campaign)

    postprocess = subparsers.add_parser("postprocess", help="samples.csv -> windows.csv de una corrida ya ejecutada.")
    postprocess.add_argument("--manifest", required=True)
    postprocess.add_argument("--run-dir", required=True)
    postprocess.add_argument("--run-id", required=True)
    postprocess.add_argument("--repetition", type=int, required=True)
    postprocess.add_argument("--kernel-ref", required=True)
    postprocess.add_argument("--node-id", required=True)
    postprocess.add_argument("--freq-level-id", required=True)
    postprocess.add_argument("--calibration-dir")
    postprocess.set_defaults(func=cmd_postprocess)

    report = subparsers.add_parser("report", help="Reporte de campaña (tabla por factor_id + advertencia D04).")
    report.add_argument("--campaign-dir", required=True)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
