from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

from . import calibration as calibration_module
from . import freqctl as freqctl_module
from . import node_profile as node_profile_module
from . import postprocess as postprocess_module
from . import preflight as preflight_module
from . import runner as runner_module
from . import validation as validation_module
from .manifest import Combination

logger = logging.getLogger(__name__)

# CAM-XX ids refer to docs/retoma/Guia_Maestra_Fase1_DVFS.md section 12.8.


class CampaignTimeoutError(RuntimeError):
    """CAM-06: the campaign exceeded its overall wall-clock budget."""


@dataclass(frozen=True)
class ScheduledRun:
    """One entry of the randomized execution order."""

    combination: Combination
    mode: str  # "baseline" | "telemetry"


@dataclass
class CampaignProgress:
    """Mutable, written to disk incrementally so an interrupted campaign
    still leaves a readable trace (CAM-02) instead of losing everything."""

    run_ids_in_order: list[str] = field(default_factory=list)
    accepted_run_ids: list[str] = field(default_factory=list)
    rejected_run_ids: list[str] = field(default_factory=list)
    # MET-06: the resumed/skipped run_ids are a distinct bucket from
    # freshly-accepted ones, not folded into accepted_run_ids.
    skipped_run_ids: list[str] = field(default_factory=list)
    total_core_hours: float = 0.0
    # MET-02: set from the boolean restore_original_state() itself returns
    # (a post-restore sysfs re-read, never "the write command didn't error").
    frequency_restored_verified: bool | None = None
    # CAM-08: (telemetry.elapsed_seconds - baseline.elapsed_seconds) /
    # baseline.elapsed_seconds * 100, one entry per baseline+telemetry pair
    # actually executed this run (skipped/resumed pairs don't add one).
    overhead_pct_values: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class CampaignResult:
    progress: CampaignProgress
    roofline_calibration: Any
    node_profile: Any
    calibration_references: Any


def build_matrix(manifest: Any, *, seed: int | None = None) -> list[Combination]:
    """CAM-01: a single flat shuffle across kernel x freq_level x repetition.
    Never randomized in blocks per kernel or per frequency -- that would
    reintroduce exactly the confound (thermal/adjacent-run drift) this
    randomization exists to break.
    """
    combinations = [
        Combination(kernel_ref, freq_level, repetition)
        for kernel_ref in manifest.kernels
        for freq_level in manifest.frequency_levels
        for repetition in range(1, manifest.repetitions_per_combination + 1)
    ]
    rng = random.Random(seed if seed is not None else manifest.seed)
    rng.shuffle(combinations)
    return combinations


def schedule_runs(combinations: Sequence[Combination]) -> list[ScheduledRun]:
    """CAM-04: baseline and telemetry form an atomic pair. Each combination
    becomes two adjacent entries (baseline immediately before telemetry) so
    the randomized order over combinations never separates them.
    """
    scheduled: list[ScheduledRun] = []
    for combination in combinations:
        scheduled.append(ScheduledRun(combination, "baseline"))
        scheduled.append(ScheduledRun(combination, "telemetry"))
    return scheduled


def _run_id_for(manifest: Any, scheduled: ScheduledRun) -> str:
    base = runner_module.build_run_id(
        manifest.campaign_id,
        scheduled.combination.kernel_ref,
        scheduled.combination.frequency_level.id,
        scheduled.combination.repetition_index,
    )
    return f"{base}__baseline" if scheduled.mode == "baseline" else base


def _previous_verdict(output_dir: str | Path, run_id: str) -> validation_module.Verdict | None:
    run_dir = Path(output_dir) / run_id
    if not (run_dir / "verdict.json").exists():
        return None
    return validation_module.load_verdict(run_dir)


def write_campaign_metadata(progress: CampaignProgress, manifest: Any, output_dir: str | Path) -> Path:
    """CAM-02/MET-06: seed and the full executed run_id order in campaign
    metadata, written incrementally so a crash mid-campaign still leaves the
    order-so-far on disk."""
    path = Path(output_dir) / "campaign_metadata.json"
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "campaign_id": manifest.campaign_id,
                "seed": manifest.seed,
                "run_ids_in_order": progress.run_ids_in_order,
                "accepted_run_ids": progress.accepted_run_ids,
                "rejected_run_ids": progress.rejected_run_ids,
                "skipped_run_ids": progress.skipped_run_ids,
                "total_core_hours": progress.total_core_hours,
                "frequency_restored_verified": progress.frequency_restored_verified,
                "overhead_pct_values": progress.overhead_pct_values,
            },
            metadata_file, indent=2, sort_keys=True,
        )
        metadata_file.write("\n")
    return path


def run_campaign(
    manifest: Any,
    catalog: Mapping[str, Any],
    environment_profile: Any,
    *,
    node_id: str,
    reference_kernel_ref: str,
    hostname: str = "",
    campaign_timeout_seconds: float | None = None,
    run_single: Callable[..., Any] = runner_module.run_single,
    apply_frequency: Callable[..., Any] = freqctl_module.apply_frequency,
    read_observed_frequency_khz: Callable[..., Any] = freqctl_module.read_observed_frequency_khz,
    snapshot_original_state: Callable[..., Any] = freqctl_module.snapshot_original_state,
    restore_original_state: Callable[..., Any] = freqctl_module.restore_original_state,
    install_emergency_handlers: Callable[..., Any] = freqctl_module.install_emergency_handlers,
    run_calibration: Callable[..., Any] = calibration_module.run_calibration,
    build_node_profile: Callable[..., Any] = node_profile_module.build_node_profile,
    write_node_profile: Callable[..., Any] = node_profile_module.write_node_profile,
    run_calibration_references: Callable[..., Any] = calibration_module.run_calibration_references,
    run_postprocess: Callable[..., Any] = postprocess_module.run_postprocess,
    detect_foreign_affinity_pids: Callable[..., Any] = preflight_module.detect_foreign_affinity_pids,
) -> CampaignResult:
    """Orchestrates one full campaign, in order: snapshot original frequency
    state -> Roofline calibration -> node_profile -> calibration_references
    (CAL-11, all three before the matrix) -> the shuffled, atomic-pair matrix
    -> postprocess + validate each accepted telemetry run -> always restore
    frequency state (CAM-07), on both the happy path and any exception.

    Every optional callable defaults to the real module function; tests
    inject fakes instead of monkeypatching module internals.
    """
    delegated_cpus = manifest.cores.delegated_cpus

    # FRQ-01/CAM-07: exactly one snapshot for the whole campaign, and both
    # the emergency handlers (crash/SIGINT/SIGTERM) and the normal-exit path
    # below restore from this same snapshot.
    original_state = snapshot_original_state(delegated_cpus, environment_profile)
    install_emergency_handlers(lambda: restore_original_state(original_state, environment_profile))

    start_time = time.monotonic()
    progress = CampaignProgress()

    try:
        roofline = run_calibration(
            manifest, catalog, environment_profile=environment_profile, node_id=node_id, run_single=run_single,
        )
        # VAL-05: in practice run_calibration() above already raised
        # CalibrationError before returning anything when D03 failed
        # (CAL-04); this call is the explicit, single-place gate so the
        # invariant is asserted, not just assumed.
        calibration_verdict = validation_module.validate_campaign_calibration(roofline)
        assert calibration_verdict.accepted, "unreachable: run_calibration() must have raised on D03 failure"

        profile = build_node_profile(environment_profile, delegated_cpus, node_id=node_id, hostname=hostname)
        write_node_profile(profile, manifest.output_dir)

        reference_entry = catalog[reference_kernel_ref]
        references = run_calibration_references(
            reference_entry, manifest, reference_kernel_ref, node_id=node_id,
            environment_profile=environment_profile, run_single=run_single,
        )

        # MET-07: every run's own metadata.json carries the same calibration
        # references windows.csv rows do, not just the windows themselves.
        calibration_refs = {
            "roofline_calibration_ref": str(Path(manifest.output_dir) / "roofline_calibration.json"),
            "node_profile_ref": str(Path(manifest.output_dir) / "node_profile.json"),
            "calibration_ref": str(Path(manifest.output_dir) / "calibration_references.json"),
        }

        combinations = build_matrix(manifest, seed=manifest.seed)
        progress.run_ids_in_order = [
            runner_module.build_run_id(
                manifest.campaign_id, c.kernel_ref, c.frequency_level.id, c.repetition_index
            )
            for c in combinations
        ]
        write_campaign_metadata(progress, manifest, manifest.output_dir)

        baseline_manifest = replace(manifest, perf_enabled=False)
        seen_run_ids: set[str] = set()

        for combination in combinations:
            if campaign_timeout_seconds is not None and time.monotonic() - start_time > campaign_timeout_seconds:
                # CAM-06: never hang indefinitely across the whole matrix,
                # on top of runner.py's own per-run timeout (RUN-03).
                raise CampaignTimeoutError(
                    f"CAM-06: la campaña excedió campaign_timeout_seconds={campaign_timeout_seconds}"
                )

            entry = catalog[combination.kernel_ref]
            telemetry_run_id = runner_module.build_run_id(
                manifest.campaign_id, combination.kernel_ref,
                combination.frequency_level.id, combination.repetition_index,
            )

            # CAM-03: accepted -> skip the whole pair (already done, baseline
            # included since it was already run alongside); rejected or never
            # run -> (re)try both. A rejection is not the same as done.
            previous = _previous_verdict(manifest.output_dir, telemetry_run_id)
            if previous is not None and previous.accepted:
                seen_run_ids.add(telemetry_run_id)
                if telemetry_run_id not in progress.skipped_run_ids:
                    progress.skipped_run_ids.append(telemetry_run_id)  # MET-06
                continue

            # PRE-E06: verificar CADA VEZ, justo antes de medir, que no haya
            # procesos ajenos con afinidad a delegated_cpus -- por Cpus_allowed
            # real de /proc, nunca por membresía de cgroup. PID+inherit ya
            # garantiza que perf atribuye las muestras al proceso correcto;
            # esto cubre un problema físico distinto (contención de L3/ancho
            # de banda de memoria con otro proceso en los mismos cores) que
            # la atribución correcta no puede detectar ni evitar.
            foreign_pids = detect_foreign_affinity_pids(
                delegated_cpus, own_pids=(os.getpid(),)
            )
            foreign_check = preflight_module.check_foreign_processes(foreign_pids)
            if not foreign_check.passed:
                logger.warning(
                    "E06: procesos ajenos con afinidad a delegated_cpus, se salta la combinación (run_id=%s, foreign_pids=%s)",
                    telemetry_run_id, foreign_pids,
                )
                run_dir = Path(manifest.output_dir) / telemetry_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                validation_module.write_verdict(
                    validation_module.Verdict(
                        accepted=False, factor_id="E06",
                        message=f"Procesos ajenos con afinidad a delegated_cpus: {foreign_pids}",
                    ),
                    run_dir,
                )
                progress.rejected_run_ids.append(telemetry_run_id)
                seen_run_ids.add(telemetry_run_id)
                write_campaign_metadata(progress, manifest, manifest.output_dir)  # CAM-02
                continue

            baseline_elapsed_seconds: float | None = None
            for item in schedule_runs([combination]):  # CAM-04: atomic baseline+telemetry pair
                run_id = _run_id_for(manifest, item)
                active_manifest = manifest if item.mode == "telemetry" else baseline_manifest
                result = run_single(
                    entry, active_manifest, item.combination.kernel_ref,
                    item.combination.frequency_level.id, item.combination.repetition_index,
                    environment_profile=environment_profile, node_id=node_id, apply_frequency=apply_frequency,
                    calibration_refs=calibration_refs,
                )
                progress.total_core_hours += result.elapsed_seconds * len(delegated_cpus) / 3600.0  # CAM-05/OPS-01

                if item.mode == "baseline":
                    baseline_elapsed_seconds = result.elapsed_seconds
                    continue  # solo mide overhead; no se valida ni se postprocesa

                # CAM-08: overhead de instrumentacion = cuanto mas lenta corre
                # telemetry frente a su gemelo baseline (mismo kernel/nivel/
                # repeticion, --no-perf). Solo se calcula si el baseline de
                # ESTE par realmente corrio (nunca contra un baseline viejo).
                if baseline_elapsed_seconds is not None and baseline_elapsed_seconds > 0:
                    overhead_pct = (
                        (result.elapsed_seconds - baseline_elapsed_seconds) / baseline_elapsed_seconds * 100.0
                    )
                    progress.overhead_pct_values.append(overhead_pct)

                verdict = validation_module.validate_run(result, entry, run_id_seen=seen_run_ids)
                validation_module.write_verdict(verdict, result.run_dir)
                seen_run_ids.add(run_id)

                if verdict.accepted:
                    progress.accepted_run_ids.append(run_id)
                    # FRQ-03/FRQ-10: whatever this run actually requested/
                    # applied (None when apply_frequency was never invoked,
                    # e.g. frequency_write_capable=False) plus the observed
                    # scaling_cur_freq right after the run, never dropped
                    # between freqctl and windows.csv.
                    applied = result.applied_frequency
                    freq_khz_observed = read_observed_frequency_khz(environment_profile, delegated_cpus[0])
                    run_postprocess(
                        result.run_dir, run_id=run_id, repetition=item.combination.repetition_index,
                        kernel_ref=item.combination.kernel_ref, kernel_entry=entry, node_id=node_id,
                        freq_level_id=item.combination.frequency_level.id, calibration_dir=manifest.output_dir,
                        freq_khz_requested=getattr(applied, "requested_khz", None),
                        freq_khz_applied=getattr(applied, "applied_khz", None),
                        freq_khz_observed=freq_khz_observed,
                        warmup_seconds=entry.warmup_seconds or 0.0, running_ratio_min=manifest.running_ratio_min,
                        rapl_enabled=bool(manifest.rapl.get("enabled", False)), calibration_references=references,
                    )
                else:
                    progress.rejected_run_ids.append(run_id)

            write_campaign_metadata(progress, manifest, manifest.output_dir)

        return CampaignResult(
            progress=progress, roofline_calibration=roofline, node_profile=profile,
            calibration_references=references,
        )
    finally:
        # CAM-07/MET-02: always restore, normal close or interruption, and
        # keep the boolean restore_original_state() itself returns (a
        # post-restore sysfs re-read, never "the write command didn't
        # error"). In "unavailable" strategy this verifies there was
        # nothing to restore (freqctl.restore_original_state handles that
        # branch itself).
        progress.frequency_restored_verified = bool(restore_original_state(original_state, environment_profile))
        write_campaign_metadata(progress, manifest, manifest.output_dir)
