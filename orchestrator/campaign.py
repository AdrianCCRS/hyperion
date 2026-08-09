from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

from . import calibration as calibration_module
from . import freqctl as freqctl_module
from . import gpu_freqctl as gpu_freqctl_module
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


class CampaignProtocolMismatchError(RuntimeError):
    """CAM-09 (ARC-94): the manifest/catalog protocol for this output_dir
    changed since a previous run left accepted verdicts here. Resuming
    under a mismatched protocol must fail closed, never mix runs silently
    -- confirmed to have already happened for real: pacca_gpu_ref_20260807
    contains both gpu_interval_ns=100ms and 5ms runs because nothing ever
    compared the manifest across a resume, and old verdicts stayed
    "accepted" after the manifest moved on to a different cadence."""


def compute_protocol_fingerprint(manifest: Any, catalog: Mapping[str, Any]) -> str:
    """CAM-09 (ARC-94): a hash of every manifest/catalog field that affects
    what a run actually measures -- sampling cadence, frequency levels,
    core pinning, SMT policy, and per-kernel exec args/checksum/warmup for
    every kernel_ref this manifest references (calibration + kernels).
    Two manifests that would produce different samples.csv/windows.csv for
    the same run_id must never share this fingerprint.
    """
    gpu = getattr(manifest, "gpu", {}) or {}
    rapl = getattr(manifest, "rapl", {}) or {}
    # ARC-94 (segunda ronda): confirmado que el fingerprint original no
    # cambiaba si se editaba gpu.enabled, gpu.calibration, rapl.enabled,
    # perf_enabled, o la intensidad operacional/precisión de un kernel GPU
    # -- los cinco afectan directamente qué mide una corrida o cómo se
    # etiqueta (phase_label_train), así que dos manifiestos que difieran
    # solo en uno de ellos NO deben compartir fingerprint.
    references = tuple(sorted(
        set(manifest.calibration) | set(manifest.kernels) | set(gpu.get("calibration", ()) or ())
    ))
    kernel_fingerprint = [
        {
            "kernel_ref": ref,
            "exec_path": str(getattr(catalog[ref], "exec_path", "")),
            "exec_args": getattr(catalog[ref], "exec_args", None),
            "binary_checksum": getattr(catalog[ref], "binary_checksum", None),
            "warmup_seconds": getattr(catalog[ref], "warmup_seconds", None),
            "success_check": getattr(catalog[ref], "success_check", None),
            "operational_intensity_flops_per_byte": getattr(catalog[ref], "operational_intensity_flops_per_byte", None),
            "gpu_precision": getattr(catalog[ref], "gpu_precision", None),
        }
        for ref in references
        if ref in catalog
    ]
    cores = manifest.cores
    payload = {
        "interval_ns": manifest.interval_ns,
        "gpu_interval_ns": getattr(manifest, "gpu_interval_ns", None),
        "running_ratio_min": manifest.running_ratio_min,
        "target_windows_per_repetition": manifest.target_windows_per_repetition,
        "repetitions_per_combination": manifest.repetitions_per_combination,
        "smt_policy": manifest.smt_policy,
        "perf_enabled": manifest.perf_enabled,
        "rapl": {"enabled": rapl.get("enabled"), "domains": sorted(rapl.get("domains", ()) or ())},
        "gpu": {
            "enabled": gpu.get("enabled"),
            "calibration": sorted(gpu.get("calibration", ()) or ()),
        },
        "cores": {
            "delegated_cpus": list(cores.delegated_cpus),
            "collector_cpu": cores.collector_cpu,
            "consumer_cpu": cores.consumer_cpu,
            "numa_node_pin": cores.numa_node_pin,
        },
        "frequency_levels": [
            {"id": level.id, "mode": level.mode, "fraction": getattr(level, "fraction", None)}
            for level in manifest.frequency_levels
        ],
        "kernels": kernel_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _has_existing_runs(output_dir: Path) -> bool:
    """CAM-09: true if `output_dir` already contains at least one run
    subdirectory with a verdict.json -- i.e., this directory has measured
    something before, regardless of whether protocol_fingerprint.json
    exists. Used to tell "genuinely fresh output_dir" apart from "a legacy
    directory that predates this fingerprinting mechanism," which must
    never be silently adopted under whatever protocol happens to run next.
    """
    if not output_dir.is_dir():
        return False
    return any((entry / "verdict.json").exists() for entry in output_dir.iterdir() if entry.is_dir())


def _check_protocol_fingerprint(manifest: Any, catalog: Mapping[str, Any]) -> None:
    """CAM-09: writes protocol_fingerprint.json on the first run in a fresh
    output_dir; on any later invocation (a resume, by definition, since
    accepted verdicts can only exist after a first successful pass),
    raises CampaignProtocolMismatchError instead of silently proceeding if
    the manifest/catalog protocol no longer matches what produced the
    verdicts already on disk.

    ARC-94 (segunda ronda): un output_dir que YA tiene corridas (verdict.json
    en al menos un run_dir) pero NUNCA tuvo protocol_fingerprint.json --
    es decir, una carpeta de campaña real anterior a este mecanismo, como
    las dos que existen hoy en paccaA100 -- ya no se "adopta" en silencio
    escribiendo el fingerprint actual como si fuera la primera corrida.
    Eso habría dejado exactamente el escenario que este chequeo existe para
    prevenir: corridas viejas de un protocolo desconocido conviviendo con
    corridas nuevas bajo un fingerprint que nunca las describió. Una
    carpeta así debe resolverse a mano (confirmar el protocolo real de las
    corridas existentes y escribir el fingerprint explícitamente, o mudarse
    a un output_dir nuevo).
    """
    output_dir = Path(manifest.output_dir)
    fingerprint_path = output_dir / "protocol_fingerprint.json"
    current = compute_protocol_fingerprint(manifest, catalog)
    if fingerprint_path.exists():
        previous = json.loads(fingerprint_path.read_text()).get("sha256")
        if previous != current:
            raise CampaignProtocolMismatchError(
                f"CAM-09: el protocolo de medición de {output_dir} cambió desde la última "
                f"corrida (fingerprint anterior={previous!r}, actual={current!r}). Reanudar "
                "aquí mezclaría corridas de dos protocolos distintos bajo el mismo "
                "campaign_id/output_dir. Usa un output_dir/campaign_id nuevo para el "
                "protocolo nuevo, o revierte el cambio de manifiesto/catálogo."
            )
        return
    if _has_existing_runs(output_dir):
        raise CampaignProtocolMismatchError(
            f"CAM-09: {output_dir} ya contiene corridas (verdict.json) pero nunca tuvo "
            "protocol_fingerprint.json -- es una carpeta de campaña anterior a este "
            "mecanismo (o de un origen no controlado). No se adopta en silencio: "
            "confirma a mano el protocolo real de las corridas existentes y escribe "
            "protocol_fingerprint.json explícitamente, o usa un output_dir nuevo."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with fingerprint_path.open("w", encoding="utf-8") as fingerprint_file:
        json.dump({"sha256": current}, fingerprint_file, indent=2, sort_keys=True)
        fingerprint_file.write("\n")


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
    # ARC-80: {"fp32": RooflineCalibration, "fp64": RooflineCalibration} para
    # el nivel de referencia, o {} si manifest.gpu no declara "calibration"
    # (campañas sin kernels de GPU). Ver calibration.run_gpu_calibration.
    gpu_roofline_calibration: Any = None


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


def _archive_rejected_run(output_dir: Path, run_id: str) -> None:
    """CAM-10 (ARC-94): retrying a rejected combination reuses the same
    deterministic run_id (CAM-03: "rejected or never run -> (re)try both").
    Without this, the retry's samples.csv/stdout.txt/metadata.json/
    verdict.json overwrite the rejected run's own directory in place --
    contradicting VAL-06's explicit promise ("rejected runs are NEVER
    deleted... this only ever adds a verdict.json"). Moves the existing
    directory aside (suffixed with the next free ``__rejectedN``) so the
    retry always starts in a clean, previously-unused directory.
    """
    run_dir = output_dir / run_id
    if not run_dir.exists():
        return
    index = 1
    while (output_dir / f"{run_id}__rejected{index}").exists():
        index += 1
    archived = output_dir / f"{run_id}__rejected{index}"
    run_dir.rename(archived)
    logger.info("CAM-10: corrida rechazada archivada antes de reintentar: %s -> %s", run_dir, archived)


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
    # ARC-87: eje de GPU, paralelo al de CPU justo arriba. apply_gpu_frequency
    # y restore_gpu_state ya son no-ops seguros (gpu_freqctl.STRATEGY_
    # UNAVAILABLE / restore_gpu_state devuelve True sin tocar nada) cuando
    # environment_profile.gpu_frequency_write_capable es falso -- una
    # campaña sin GPU, o corriendo antes de que llegue el permiso P4, no
    # cambia de comportamiento por tener este wiring presente.
    apply_gpu_frequency: Callable[..., Any] = gpu_freqctl_module.apply_gpu_frequency,
    restore_gpu_state: Callable[..., Any] = gpu_freqctl_module.restore_gpu_state,
    run_calibration: Callable[..., Any] = calibration_module.run_calibration,
    run_gpu_calibration: Callable[..., Any] = calibration_module.run_gpu_calibration,
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

    # CAM-09 (ARC-94): fail closed, before touching any hardware state, if
    # this output_dir already has accepted verdicts from a different
    # measurement protocol (see CampaignProtocolMismatchError).
    _check_protocol_fingerprint(manifest, catalog)

    # FRQ-01/CAM-07: exactly one snapshot for the whole campaign, and both
    # the emergency handlers (crash/SIGINT/SIGTERM) and the normal-exit path
    # below restore from this same snapshot.
    original_state = snapshot_original_state(delegated_cpus, environment_profile)

    # ARC-87: install_emergency_handlers() replaces the SIGINT/SIGTERM
    # handler outright (signal.signal), so calling it a second time for GPU
    # would silently drop the CPU restore from the signal path (atexit would
    # still run both, but a Ctrl-C wouldn't). Both restores are combined into
    # one closure and registered once instead. restore_gpu_state() is a
    # verified no-op when gpu_frequency_write_capable is False, so this is
    # safe to call unconditionally, exactly like restore_original_state()
    # already is for the "unavailable" CPU strategy.
    def _restore_all() -> bool:
        cpu_ok = restore_original_state(original_state, environment_profile)
        gpu_ok = restore_gpu_state(environment_profile)
        return bool(cpu_ok) and bool(gpu_ok)

    install_emergency_handlers(_restore_all)

    # ARC-78: freqctl.apply_frequency(cpus, level, env, *, original=...)
    # necesita el snapshot original para el modo native_governor (restaurar
    # el governor que tenía el CPU antes de la campaña) -- se liga aquí, una
    # sola vez, para que run_single/run_calibration sigan invocando
    # apply_frequency con el mismo contrato de 3 argumentos que ya usan
    # (cpus, level, env), sin tener que conocer original_state.
    bound_apply_frequency = functools.partial(apply_frequency, original=original_state)

    start_time = time.monotonic()
    progress = CampaignProgress()

    try:
        roofline = run_calibration(
            manifest, catalog, environment_profile=environment_profile, node_id=node_id, run_single=run_single,
            apply_frequency=bound_apply_frequency,
        )
        # VAL-05: in practice run_calibration() above already raised
        # CalibrationError before returning anything when D03 failed
        # (CAL-04); this call is the explicit, single-place gate so the
        # invariant is asserted, not just assumed.
        calibration_verdict = validation_module.validate_campaign_calibration(roofline)
        assert calibration_verdict.accepted, "unreachable: run_calibration() must have raised on D03 failure"

        # ARC-80: infraestructura separada de la de CPU -- devuelve {} sin
        # tocar nada cuando manifest.gpu no declara "calibration" (campañas
        # sin kernels de GPU), así que no hace falta gatear esta llamada por
        # si el manifiesto tiene GPU o no.
        gpu_roofline = run_gpu_calibration(
            manifest, catalog, environment_profile=environment_profile, node_id=node_id, run_single=run_single,
            apply_frequency=bound_apply_frequency, apply_gpu_frequency=apply_gpu_frequency,
        )

        profile = build_node_profile(environment_profile, delegated_cpus, node_id=node_id, hostname=hostname)
        write_node_profile(profile, manifest.output_dir)

        reference_entry = catalog[reference_kernel_ref]
        references = run_calibration_references(
            reference_entry, manifest, reference_kernel_ref, node_id=node_id,
            environment_profile=environment_profile, run_single=run_single,
            # ARC-94: sin esto, las referencias se medían bajo el último
            # nivel fixed que run_calibration()/run_gpu_calibration() dejó
            # aplicado (típicamente F4), no a frecuencia nativa.
            apply_frequency=bound_apply_frequency,
        )

        # MET-07: every run's own metadata.json carries the same calibration
        # references windows.csv rows do, not just the windows themselves.
        # ARC-78: roofline_calibration_ref se arma por combinación, no una
        # sola vez aquí -- cada nivel de frecuencia tiene su propio archivo
        # (calibration_module.calibration_filename), y una corrida a REF
        # nunca debe apuntar al i_ridge calibrado a otro nivel.
        shared_calibration_refs = {
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
            if previous is not None and not previous.accepted:
                # CAM-10: a genuine retry, not a fresh first attempt --
                # archive the rejected run's evidence (and its baseline
                # sibling, if any) before writing into the same run_id again.
                _archive_rejected_run(Path(manifest.output_dir), telemetry_run_id)
                _archive_rejected_run(Path(manifest.output_dir), f"{telemetry_run_id}__baseline")

            # PRE-E06: verificar CADA VEZ, justo antes de medir, que no haya
            # procesos ajenos CORRIENDO AHORA MISMO en delegated_cpus -- por
            # el campo "processor" real de /proc/<pid>/stat, nunca por
            # membresía de cgroup ni por Cpus_allowed (ver ARC-44: casi todo
            # proceso del sistema en reposo tiene Cpus_allowed sin
            # restringir, así que filtrar solo por esa máscara marca como
            # "ajeno" a decenas de daemons inactivos que no compiten por
            # nada). PID+inherit ya garantiza que perf atribuye las muestras
            # al proceso correcto; esto cubre un problema físico distinto
            # (contención de L3/ancho de banda de memoria con otro proceso
            # activo en los mismos cores) que la atribución correcta no
            # puede detectar ni evitar.
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

            combination_calibration_refs = {
                **shared_calibration_refs,
                "roofline_calibration_ref": str(
                    Path(manifest.output_dir)
                    / calibration_module.calibration_filename(combination.frequency_level.id)
                ),
            }

            baseline_elapsed_seconds: float | None = None
            for item in schedule_runs([combination]):  # CAM-04: atomic baseline+telemetry pair
                run_id = _run_id_for(manifest, item)
                active_manifest = manifest if item.mode == "telemetry" else baseline_manifest
                result = run_single(
                    entry, active_manifest, item.combination.kernel_ref,
                    item.combination.frequency_level.id, item.combination.repetition_index,
                    environment_profile=environment_profile, node_id=node_id,
                    apply_frequency=bound_apply_frequency,
                    # ARC-87: sin functools.partial -- a diferencia de
                    # freqctl.apply_frequency, apply_gpu_frequency(level, env)
                    # no necesita un snapshot "original" (no hay estado de GPU
                    # que preservar más allá de "sin reloj fijado", ver
                    # gpu_freqctl.py); run_single ya gatea la llamada por
                    # entry.device=="gpu" internamente, así que las corridas
                    # de kernels CPU nunca la invocan.
                    apply_gpu_frequency=apply_gpu_frequency,
                    calibration_refs=combination_calibration_refs,
                    # ARC-94: sin esto, run_single() reconstruía su propio
                    # run_id "plano" (sin el sufijo __baseline que
                    # _run_id_for ya calculó arriba) -- baseline y telemetry
                    # del mismo combo escribían en el MISMO run_dir, y el
                    # segundo pisaba los artefactos del primero.
                    run_id=run_id,
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

                # ARC-94: validate_run() es solo la PRIMERA etapa -- solo
                # puede rechazar por metadata que existe antes de
                # windows.csv (I04/C02/C03/E06-E08/I07). Un veredicto
                # aceptado aquí es provisional: la aceptación final
                # depende de cuántas ventanas usables y qué etiqueta
                # produjo postprocess.py, no solo de que el binario
                # terminara bien -- antes de este cambio, una corrida
                # podía quedar accepted=true con cero ventanas 'ok'/
                # 'gpu_telemetry' o ninguna etiqueta, porque
                # target_windows_per_repetition se declaraba en el
                # manifiesto pero nunca se usaba para decidir nada.
                provisional_verdict = validation_module.validate_run(
                    result, entry, run_id_seen=seen_run_ids, node_id=node_id
                )
                seen_run_ids.add(run_id)

                if provisional_verdict.accepted:
                    # FRQ-03/FRQ-10: whatever this run actually requested/
                    # applied (None when apply_frequency was never invoked,
                    # e.g. frequency_write_capable=False) plus the observed
                    # scaling_cur_freq right after the run, never dropped
                    # between freqctl and windows.csv.
                    applied = result.applied_frequency
                    freq_khz_observed = read_observed_frequency_khz(environment_profile, delegated_cpus[0])
                    windows_path = run_postprocess(
                        result.run_dir, run_id=run_id, repetition=item.combination.repetition_index,
                        kernel_ref=item.combination.kernel_ref, kernel_entry=entry, node_id=node_id,
                        freq_level_id=item.combination.frequency_level.id, calibration_dir=manifest.output_dir,
                        freq_khz_requested=getattr(applied, "requested_khz", None),
                        freq_khz_applied=getattr(applied, "applied_khz", None),
                        freq_khz_observed=freq_khz_observed,
                        warmup_seconds=entry.warmup_seconds or 0.0, running_ratio_min=manifest.running_ratio_min,
                        rapl_enabled=bool(manifest.rapl.get("enabled", False)), calibration_references=references,
                    )
                    verdict = validation_module.validate_windows(
                        windows_path,
                        target_windows_per_repetition=manifest.target_windows_per_repetition,
                        device=entry.device,
                    )
                else:
                    verdict = provisional_verdict

                validation_module.write_verdict(verdict, result.run_dir)

                if verdict.accepted:
                    progress.accepted_run_ids.append(run_id)
                else:
                    progress.rejected_run_ids.append(run_id)

            write_campaign_metadata(progress, manifest, manifest.output_dir)

        return CampaignResult(
            progress=progress, roofline_calibration=roofline, node_profile=profile,
            calibration_references=references, gpu_roofline_calibration=gpu_roofline,
        )
    finally:
        # CAM-07/MET-02: always restore, normal close or interruption, and
        # keep the boolean restore_original_state() itself returns (a
        # post-restore sysfs re-read, never "the write command didn't
        # error"). In "unavailable" strategy this verifies there was
        # nothing to restore (freqctl.restore_original_state handles that
        # branch itself). ARC-87: GPU restore folded into the same verified
        # flag -- a campaign is not "cleanly restored" if the CPU frequency
        # came back but the GPU clock stayed locked, or vice versa.
        cpu_restored = bool(restore_original_state(original_state, environment_profile))
        gpu_restored = bool(restore_gpu_state(environment_profile))
        progress.frequency_restored_verified = cpu_restored and gpu_restored
        write_campaign_metadata(progress, manifest, manifest.output_dir)
