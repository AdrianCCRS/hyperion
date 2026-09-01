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
from common.hpc import freqctl as freqctl_module
from common.hpc import gpu_freqctl as gpu_freqctl_module
from common.hpc import node_profile as node_profile_module
from . import postprocess as postprocess_module
from common.hpc import preflight as preflight_module
from . import runner as runner_module
from . import validation as validation_module
from common.hpc.config import load_config
from common.hpc.manifest import Combination

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


class CampaignPreflightError(RuntimeError):
    """ARC-102: a campaign-wide precondition (checked once, before the first
    real measurement -- e.g. E08 external load ahead of calibration) failed.
    Unlike a per-combination rejection (E06/E08 inside the matrix loop,
    which skips just that combination and continues), there is no run_id to
    reject yet at this point -- the whole campaign must not start."""


class CampaignCalibrationMissingError(RuntimeError):
    """CAM-11 (ARC-142): output_dir already has accepted runs (a resume,
    per _has_existing_runs) but is missing a calibration/profile/references
    artifact this resume needs to load. Re-measuring instead of raising
    would silently overwrite roofline_calibration_<id>.json and
    calibration_references.json -- shared files that already-accepted runs
    reference by path -- swapping the i_ridge/reference data out from under
    them, and classifying any remaining combination against a different
    calibration than the ones already accepted, with no record of the
    split. A resume with missing calibration artifacts must fail closed and
    be resolved by hand, never silently re-measure."""


def _launcher_checksum(harness: Any) -> str | None:
    """ARC-141: sha256 del binario del launcher C++ (harness.binary_path),
    mismo patrón que catalog.verify_binary() usa para los binarios de los
    kernels. None (nunca "" ni un valor inventado) cuando el archivo no
    existe o no es legible -- una campaña sin este dato disponible no debe
    fallar por eso, pero tampoco debe fingir un checksum que no verificó."""
    binary_path = getattr(harness, "binary_path", None)
    if not binary_path:
        return None
    try:
        with open(binary_path, "rb") as binary_file:
            return f"sha256:{hashlib.file_digest(binary_file, 'sha256').hexdigest()}"
    except OSError:
        return None


def compute_protocol_fingerprint(manifest: Any, catalog: Mapping[str, Any], harness: Any = None) -> str:
    """CAM-09 (ARC-94): a hash of every manifest/catalog field that affects
    what a run actually measures -- sampling cadence, frequency levels,
    core pinning, SMT policy, and per-kernel exec args/checksum/warmup for
    every kernel_ref this manifest references (calibration + kernels).
    Two manifests that would produce different samples.csv/windows.csv for
    the same run_id must never share this fingerprint.

    `harness` (ARC-141): when given, its binary's sha256 enters the
    fingerprint too -- a resumed campaign that runs against a rebuilt
    launcher (different C++ instrument, same manifest) previously went
    completely undetected, since nothing here ever looked past the Python
    orchestrator's own inputs. None (the default, and every call site
    before this change) omits it, exactly as before.
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
            # ARC-95: cambiar device (cpu<->gpu) de un kernel en el catálogo
            # cambia si se le aplica frecuencia de GPU y si su telemetría se
            # valida como CPU u GPU en validate_windows() -- sin este campo
            # el fingerprint no lo notaba.
            "device": getattr(catalog[ref], "device", None),
        }
        for ref in references
        if ref in catalog
    ]
    cores = manifest.cores
    payload = {
        # ARC-95: campaign_id entra en build_run_id() -- si cambia, TODOS
        # los run_id de esta carpeta cambian de forma (no hay colisión
        # posible con corridas viejas), así que no es un riesgo de mezcla
        # silenciosa como los demás campos. Se incluye de todas formas para
        # que dos manifiestos que solo difieran en campaign_id dentro del
        # mismo output_dir queden marcados como protocolos distintos, en
        # vez de indistinguibles.
        "campaign_id": manifest.campaign_id,
        # ARC-141: la revisión pre-vuelo del dataset final encontró que la
        # huella no cambiaba al editar la semilla, uncore, los timeouts, el
        # umbral de carga externa o la ficha técnica de calibración -- los
        # cinco pueden alterar qué se mide o cómo se clasifica sin que
        # ninguna corrida quedara marcada como de protocolo distinto:
        # `seed` decide el orden de la matriz (CAM-01), no reproducible si
        # cambia entre una reanudación y la siguiente sin que se note;
        # `uncore.enabled` decide si `operational_intensity`/
        # `phase_label_train` se calculan con bytes reales o quedan
        # indefinidos (E12/ARC-123) -- el caso más grave de los cinco, dos
        # corridas bajo el mismo run_id podrían tener criterios de
        # clasificación incompatibles; `timeouts_seconds`/`load_threshold`
        # afectan qué combinaciones se rechazan (RUN-03/E08) sin afectar el
        # contenido de una corrida aceptada, pero igual identifican un
        # protocolo distinto; `hardware_datasheet` decide si D03 acepta o
        # rechaza la calibración (plausibilidad de BW_pico/P_pico).
        "seed": manifest.seed,
        "interval_ns": manifest.interval_ns,
        "gpu_interval_ns": getattr(manifest, "gpu_interval_ns", None),
        "running_ratio_min": manifest.running_ratio_min,
        "target_windows_per_repetition": manifest.target_windows_per_repetition,
        "repetitions_per_combination": manifest.repetitions_per_combination,
        "turbo": dict(getattr(manifest, "turbo", None) or {}),
        "frequency_validation": dict(getattr(manifest, "frequency_validation", None) or {}),
        "temperature": dict(getattr(manifest, "temperature", None) or {}),
        "uncore": dict(getattr(manifest, "uncore", None) or {}),
        "load_threshold": getattr(manifest, "load_threshold", None),
        "timeouts_seconds": {
            "ready": manifest.timeouts_seconds.ready,
            "run": manifest.timeouts_seconds.run,
            "shutdown": manifest.timeouts_seconds.shutdown,
        },
        "hardware_datasheet": dict(getattr(manifest, "hardware_datasheet", None) or {}),
        "launcher_checksum": _launcher_checksum(harness),
        "smt_policy": manifest.smt_policy,
        "perf_enabled": manifest.perf_enabled,
        "rapl": {"enabled": rapl.get("enabled"), "domains": sorted(rapl.get("domains", ()) or ())},
        "gpu": {
            "enabled": gpu.get("enabled"),
            "calibration": list(gpu.get("calibration", ()) or ()),
        },
        # El orden también es parte del protocolo: con la misma semilla,
        # reordenar kernels cambia la permutación final y por tanto su
        # posición respecto a deriva térmica/temporal.
        "kernel_refs": list(manifest.kernels),
        "calibration_refs": list(manifest.calibration),
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
        "gpu_frequency_levels": [
            {"id": level.id, "mode": level.mode, "fraction": getattr(level, "fraction", None)}
            for level in (getattr(manifest, "gpu_frequency_levels", None) or ())
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


def _has_existing_setup_artifacts(output_dir: Path) -> bool:
    """CAM-11: detecta una calibración iniciada aunque la campaña se haya
    interrumpido antes de escribir el primer verdict.json.

    Esos archivos tampoco se pueden adoptar bajo una huella nueva ni
    sobrescribir parcialmente: ya son evidencia medida del protocolo.
    """
    if not output_dir.is_dir():
        return False
    if any(output_dir.glob("roofline_calibration*.json")):
        return True
    if any(
        (output_dir / filename).exists()
        for filename in ("node_profile.json", "calibration_references.json")
    ):
        return True
    # Una interrupción puede ocurrir dentro de STREAM/ERT o de la primera
    # combinación, antes de que exista un ridge o verdict.json. Sus crudos
    # dentro de un subdirectorio también son evidencia: no se adoptan ni se
    # pisan como si el output_dir estuviera vacío.
    return any(
        entry.is_dir() and any(entry.iterdir())
        for entry in output_dir.iterdir()
    )


def _has_existing_campaign_artifacts(output_dir: Path) -> bool:
    return _has_existing_runs(output_dir) or _has_existing_setup_artifacts(output_dir)


def _check_protocol_fingerprint(manifest: Any, catalog: Mapping[str, Any], harness: Any = None) -> None:
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
    current = compute_protocol_fingerprint(manifest, catalog, harness)
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
    if _has_existing_campaign_artifacts(output_dir):
        raise CampaignProtocolMismatchError(
            f"CAM-09: {output_dir} ya contiene corridas o artefactos de calibración pero nunca tuvo "
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
    # ARC-138: evidencia pre-corrida de los factores dinámicos que antes se
    # comprobaban (E08) o quedaban pendientes (E02) sin persistir el valor.
    pre_run_observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    pre_calibration_observation: dict[str, Any] = field(default_factory=dict)


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


def build_matrix(
    manifest: Any, catalog: Mapping[str, Any] | None = None, *, seed: int | None = None
) -> list[Combination]:
    """CAM-01: a single flat shuffle across kernel x freq_level x repetition.
    Never randomized in blocks per kernel or per frequency -- that would
    reintroduce exactly the confound (thermal/adjacent-run drift) this
    randomization exists to break.

    ARC-129: `catalog`, if given, enables a real cartesian CPU x GPU
    frequency sweep for kernels with `device=="gpu"` when
    `manifest.gpu_frequency_levels` is declared -- every (cpu_level,
    gpu_level) pair, not just a shared id walked once. `catalog=None` (the
    default, and every existing call site before this change) preserves the
    exact old behavior: `frequency_levels` alone, `gpu_frequency_level=None`
    on every Combination -- decoupling never activates without a catalog to
    tell a GPU kernel_ref apart from a CPU one.
    """
    combinations: list[Combination] = []
    for kernel_ref in manifest.kernels:
        device = "cpu"
        if catalog is not None and kernel_ref in catalog:
            device = getattr(catalog[kernel_ref], "device", "cpu")
        if device == "gpu" and manifest.gpu_frequency_levels:
            level_pairs = [
                (cpu_level, gpu_level)
                for cpu_level in manifest.frequency_levels
                for gpu_level in manifest.gpu_frequency_levels
            ]
        else:
            level_pairs = [(cpu_level, None) for cpu_level in manifest.frequency_levels]
        for cpu_level, gpu_level in level_pairs:
            for repetition in range(1, manifest.repetitions_per_combination + 1):
                combinations.append(Combination(kernel_ref, cpu_level, repetition, gpu_frequency_level=gpu_level))
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
    """ARC-172: a diferencia de las otras dos llamadas a build_run_id() en
    este archivo (líneas ~777 y ~800), esta nunca pasó gpu_freq_level_id --
    nunca se notó porque hasta ARC-170/171 ninguna campaña real había
    declarado manifest.gpu_frequency_levels con kernels de GPU en la
    matriz. Sin el sufijo __gpu<id>, las 6 combinaciones de un mismo
    (kernel, nivel de CPU, repetición) que solo difieren en el nivel de
    GPU colapsan sobre el MISMO run_id -- y por tanto el mismo run_dir en
    disco: cada una sobrescribe la telemetría de la anterior en vez de
    escribir en su propio directorio, confirmado en el smoke de ARC-170
    (job 6344, 6/36 aceptadas, 30/36 rechazadas, los 6 run_id "sin sufijo"
    esperados apareciendo cada uno con 1 aceptación + 5 rechazos en vez de
    las 6 combinaciones reales completamente aisladas entre sí)."""
    base = runner_module.build_run_id(
        manifest.campaign_id,
        scheduled.combination.kernel_ref,
        scheduled.combination.frequency_level.id,
        scheduled.combination.repetition_index,
        scheduled.combination.gpu_frequency_level.id
        if scheduled.combination.gpu_frequency_level is not None else None,
    )
    return f"{base}__baseline" if scheduled.mode == "baseline" else base


def _archive_run(output_dir: Path, run_id: str, *, archive_kind: str) -> None:
    """Mueve una corrida existente al siguiente sufijo de auditoría libre."""
    run_dir = output_dir / run_id
    if not run_dir.exists():
        return
    index = 1
    while (output_dir / f"{run_id}__{archive_kind}{index}").exists():
        index += 1
    archived = output_dir / f"{run_id}__{archive_kind}{index}"
    run_dir.rename(archived)
    logger.info("CAM-10: corrida %s archivada antes de reintentar: %s -> %s", archive_kind, run_dir, archived)


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
    _archive_run(output_dir, run_id, archive_kind="rejected")


def _archive_incomplete_run(output_dir: Path, run_id: str) -> None:
    """CAM-10: conserva crudos de una invocación interrumpida antes de que
    pudiera escribir verdict.json, en vez de sobrescribirlos al reintentar."""
    _archive_run(output_dir, run_id, archive_kind="incomplete")


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
                "pre_run_observations": progress.pre_run_observations,
                "pre_calibration_observation": progress.pre_calibration_observation,
            },
            metadata_file, indent=2, sort_keys=True,
        )
        metadata_file.write("\n")
    return path


def _seed_progress_from_previous_metadata(progress: CampaignProgress, output_dir: str | Path) -> None:
    """ARC-142: carries `total_core_hours`/`overhead_pct_values` forward
    from a previous campaign_metadata.json in this output_dir, if one
    exists -- see the call site's comment for why. Silently does nothing
    (fresh progress, matching behavior before this change) when the file is
    absent or unreadable; a resume with a corrupt/missing metadata file
    should still be able to proceed, it just starts the accumulation over,
    same as a genuinely fresh output_dir always did.
    """
    path = Path(output_dir) / "campaign_metadata.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    progress.total_core_hours = float(previous.get("total_core_hours", 0.0) or 0.0)
    progress.overhead_pct_values = list(previous.get("overhead_pct_values", ()) or ())


def run_campaign(
    manifest: Any,
    catalog: Mapping[str, Any],
    environment_profile: Any,
    *,
    node_id: str,
    reference_kernel_ref: str,
    hostname: str = "",
    campaign_timeout_seconds: float | None = None,
    # ARC-141: sha256 del launcher C++ entra a la huella de protocolo
    # (CAM-09) para que un instrumento reconstruido durante una reanudación
    # no pase desapercibido. None (default) resuelve harness.binary_path
    # desde orchestrator.toml, mismo criterio que runner.run_single() ya
    # usa para el mismo dato.
    harness: Any = None,
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
    # CAM-11 (ARC-142): a resume (_has_existing_runs) loads the calibration
    # already on disk through these instead of calling the run_*
    # equivalents above, which would re-measure and overwrite the shared
    # files that already-accepted runs reference by path.
    load_calibration: Callable[..., Any] = calibration_module.load_calibration,
    load_node_profile: Callable[..., Any] = node_profile_module.load_node_profile,
    load_calibration_references: Callable[..., Any] = calibration_module.load_calibration_references,
    run_postprocess: Callable[..., Any] = postprocess_module.run_postprocess,
    detect_foreign_affinity_pids: Callable[..., Any] = preflight_module.detect_foreign_affinity_pids,
    # ARC-101: preflight.run_reduced_preflight() ya implementaba este check
    # (E08, carga externa) pero nunca estuvo conectado a ningun camino de
    # produccion real -- ni campaign.py ni runner.py lo llamaban. Se agrega
    # aqui, no como llamada a run_reduced_preflight() completa, porque esa
    # funcion tambien duplica E06 (ya cubierto abajo por su cuenta) y exige
    # E02 (temperatura), que no tiene todavia una lectura real de sensor en
    # ningun lugar del proyecto (check_temperature siempre se alimento de un
    # valor estatico del manifiesto, nunca de hwmon/coretemp) -- conectar
    # temperatura con una ruta de sysfs adivinada sin poder verificarla en
    # pacca es mas riesgoso que dejarla pendiente explicitamente.
    load_reader: Callable[[], tuple[float, float, float]] = os.getloadavg,
    load_threshold: float = 1.0,
    package_temperature_reader: Callable[[], float | None] = preflight_module.read_package_temperature_c,
    turbo_state_check: Callable[..., Any] = preflight_module.check_turbo_hwp_unchanged,
    # ARC-129: G01 (GPU sin actividad ajena), por combinación, mismo
    # criterio que E06 arriba pero para el eje de GPU -- None (el default,
    # y todo caller anterior a este cambio) desactiva el check por completo,
    # nunca falla "cerrado" por falta de un inspector NVML en una campaña
    # de solo CPU. Solo se consulta para combinaciones cuyo kernel es
    # device=="gpu".
    gpu_inspector: Any = None,
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
    resolved_harness = harness or load_config().harness

    # CAM-09 (ARC-94): fail closed, before touching any hardware state, if
    # this output_dir already has accepted verdicts from a different
    # measurement protocol (see CampaignProtocolMismatchError).
    _check_protocol_fingerprint(manifest, catalog, resolved_harness)

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
    # ARC-142: CampaignProgress() siempre arranca en cero -- sin esto, una
    # reanudación sobrescribe campaign_metadata.json con total_core_hours/
    # overhead_pct_values que solo cuentan lo medido EN ESTA invocación,
    # descartando en silencio las horas-núcleo y los pares overhead de
    # cualquier sesión anterior (accepted_run_ids/rejected_run_ids no tienen
    # este problema: MET-06 ya los distingue de skipped_run_ids, que
    # preserva las corridas aceptadas antes). Carga el metadata previo, si
    # existe, y usa sus valores como punto de partida.
    _seed_progress_from_previous_metadata(progress, manifest.output_dir)

    # ARC-102: manifest.load_threshold (opcional) tiene prioridad sobre el
    # default de este parámetro -- así un manifiesto YAML real puede
    # declararlo (antes era imposible, el 1.0 estaba fijo como default de
    # función, sin campo real en Manifest que lo expusiera).
    effective_load_threshold = (
        manifest.load_threshold if getattr(manifest, "load_threshold", None) is not None else load_threshold
    )
    temperature_config = getattr(manifest, "temperature", None) or {}
    require_package_sensor = bool(temperature_config.get("require_package_sensor", False))
    temperature_min_c = float(temperature_config.get("minimum_c", 0.0))
    temperature_max_c = float(temperature_config.get("maximum_c", 90.0))
    turbo_config = getattr(manifest, "turbo", None) or {}
    require_turbo_disabled = bool(turbo_config.get("require_disabled", False))
    turbo_snapshot = getattr(environment_profile, "turbo_hwp_state", None) or {}

    def _temperature_check() -> Any:
        temperature_c = package_temperature_reader() if require_package_sensor else None
        check = preflight_module.check_temperature(
            temperature_c, temperature_min_c, temperature_max_c
        )
        if require_package_sensor and temperature_c is None:
            return preflight_module.CheckResult(
                "E02", "Temperatura de paquete", False, True,
                {"temperature_c": "unavailable", "range_c": [temperature_min_c, temperature_max_c]},
                "La campaña exige un sensor de temperatura de paquete legible",
            )
        return check

    def _turbo_check() -> Any | None:
        if not require_turbo_disabled:
            return None
        return turbo_state_check(turbo_snapshot)

    try:
        # E08 (ARC-102): la calibración Roofline/GPU/referencias, igual que
        # cada combinación de la matriz más abajo, puede contaminarse por
        # contención externa -- y a diferencia de una combinación individual
        # (que se puede saltar y reintentar), una calibración contaminada
        # desplaza el ridge point que clasifica TODA la campaña. Se verifica
        # una sola vez aquí, antes de la primera medición real, y aborta la
        # campaña completa (nunca "salta" la calibración) si la carga externa
        # ya está por encima del umbral -- no hay combinación que rechazar
        # todavía, así que la única opción segura es no empezar.
        pre_calibration_load = preflight_module.check_external_load(
            effective_load_threshold, load_reader, max(len(delegated_cpus), 1)
        )
        pre_calibration_temperature = _temperature_check()
        pre_calibration_turbo = _turbo_check()
        progress.pre_calibration_observation = {
            "external_load": pre_calibration_load.observed,
            "package_temperature": pre_calibration_temperature.observed,
            "turbo_hwp": pre_calibration_turbo.observed if pre_calibration_turbo is not None else None,
        }
        write_campaign_metadata(progress, manifest, manifest.output_dir)
        if not pre_calibration_load.passed:
            raise CampaignPreflightError(
                f"E08: carga externa por encima del umbral antes de calibrar, "
                f"observado={pre_calibration_load.observed}"
            )
        if not pre_calibration_temperature.passed:
            raise CampaignPreflightError(
                f"E02: temperatura no apta antes de calibrar, observado={pre_calibration_temperature.observed}"
            )
        if pre_calibration_turbo is not None and not pre_calibration_turbo.passed:
            raise CampaignPreflightError(
                f"E01: estado Turbo/HWP cambió antes de calibrar, observado={pre_calibration_turbo.observed}"
            )

        # CAM-11 (ARC-142): una reanudación también puede ocurrir después
        # de calibrar pero antes del primer verdict.json. Cualquier artefacto
        # de setup existente activa la carga/fallo cerrado: nunca se
        # sobrescribe una calibración parcial o completa en silencio.
        is_resume = _has_existing_campaign_artifacts(Path(manifest.output_dir))

        if is_resume:
            cpu_levels = tuple(getattr(manifest, "frequency_levels", ()) or ())
            native_level_id = next(
                (lvl.id for lvl in cpu_levels if getattr(lvl, "mode", None) == "native_governor"),
                cpu_levels[0].id if cpu_levels else "",
            )
            try:
                # No basta cargar REF: postprocess abre el ridge de la
                # frecuencia de cada combinación. Comprobarlos todos ahora
                # evita ejecutar nuevas cargas antes de descubrir que F2,
                # F3, etc. faltaban o no eran plausibles.
                cpu_calibrations = {
                    level.id: load_calibration(manifest.output_dir, level.id)
                    for level in cpu_levels
                }
                roofline = (
                    cpu_calibrations[native_level_id]
                    if cpu_calibrations else load_calibration(manifest.output_dir, "")
                )
                profile = load_node_profile(manifest.output_dir)
                references = load_calibration_references(manifest.output_dir)
            except (OSError, ValueError, TypeError, calibration_module.CalibrationError) as exc:
                raise CampaignCalibrationMissingError(
                    f"CAM-11: {manifest.output_dir} ya tiene corridas o calibración previa pero falta "
                    f"un artefacto de calibración/perfil/referencias esperado ({exc}). No se "
                    "vuelve a medir en silencio -- resuelve a mano antes de reanudar."
                ) from exc

            gpu_roofline = {}
            gpu_config = getattr(manifest, "gpu", None) or {}
            gpu_calibration_refs = gpu_config.get("calibration") if isinstance(gpu_config, Mapping) else None
            if gpu_calibration_refs:
                gpu_levels = tuple(getattr(manifest, "gpu_frequency_levels", None) or cpu_levels)
                gpu_native_level_id = next(
                    (lvl.id for lvl in gpu_levels if getattr(lvl, "mode", None) == "native_governor"),
                    gpu_levels[0].id if gpu_levels else "",
                )
                try:
                    for precision in ("fp32", "fp64"):
                        calibrations = {
                            level.id: load_calibration(manifest.output_dir, level.id, precision)
                            for level in gpu_levels
                        }
                        gpu_roofline[precision] = (
                            calibrations[gpu_native_level_id]
                            if calibrations else load_calibration(manifest.output_dir, "", precision)
                        )
                except (OSError, ValueError, TypeError, calibration_module.CalibrationError) as exc:
                    raise CampaignCalibrationMissingError(
                        f"CAM-11: {manifest.output_dir} ya tiene corridas o calibración previa pero falta "
                        f"una calibración de GPU esperada ({exc}). No se vuelve a medir en silencio -- "
                        "resuelve a mano antes de reanudar."
                    ) from exc
        else:
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

        combinations = build_matrix(manifest, catalog, seed=manifest.seed)
        progress.run_ids_in_order = [
            runner_module.build_run_id(
                manifest.campaign_id, c.kernel_ref, c.frequency_level.id, c.repetition_index,
                c.gpu_frequency_level.id if c.gpu_frequency_level is not None else None,
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
            combination_gpu_freq_level_id = (
                combination.gpu_frequency_level.id if combination.gpu_frequency_level is not None else None
            )
            telemetry_run_id = runner_module.build_run_id(
                manifest.campaign_id, combination.kernel_ref,
                combination.frequency_level.id, combination.repetition_index,
                combination_gpu_freq_level_id,
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
            elif previous is None:
                # Un directorio sin verdict.json no equivale a "nunca
                # ejecutado": puede ser una interrupción entre el launcher
                # y write_verdict(). Preservar ambos miembros del par antes
                # de usar otra vez los run_id deterministas.
                _archive_incomplete_run(Path(manifest.output_dir), telemetry_run_id)
                _archive_incomplete_run(Path(manifest.output_dir), f"{telemetry_run_id}__baseline")

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

            # G01 (ARC-129): mismo principio que E06 arriba, pero para el eje
            # de GPU -- otro job del clúster compartido puede empezar a usar
            # la GPU a mitad de una campaña de horas, y G01 antes de este
            # cambio solo se corría una vez al inicio (run_campaign_preflight,
            # nunca conectado por combinación). Solo aplica a combinaciones
            # cuyo kernel es device=="gpu"; gpu_inspector=None desactiva el
            # check por completo (nunca falla "cerrado" en una campaña de
            # solo CPU sin inspector NVML disponible).
            if getattr(entry, "device", "cpu") == "gpu" and gpu_inspector is not None:
                gpu_foreign_check = preflight_module.check_gpu_foreign_activity(gpu_inspector)
                if not gpu_foreign_check.passed:
                    logger.warning(
                        "G01: procesos CUDA ajenos en la GPU, se salta la combinación (run_id=%s, pids=%s)",
                        telemetry_run_id, gpu_foreign_check.observed.get("pids"),
                    )
                    run_dir = Path(manifest.output_dir) / telemetry_run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    validation_module.write_verdict(
                        validation_module.Verdict(
                            accepted=False, factor_id="G01",
                            message=f"Procesos CUDA ajenos en la GPU: {gpu_foreign_check.observed.get('pids')}",
                        ),
                        run_dir,
                    )
                    progress.rejected_run_ids.append(telemetry_run_id)
                    seen_run_ids.add(telemetry_run_id)
                    write_campaign_metadata(progress, manifest, manifest.output_dir)  # CAM-02
                    continue

            # E08 (ARC-101): carga externa, verificada CADA VEZ igual que E06
            # justo arriba -- una corrida que arranca durante un pico de
            # carga ajena (otro job del clúster compartido, por ejemplo)
            # produce contención real que ninguna atribución por PID puede
            # evitar ni corregir después. check_external_load() ya existía
            # en preflight.py, nunca estuvo conectada a un camino de
            # producción real.
            load_check = preflight_module.check_external_load(
                effective_load_threshold, load_reader, max(len(delegated_cpus), 1)
            )
            temperature_check = _temperature_check()
            turbo_check = _turbo_check()
            progress.pre_run_observations[telemetry_run_id] = {
                "external_load": load_check.observed,
                "package_temperature": temperature_check.observed,
                "turbo_hwp": turbo_check.observed if turbo_check is not None else None,
            }
            # Persistir antes de lanzar el baseline: si el proceso o el nodo
            # falla durante la medición, la observación previa no se pierde.
            write_campaign_metadata(progress, manifest, manifest.output_dir)
            if not load_check.passed:
                logger.warning(
                    "E08: carga externa por encima del umbral, se salta la combinación (run_id=%s, observed=%s)",
                    telemetry_run_id, load_check.observed,
                )
                run_dir = Path(manifest.output_dir) / telemetry_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                validation_module.write_verdict(
                    validation_module.Verdict(
                        accepted=False, factor_id="E08",
                        message=f"Carga externa por encima del umbral: {load_check.observed}",
                    ),
                    run_dir,
                )
                progress.rejected_run_ids.append(telemetry_run_id)
                seen_run_ids.add(telemetry_run_id)
                write_campaign_metadata(progress, manifest, manifest.output_dir)  # CAM-02
                continue
            if not temperature_check.passed:
                logger.warning(
                    "E02: temperatura de paquete no apta, se salta la combinación (run_id=%s, observed=%s)",
                    telemetry_run_id, temperature_check.observed,
                )
                run_dir = Path(manifest.output_dir) / telemetry_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                validation_module.write_verdict(
                    validation_module.Verdict(
                        accepted=False, factor_id="E02",
                        message=f"Temperatura de paquete no apta: {temperature_check.observed}",
                    ),
                    run_dir,
                )
                progress.rejected_run_ids.append(telemetry_run_id)
                seen_run_ids.add(telemetry_run_id)
                write_campaign_metadata(progress, manifest, manifest.output_dir)
                continue
            if turbo_check is not None and not turbo_check.passed:
                logger.warning(
                    "E01: estado Turbo/HWP cambió, se salta la combinación (run_id=%s, observed=%s)",
                    telemetry_run_id, turbo_check.observed,
                )
                run_dir = Path(manifest.output_dir) / telemetry_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                validation_module.write_verdict(
                    validation_module.Verdict(
                        accepted=False, factor_id="E01",
                        message=f"Estado Turbo/HWP cambió: {turbo_check.observed}",
                    ),
                    run_dir,
                )
                progress.rejected_run_ids.append(telemetry_run_id)
                seen_run_ids.add(telemetry_run_id)
                write_campaign_metadata(progress, manifest, manifest.output_dir)
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
                    # ARC-129: eje de GPU desacoplado del de CPU cuando la
                    # combinación trae su propio nivel (producto cartesiano);
                    # None para toda combinación de CPU o de GPU sin
                    # gpu_frequency_levels declarado (mismo acoplamiento de
                    # siempre).
                    gpu_freq_level_id=combination_gpu_freq_level_id,
                )
                progress.total_core_hours += result.elapsed_seconds * len(delegated_cpus) / 3600.0  # CAM-05/OPS-01

                if item.mode == "baseline":
                    # ARC-142: un baseline que no terminó bien (crash, kernel
                    # que salió con código de error) puede tener
                    # elapsed_seconds > 0 sin haber corrido el workload
                    # completo -- usarlo como referencia infla o distorsiona
                    # overhead_pct sin ninguna señal de que el número no
                    # significa lo que dice. None (igual que un baseline que
                    # nunca corrió) hace que el bloque de abajo se salte el
                    # cálculo para este par en vez de comparar contra un
                    # tiempo que no representa una corrida real.
                    baseline_elapsed_seconds = result.elapsed_seconds if result.success else None
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
                    # ARC-174: insumos para la clasificación de frecuencia
                    # POR VENTANA (validation.classify_frequency_window(),
                    # vía build_windows()) -- mismo dict que runner.py ya lee
                    # para el chequeo estructural agregado, y el mismo modo
                    # (frequency_level.mode) que MAN-10 ya exige que sea
                    # "native_governor"/"fixed" -- nunca inferido de
                    # freq_khz_applied is None (ver WindowContext).
                    frequency_validation = getattr(manifest, "frequency_validation", None) or {}
                    windows_path = run_postprocess(
                        result.run_dir, run_id=run_id, repetition=item.combination.repetition_index,
                        kernel_ref=item.combination.kernel_ref, kernel_entry=entry, node_id=node_id,
                        freq_level_id=item.combination.frequency_level.id, calibration_dir=manifest.output_dir,
                        freq_khz_requested=getattr(applied, "requested_khz", None),
                        freq_khz_applied=getattr(applied, "applied_khz", None),
                        freq_khz_observed=freq_khz_observed,
                        warmup_seconds=entry.warmup_seconds or 0.0, running_ratio_min=manifest.running_ratio_min,
                        rapl_enabled=bool(manifest.rapl.get("enabled", False)), calibration_references=references,
                        # ARC-129: ridge de GPU calibrado por SU PROPIO nivel
                        # de reloj cuando la combinación trae uno independiente
                        # (producto cartesiano); None reusa freq_level_id como
                        # siempre (kernels de CPU, o de GPU sin
                        # gpu_frequency_levels declarado).
                        gpu_freq_level_id=(
                            item.combination.gpu_frequency_level.id
                            if item.combination.gpu_frequency_level is not None else None
                        ),
                        freq_tolerance_fraction=frequency_validation.get("tolerance_fraction"),
                        freq_expected_cpu_count=len(delegated_cpus),
                        freq_grace_seconds=float(frequency_validation.get("grace_seconds", 0.0)),
                        freq_tail_grace_seconds=float(frequency_validation.get("tail_grace_seconds", 0.0)),
                        freq_is_native_governor=item.combination.frequency_level.mode == "native_governor",
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
