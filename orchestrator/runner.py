from __future__ import annotations

from dataclasses import dataclass
import contextlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping

from . import freqctl
from .catalog import KernelEntry, resolve_exec_command, verify_binary
from .config import HarnessConfig, load_config
from .gpu_shim import compiled_blocking_sync_shim, cuda_lib_dirs
from .metadata_schema import merge_metadata
from . import validation as validation_module

logger = logging.getLogger(__name__)

# RUN-03: the launcher's own timeout must never be the only guard. This is a
# generous multiple of the catalog's expected runtime so normal variance
# never triggers a false timeout kill.
SAFETY_MARGIN = 3.0

# RUN-04: after killing the process group, poll a few times before declaring
# the cleanup failed instead of giving up on the first check.
_GROUP_GONE_ATTEMPTS = 20
_GROUP_GONE_INTERVAL_SECONDS = 0.05

# Contrato de los 12 binarios duales. Los sellos usan CLOCK_MONOTONIC, el
# mismo reloj del launcher, y separan dos preguntas distintas: el primer
# despacho en frio (setup de biblioteca/device incluido) y los despachos
# posteriores con recursos ya creados. No se infieren fronteras a partir de
# la vida completa del proceso porque eso volveria a incluir generacion de
# datos y verificacion, que estan deliberadamente fuera de la decision.
_DISPATCH_TIMING_PATTERNS = {
    # Los benches historicamente anteponen un espacio a sus lineas de
    # salida. El contrato es estricto en nombre/conteo, no en indentacion.
    "cold_t0_ns": re.compile(r"^\s*Cold region t0_ns\s*=\s*(\d+)\s*$", re.MULTILINE),
    "setup_complete_ns": re.compile(r"^\s*Setup complete t_ns\s*=\s*(\d+)\s*$", re.MULTILINE),
    "cold_t1_ns": re.compile(r"^\s*Cold region t1_ns\s*=\s*(\d+)\s*$", re.MULTILINE),
    "warm_t0_ns": re.compile(r"^\s*Measured region t0_ns\s*=\s*(\d+)\s*$", re.MULTILINE),
    "warm_t1_ns": re.compile(r"^\s*Measured region t1_ns\s*=\s*(\d+)\s*$", re.MULTILINE),
}


class RunTimeoutError(RuntimeError):
    """Raised when the measured process group could not be cleaned up."""


def _read_dispatch_timing(stdout_path: Path) -> dict[str, Any]:
    """Extrae y valida las fronteras cold/warm de un kernel dual.

    Falla cerrado: un sello ausente, duplicado o no monotonico invalida el
    contrato. Guardar los nanosegundos absolutos en metadata permite integrar
    despues tiempo, RAPL y NVML sobre exactamente el mismo intervalo.
    """
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, int] = {}
    for name, pattern in _DISPATCH_TIMING_PATTERNS.items():
        matches = pattern.findall(stdout)
        if len(matches) != 1:
            raise ValueError(
                f"RUN-10: marcador {name!r} esperado exactamente una vez en {stdout_path}, "
                f"encontrado {len(matches)}"
            )
        values[name] = int(matches[0])

    ordered = [
        values["cold_t0_ns"], values["setup_complete_ns"], values["cold_t1_ns"],
        values["warm_t0_ns"], values["warm_t1_ns"],
    ]
    if ordered != sorted(ordered) or ordered[0] == ordered[2] or ordered[3] == ordered[4]:
        raise ValueError(f"RUN-10: fronteras de tiempo invalidas/no monotonicas: {values}")

    return {
        "contract_version": "cold_warm_v1",
        **values,
        "setup_seconds": (values["setup_complete_ns"] - values["cold_t0_ns"]) / 1e9,
        "first_dispatch_seconds": (values["cold_t1_ns"] - values["setup_complete_ns"]) / 1e9,
        "cold_total_seconds": (values["cold_t1_ns"] - values["cold_t0_ns"]) / 1e9,
        "warm_total_seconds": (values["warm_t1_ns"] - values["warm_t0_ns"]) / 1e9,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Escribe JSON completo y lo publica con rename atomico en el mismo FS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True, default=str)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(temporary_name).unlink()


@dataclass(frozen=True)
class RunResult:
    """Outcome of one telemetry_kernel_launcher invocation (RUN-01..RUN-08)."""

    run_id: str
    kernel_ref: str
    freq_level_id: str
    repetition_index: int
    command: tuple[str, ...]
    exit_code: int
    timed_out: bool
    success: bool
    elapsed_seconds: float
    run_dir: Path
    stdout_path: Path
    stderr_path: Path
    metadata: Mapping[str, Any]
    # FRQ-03: the freqctl.AppliedFrequency this run used (or None when
    # apply_frequency was never invoked, e.g. RUN-08 / frequency_write_capable
    # False). Exposed directly so callers (campaign.py) don't have to dig it
    # back out of `metadata`.
    applied_frequency: Any = None
    # ARC-87: mirrors applied_frequency for the GPU clock axis -- None for
    # CPU-device kernels (never attempted) and for GPU-device kernels when
    # gpu_frequency_write_capable is False (RUN-08's GPU equivalent).
    applied_gpu_frequency: Any = None


def build_run_id(
    campaign_id: str, kernel_ref: str, freq_level_id: str, repetition_index: int,
    gpu_freq_level_id: str | None = None,
) -> str:
    """RUN-02: deterministic run_id, reproducible from the manifest alone.

    ARC-129: `gpu_freq_level_id`, when given, is folded into the id so a
    cartesian CPU x GPU combination (same CPU freq_level_id, different GPU
    level) never collides on run_id -- absent (None, the default) reproduces
    the exact id format every existing manifest/test already depends on.
    """
    gpu_suffix = f"__gpu{gpu_freq_level_id}" if gpu_freq_level_id is not None else ""
    return f"{campaign_id}__{kernel_ref}__{freq_level_id}{gpu_suffix}__rep{repetition_index:02d}"


def _resolve_frequency_level(manifest: Any, freq_level_id: str) -> Any:
    """ARC-78: freqctl.apply_frequency() needs the full FrequencyLevel object
    (reads .mode/.fraction), not just its .id -- passing the bare string
    (the bug this fixes) makes every non-native_governor level silently take
    the wrong code path in freqctl (getattr(level, "mode", None) is None for
    a str), and would raise AttributeError for native_governor/fixed alike
    the first time frequency_write_capable is ever True."""
    for level in manifest.frequency_levels:
        if level.id == freq_level_id:
            return level
    raise ValueError(
        f"RUN-08: freq_level_id={freq_level_id!r} no coincide con ningún "
        "manifest.frequency_levels[*].id -- apply_frequency no puede resolver "
        "el nivel completo"
    )


def _resolve_gpu_frequency_level(manifest: Any, gpu_freq_level_id: str) -> Any:
    """ARC-129: espejo de _resolve_frequency_level() para el eje de GPU,
    contra manifest.gpu_frequency_levels (nunca manifest.frequency_levels --
    esa lista es del eje de CPU, aunque comparta forma/tipo)."""
    for level in manifest.gpu_frequency_levels or ():
        if level.id == gpu_freq_level_id:
            return level
    raise ValueError(
        f"RUN-08: gpu_freq_level_id={gpu_freq_level_id!r} no coincide con ningún "
        "manifest.gpu_frequency_levels[*].id -- apply_gpu_frequency no puede "
        "resolver el nivel completo"
    )


def _format_cpu_list(cpus: Iterable[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def build_command(
    entry: KernelEntry, manifest: Any, run_id: str, harness: HarnessConfig,
    environment_profile: Any = None,
) -> list[str]:
    """RUN-01: the launcher argv is always derived from the catalog entry and
    the manifest, never hardcoded for a specific kernel or campaign."""
    exec_command = resolve_exec_command(entry, harness)
    cores = manifest.cores
    command = [
        harness.binary_path,
        *exec_command,
        "--perf-cpus", _format_cpu_list(cores.delegated_cpus),
        # ARC-55: cores.delegated_cpus se declaraba en cada manifiesto pero
        # nunca restringía dónde corre el proceso medido -- --perf-cpus solo
        # le dice a perf_event_open qué CPUs escuchar, no afinidad real. El
        # launcher ya soporta --pin-workload-cpus (set_affinity() real sobre
        # el hijo antes de execv, usado manualmente en la validación F3.4 de
        # felix pero nunca conectado a la ruta de producción). Encontrado al
        # correr la primera campaña real en pacca: sin esto, ert_probe
        # corría con 32 hilos (todo el nodo) en vez de los 6 delegados,
        # sesgando BW_pico/P_pico y por lo tanto i_ridge.
        "--pin-workload-cpus", _format_cpu_list(cores.delegated_cpus),
        "--collector-cpu", str(cores.collector_cpu),
        "--consumer-cpu", str(cores.consumer_cpu),
        "--interval-ns", str(manifest.interval_ns),
        "--output-dir", str(manifest.output_dir),
        "--run-id", run_id,
    ]
    # ARC-135: real per-window frequency sampling, sourced from the SAME
    # producer tick as the PMU counters -- replaces the old post-hoc single
    # Python read (campaign.py, taken after the workload process already
    # exited) that turned out not to correlate with the requested level at
    # all (confirmed on real campaign data: F4's 0.8GHz floor reading above
    # F0's 3.6GHz ceiling in different runs). Read-only, so this is safe to
    # request unconditionally -- never gated on frequency_write_capable,
    # useful for REF-only runs too. Silently omitted when environment_profile
    # is unavailable or the path can't be resolved (degrades to "not
    # sampled", never a fabricated reading, same contract as UncoreReader).
    if environment_profile is not None:
        freq_path = freqctl.cur_freq_path(environment_profile, cores.delegated_cpus[0])
        if freq_path is not None:
            command += ["--cpu-freq-sysfs-path", str(freq_path)]
            # ARC-142: sample the REST of the delegated CPUs too, not just
            # CPU0 -- pacca's cpufreq domain is per-core (not per-socket
            # like felix's), so the other cores can diverge from CPU0 under
            # Turbo/HWP without this, which the single scalar column could
            # never reveal. Only attempted when CPU0's own path resolved
            # (freqctl.cur_freq_path returning None for CPU0 already means
            # this environment can't read cpufreq at all); any individual
            # extra CPU whose path fails to resolve is simply omitted from
            # the list, degrading that one CPU to "not sampled" instead of
            # dropping the whole feature.
            extra_paths = [
                str(path)
                for cpu in cores.delegated_cpus[1:]
                if (path := freqctl.cur_freq_path(environment_profile, cpu)) is not None
            ]
            if extra_paths:
                command += ["--cpu-freq-sysfs-path-extra", ",".join(extra_paths)]
    # --cgroup-path is optional isolation only (CPP-05); it is never required
    # for perf to attach correctly.
    if manifest.cgroup_path:
        command += ["--cgroup-path", manifest.cgroup_path]
    if not manifest.perf_enabled:
        command.append("--no-perf")
    # ARC-54: manifest.rapl.enabled nunca se traducía a --rapl-pkg/--rapl-dram
    # para el launcher -- la captura de energía nunca funcionó de punta a
    # punta en ninguna campaña real del proyecto (invisible en felix, que no
    # tiene RAPL físicamente; encontrado al correr la primera campaña real en
    # pacca, que sí lo tiene). Resuelve la ruta del dominio RAPL que
    # corresponde al socket pineado (cores.numa_node_pin) usando los alias
    # que environment.py ya deriva del archivo `name` de sysfs
    # ("package-N"/"dram-package-N") -- nunca asume una ruta fija ni el
    # primer dominio disponible; si no hay coincidencia exacta, RAPL
    # simplemente no se activa para esta corrida en vez de adivinar.
    if getattr(manifest, "rapl", {}).get("enabled") and environment_profile is not None:
        domain_paths = getattr(environment_profile, "rapl_domain_paths", None) or {}
        numa_node = getattr(cores, "numa_node_pin", None)
        if numa_node is not None:
            pkg_path = domain_paths.get(f"package-{numa_node}")
            if pkg_path:
                command += ["--rapl-pkg", pkg_path]
            dram_path = domain_paths.get(f"dram-package-{numa_node}")
            if dram_path:
                command += ["--rapl-dram", dram_path]
    # ARC-116: manifest.uncore.enabled -> --enable-uncore. A diferencia de
    # RAPL, uncore_imc no necesita una ruta resuelta aquí -- UncoreReader
    # descubre sus propios boxes desde sysfs en open() (ver
    # telemetry/src/uncore_reader.cpp). check_exclusive_node_allocation
    # (E11, preflight.py) ya bloqueó la campaña si el nodo no está reservado
    # por completo antes de que este comando se construya.
    if getattr(manifest, "uncore", {}).get("enabled"):
        command.append("--enable-uncore")
        # ARC-131: sin esto, el subproceso `perf stat` de uncore hereda la
        # afinidad del launcher (sin restringir) y el kernel puede
        # planificarlo sobre los MISMOS delegated_cpus donde se miden los
        # contadores por-PID del workload -- confirmado empíricamente en
        # paccaA100 (smoke test, ARC-130) que esto degrada
        # FP_ARITH_INST_RETIRED (66% de las ventanas perdían el contador
        # por completo) por contención de scheduling, no de PMCs (uncore_imc
        # es hardware físicamente separado). Se ancla al primer CPU lógico
        # libre después de delegated_cpus/collector_cpu/consumer_cpu --
        # asume numeración contigua (0..N-1 reservados, N+ libre), cierto
        # para todo manifiesto real de este proyecto a la fecha; si algún
        # manifiesto futuro no sigue ese patrón, revisar aquí antes de
        # asumir que sigue siendo seguro.
        reserved_cpus = set(cores.delegated_cpus) | {cores.collector_cpu, cores.consumer_cpu}
        command += ["--uncore-pin-cpu", str(max(reserved_cpus) + 1)]
    # ARC-70: todo kernel GPU necesita NVML. Ademas, una campaña CPU puede
    # habilitarlo explicitamente para medir el mismo subtotal energetico en
    # ambos candidatos (RAPL package+DRAM + GPU, ociosa durante CPU). Antes
    # solo se miraba entry.device y manifest.gpu.enabled se ignoraba para
    # kernels CPU: el smoke 6702 lo demostro con enable_gpu=false, dejando
    # imposible una comparacion energetica simetrica CPU/GPU.
    gpu_config = getattr(manifest, "gpu", {}) or {}
    collect_gpu = getattr(entry, "device", "cpu") == "gpu" or bool(gpu_config.get("enabled", False))
    if collect_gpu:
        command.append("--enable-gpu")
        gpu_interval_ns = getattr(manifest, "gpu_interval_ns", None)
        if gpu_interval_ns is not None:
            command += ["--gpu-interval-ns", str(gpu_interval_ns)]
    return command


def _frequency_slowdown_ratio(manifest: Any, freq_level_id: str, environment_profile: Any) -> float:
    """ARC-141: how much slower a CPU-bound kernel can legitimately run at
    `freq_level_id` relative to the node's fastest available frequency --
    1.0 (no adjustment) for native_governor, or when the frequency range
    cannot be resolved (preserves the old behavior exactly, never raises).
    A fixed level near the bottom of the range (e.g. F4, ARC-136/140: base
    ~3.2GHz down to 0.8GHz once Turbo is disabled) can legitimately take
    ~4x longer than the same kernel at the top of the range -- a flat
    SAFETY_MARGIN calibrated against ordinary run-to-run variance was never
    meant to also absorb a systematic DVFS slowdown of that magnitude."""
    try:
        level = _resolve_frequency_level(manifest, freq_level_id)
    except ValueError:
        return 1.0
    if level.mode != "fixed" or level.fraction is None:
        return 1.0
    available = getattr(environment_profile, "available_frequencies_khz", None) or []
    if not available:
        return 1.0
    max_khz, min_khz = max(available), min(available)
    if max_khz <= 0 or min_khz < 0 or max_khz <= min_khz:
        return 1.0
    target_khz = min_khz + level.fraction * (max_khz - min_khz)
    if target_khz <= 0:
        return 1.0
    return max(1.0, max_khz / target_khz)


def _resolve_timeout_seconds(
    entry: KernelEntry, manifest: Any, *, freq_level_id: str | None = None, environment_profile: Any = None,
) -> float:
    """RUN-03: expected_runtime_seconds x SAFETY_MARGIN, scaled by the
    expected DVFS slowdown of `freq_level_id` (ARC-141) when both are
    known -- never smaller than the manifest's own generic run timeout,
    which used to be silently ignored whenever the catalog declared
    expected_runtime_seconds (i.e. always). Entries without a declared
    runtime (e.g. some calibration kernels) fall back to the manifest
    timeout alone, as before."""
    manifest_timeout = float(manifest.timeouts_seconds.run)
    if entry.expected_runtime_seconds is None:
        return manifest_timeout
    ratio = (
        _frequency_slowdown_ratio(manifest, freq_level_id, environment_profile)
        if freq_level_id is not None else 1.0
    )
    return max(float(entry.expected_runtime_seconds) * ratio * SAFETY_MARGIN, manifest_timeout)


def _terminate_process_group(pgid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_process_group_gone(
    pgid: int, attempts: int = _GROUP_GONE_ATTEMPTS, interval: float = _GROUP_GONE_INTERVAL_SECONDS
) -> bool:
    for _ in range(attempts):
        if not _process_group_alive(pgid):
            return True
        time.sleep(interval)
    return not _process_group_alive(pgid)


def _check_success(entry: KernelEntry, exit_code: int, stdout_path: Path) -> bool:
    """RUN-05: apply entry.success_check against the real result. catalog.py
    already validated the check's shape (CAT-03/C03) when the catalog loaded."""
    check = entry.success_check
    check_type = check.get("type")
    if check_type == "exit_code":
        return exit_code == check.get("expected", 0)
    if check_type == "stdout_regex":
        pattern = check.get("pattern")
        text = stdout_path.read_text(errors="replace")
        return re.search(pattern, text) is not None
    return False


def _read_launcher_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            return json.load(metadata_file)
    except FileNotFoundError:
        # The launcher only writes metadata.json on a clean exit; a killed or
        # crashed run legitimately has none. success is already False by the
        # time this matters.
        return {}


def _merge_metadata(
    launcher_metadata: Mapping[str, Any],
    entry: KernelEntry,
    manifest: Any,
    kernel_ref: str,
    freq_level_id: str,
    repetition_index: int,
    node_id: str | None,
    calibration_refs: Mapping[str, Any] | None,
    applied_frequency: Any = None,
    applied_gpu_frequency: Any = None,
    gpu_freq_level_id: str | None = None,
) -> dict[str, Any]:
    """RUN-06: merge launcher metadata (samples_collected, push_retries,
    perf_attach_mode, measured_pids, ...) with orchestrator-level metadata,
    via metadata_schema.merge_metadata() (MET-01: never {**a, **b})."""
    orchestrator_fields: dict[str, Any] = {
        "campaign_id": manifest.campaign_id,
        "kernel_ref": kernel_ref,
        "kernel_suite": entry.suite,
        "kernel_role": entry.role,
        "freq_level_id": freq_level_id,
        # ARC-129: None para toda corrida de CPU, o de GPU sin producto
        # cartesiano declarado (manifest.gpu_frequency_levels ausente) --
        # el eje de GPU en ese caso sigue acoplado a freq_level_id, como
        # siempre.
        "gpu_freq_level_id": gpu_freq_level_id,
        "repetition_index": repetition_index,
        "node_id": node_id,
        "binary_checksum": entry.binary_checksum,
    }
    if applied_frequency is not None:
        # FRQ-03: both the requested and the applied value, never only one,
        # and never silently dropped between freqctl.apply_frequency() and
        # this run's own persisted metadata.
        orchestrator_fields["freq_khz_requested"] = getattr(applied_frequency, "requested_khz", None)
        orchestrator_fields["freq_khz_applied"] = getattr(applied_frequency, "applied_khz", None)
        orchestrator_fields["freq_governor_applied"] = getattr(applied_frequency, "governor_applied", None)
        orchestrator_fields["freq_write_skipped_reason"] = getattr(applied_frequency, "write_skipped_reason", None)
    if applied_gpu_frequency is not None:
        # ARC-87: espejo de FRQ-03 para el eje de GPU -- requested/applied en
        # MHz (no kHz, unidad nativa de nvidia-smi/NVML) nunca se calculan y
        # luego se descartan silenciosamente antes de llegar a metadata.json.
        orchestrator_fields["gpu_freq_mhz_requested"] = getattr(applied_gpu_frequency, "requested_mhz", None)
        orchestrator_fields["gpu_freq_mhz_applied"] = getattr(applied_gpu_frequency, "applied_mhz", None)
        orchestrator_fields["gpu_freq_write_skipped_reason"] = getattr(
            applied_gpu_frequency, "write_skipped_reason", None
        )
    if calibration_refs:
        orchestrator_fields = merge_metadata(orchestrator_fields, calibration_refs, context="RUN-06")

    return merge_metadata(launcher_metadata, orchestrator_fields, context="RUN-06")


def run_single(
    entry: KernelEntry,
    manifest: Any,
    kernel_ref: str,
    freq_level_id: str,
    repetition_index: int,
    *,
    environment_profile: Any = None,
    harness: HarnessConfig | None = None,
    node_id: str | None = None,
    calibration_refs: Mapping[str, Any] | None = None,
    apply_frequency: Callable[[Any, Any, Any], Any] | None = None,
    apply_gpu_frequency: Callable[[Any, Any], Any] | None = None,
    run_id: str | None = None,
    gpu_freq_level_id: str | None = None,
) -> RunResult:
    """Run one telemetry_kernel_launcher invocation and collect its result.

    ARC-94: `run_id`, if given, overrides the one this function would
    otherwise derive from `build_run_id(campaign_id, kernel_ref,
    freq_level_id, repetition_index)`. Before this parameter existed,
    campaign.py computed a mode-suffixed id for the baseline/telemetry pair
    (CAM-04, e.g. `..._rep01__baseline` vs. `..._rep01`) but had no way to
    pass it in -- `run_single()` always rebuilt the plain telemetry id
    internally, so the baseline half of the pair silently wrote into the
    SAME run_dir as its telemetry sibling (whichever ran second overwrote
    the other's artifacts). The same gap meant retrying a rejected run
    overwrote the rejected run's own evidence instead of landing in the
    directory the caller intended.

    `apply_frequency`, if given, is only ever called when
    environment_profile.frequency_write_capable is True (RUN-08); freqctl.py
    is the caller's concern, run_single just enforces the gate. Its return
    value (a freqctl.AppliedFrequency, or whatever the caller's fake
    returns) is kept on RunResult.applied_frequency and folded into this
    run's metadata.json (FRQ-03): the requested/applied frequency must never
    be computed and then silently dropped before it reaches persisted data.

    `apply_gpu_frequency`, if given, mirrors `apply_frequency` for the GPU
    clock axis (ARC-87): only called when `entry.device == "gpu"` AND
    `environment_profile.gpu_frequency_write_capable` is true. A CPU-device
    kernel never touches the GPU clock, regardless of write capability --
    the two axes are gated independently, exactly like the two independent
    control domains (CPU/GPU) described in the DVFS policy design.

    `gpu_freq_level_id` (ARC-129): when given, the GPU axis resolves against
    `manifest.gpu_frequency_levels` using THIS id, independent of
    `freq_level_id` (which always drives the CPU axis) -- enables a real
    cartesian CPU x GPU sweep (campaign.build_matrix). None (the default)
    preserves the old coupled behavior: the GPU axis reuses `freq_level_id`
    against `manifest.frequency_levels`, exactly as before this existed.
    """
    harness = harness or load_config().harness

    # CAT-07: re-verify the binary on disk immediately before every run, not
    # only once during preflight.
    if not verify_binary(entry, node_id):
        raise ValueError(f"C02: checksum de {entry.exec_path!r} no coincide antes de ejecutar")

    applied_frequency = None
    if apply_frequency is not None:
        frequency_level = _resolve_frequency_level(manifest, freq_level_id)
        if environment_profile is not None and getattr(environment_profile, "frequency_write_capable", False):
            applied_frequency = apply_frequency(manifest.cores.delegated_cpus, frequency_level, environment_profile)
            # ARC-161: espera activa opcional (manifest.frequency_settle) --
            # sin esto, una corrida corta puede medirse bajo el techo de
            # frecuencia anterior mientras el hardware todavía decae hacia
            # el nivel pedido (EPP=performance bajo HWP, confirmado en
            # paccaA100 con scaling_cur_freq muestreado en vivo).
            freqctl.settle_if_configured(
                manifest.cores.delegated_cpus, applied_frequency, environment_profile,
                settle_config=getattr(manifest, "frequency_settle", None),
            )
        elif frequency_level.mode != "native_governor":
            # RUN-09 (ARC-101): a "fixed" level was requested but this node
            # cannot actually write frequency. Silently running at native
            # frequency while the run/window still gets labeled with this
            # freq_level_id would corrupt the dataset without any signal --
            # confirmed to happen in practice (ARC-97 campaign produced
            # accepted F0-F4 runs at ~800MHz/native, freq_khz_requested/
            # applied both empty). Fail loud instead of degrading silently.
            raise RuntimeError(
                f"RUN-09: freq_level_id={freq_level_id!r} (mode={frequency_level.mode!r}) "
                f"para {kernel_ref!r} requiere escritura real de frecuencia CPU, pero "
                "frequency_write_capable=False -- no se aplica silenciosamente a la "
                "frecuencia nativa. Si esta corrida debía ser solo REF, revisa el manifiesto."
            )
        else:
            logger.debug(
                "RUN-08: frequency_write_capable=False, nivel %s es native_governor y no requiere escritura",
                freq_level_id,
            )

    applied_gpu_frequency = None
    if apply_gpu_frequency is not None and getattr(entry, "device", "cpu") == "gpu":
        # ARC-129: eje de GPU desacoplado del de CPU cuando el caller pasa
        # gpu_freq_level_id (campaign.build_matrix ya lo resuelve así para
        # un producto cartesiano real); sin él, mismo comportamiento
        # acoplado de siempre (reusa freq_level_id contra frequency_levels).
        gpu_frequency_level = (
            _resolve_gpu_frequency_level(manifest, gpu_freq_level_id)
            if gpu_freq_level_id is not None
            else _resolve_frequency_level(manifest, freq_level_id)
        )
        if environment_profile is not None and getattr(environment_profile, "gpu_frequency_write_capable", False):
            applied_gpu_frequency = apply_gpu_frequency(gpu_frequency_level, environment_profile)
        elif gpu_frequency_level.mode != "native_governor":
            # RUN-09 (ARC-101): same principle as the CPU branch above, for
            # the GPU clock axis (ARC-87).
            effective_gpu_level_id = gpu_freq_level_id if gpu_freq_level_id is not None else freq_level_id
            raise RuntimeError(
                f"RUN-09: gpu_freq_level_id={effective_gpu_level_id!r} "
                f"(mode={gpu_frequency_level.mode!r}) para {kernel_ref!r} requiere escritura real de "
                "frecuencia GPU, pero gpu_frequency_write_capable=False -- no se aplica silenciosamente "
                "al reloj nativo."
            )
        else:
            logger.debug(
                "ARC-87: gpu_frequency_write_capable=False, nivel %s es native_governor y no requiere escritura",
                gpu_frequency_level.id,
            )

    run_id = run_id if run_id is not None else build_run_id(
        manifest.campaign_id, kernel_ref, freq_level_id, repetition_index, gpu_freq_level_id
    )
    run_dir = Path(manifest.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(entry, manifest, run_id, harness, environment_profile)
    timeout_seconds = _resolve_timeout_seconds(
        entry, manifest, freq_level_id=freq_level_id, environment_profile=environment_profile,
    )

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    start = time.monotonic()
    timed_out = False
    # ARC-55: sin esto, binarios OpenMP (STREAM, ERT, NPB) heredan el
    # OMP_NUM_THREADS del proceso del orquestador (normalmente sin fijar) y
    # spawean tantos hilos como CPUs ve el nodo, no los delegados a esta
    # campaña -- confirmado empíricamente en pacca (ert_probe reportaba
    # OPENMP_THREADS=32 en vez de 6). Complementa --pin-workload-cpus
    # (afinidad real) con el conteo de hilos que ese subconjunto de cores
    # justifica; ninguno de los dos por separado alcanza.
    run_env = dict(os.environ)
    run_env["OMP_NUM_THREADS"] = str(len(manifest.cores.delegated_cpus))
    # ARC-70: cudaDeviceSynchronize() hace spin por defecto -- un CPU
    # esperando a la GPU se ve compute_bound (IPC alto, casi cero
    # cache-misses) para el clasificador, un error real, no solo ruido (ver
    # Diseno_Politica_DVFS_CPU_GPU.md sección 3.5.a/4.1). El shim LD_PRELOAD
    # fuerza cudaDeviceScheduleBlockingSync sin tocar el binario de terceros
    # (Rodinia). Si no se puede compilar en este nodo (sin nvcc/CUDA), la
    # corrida sigue -- degradación conocida, no un fallo duro, igual que
    # stalled_cycles_backend/l2_lines_in_all cuando el nodo no los soporta.
    if getattr(entry, "device", "cpu") == "gpu":
        # Los kernels duales son propios (config_id no nulo) y fijan
        # cudaDeviceScheduleBlockingSync dentro de main(), despues de
        # cold_t0 y antes de la primera llamada CUDA. Inyectar el shim en
        # ellos ejecutaria cudaSetDeviceFlags() en un constructor previo a
        # main y podria sacar la inicializacion de contexto fuera de la
        # region fria que el experimento necesita medir. El shim se conserva
        # para binarios externos que no podemos modificar.
        if entry.config_id is None:
            shim_path = compiled_blocking_sync_shim()
            if shim_path is not None:
                run_env["LD_PRELOAD"] = f"{shim_path}:{run_env.get('LD_PRELOAD', '')}".rstrip(":")
            else:
                logger.warning(
                    "ARC-70: no se pudo compilar el shim de blocking sync para %s -- "
                    "cudaDeviceSynchronize() hará spin (comportamiento por defecto de CUDA)",
                    kernel_ref,
                )
        # ARC-74: cudart y cublas viven en árboles distintos del mismo HPC
        # SDK en paccaA100 (cuda/<ver>/lib64 vs math_libs/.../lib) -- ambos
        # se agregan cuando existen, nunca se asume que un kernel GPU solo
        # necesita uno de los dos (cublas_dgemm_bench falló en la primera
        # corrida real con "libcublas.so.12: cannot open shared object
        # file" usando solo el primero).
        lib_dirs = cuda_lib_dirs()
        if lib_dirs:
            prefix = ":".join(str(d) for d in lib_dirs)
            run_env["LD_LIBRARY_PATH"] = f"{prefix}:{run_env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    # start_new_session=True makes the child its own process group leader, so
    # os.killpg(child.pid, ...) also reaches everything it forks (RUN-03/04).
    with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
        process = subprocess.Popen(
            command, stdout=stdout_file, stderr=stderr_file, start_new_session=True, env=run_env
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process.pid)
            exit_code = process.wait()
    elapsed = time.monotonic() - start

    # RUN-04: the next combination must not start with leftover processes
    # from this one, whether it timed out or exited on its own.
    group_gone = _wait_process_group_gone(process.pid)
    if not group_gone:
        _terminate_process_group(process.pid)
        group_gone = _wait_process_group_gone(process.pid, attempts=10)
    if not group_gone:
        raise RunTimeoutError(
            f"RUN-04: quedan procesos vivos en el grupo de {run_id} (pgid={process.pid})"
        )

    success = (not timed_out) and _check_success(entry, exit_code, stdout_path)

    launcher_metadata = _read_launcher_metadata(run_dir)
    # Todo kernel del catalogo dual (config_id != None) debe demostrar las
    # dos fronteras temporales acordadas. Una verificacion numerica exitosa
    # no basta si no se puede saber que energia/tiempo pertenecen al primer
    # despacho y cuales al estado reutilizado. Se degrada a corrida fallida
    # sin abortar la campaña completa; los crudos y el motivo se conservan.
    if success and entry.config_id is not None:
        try:
            dispatch_timing = _read_dispatch_timing(stdout_path)
        except ValueError as error:
            success = False
            launcher_metadata = merge_metadata(
                launcher_metadata,
                {"dispatch_timing_contract_valid": False, "dispatch_timing_error": str(error)},
                context="RUN-10",
            )
        else:
            launcher_metadata = merge_metadata(
                launcher_metadata,
                {"dispatch_timing_contract_valid": True, "dispatch_timing": dispatch_timing},
                context="RUN-10",
            )
    metadata = _merge_metadata(
        launcher_metadata,
        entry,
        manifest,
        kernel_ref,
        freq_level_id,
        repetition_index,
        node_id,
        calibration_refs,
        applied_frequency,
        applied_gpu_frequency,
        gpu_freq_level_id,
    )

    frequency_validation = getattr(manifest, "frequency_validation", None) or {}
    require_per_window_frequency = bool(frequency_validation.get("require_per_window", False))
    frequency_tolerance_fraction = frequency_validation.get("tolerance_fraction")
    # ARC-149: success ya refleja si el launcher realmente corrio (RUN-04);
    # sin este chequeo, una corrida fallida (timeout, exit_code != 0) nunca
    # escribio samples.csv y validate_cpu_frequency_trace() crasheaba con un
    # FileNotFoundError sin manejar en vez de dejar que el resto del
    # pipeline (RunResult.success) marque la corrida como rechazada.
    if success and getattr(entry, "device", "cpu") == "cpu" and require_per_window_frequency:
        expected_khz = getattr(applied_frequency, "requested_khz", None)
        frequency_verdict, frequency_summary = validation_module.validate_cpu_frequency_trace(
            run_dir / "samples.csv",
            require_per_window=True,
            expected_khz=expected_khz,
            tolerance_fraction=frequency_tolerance_fraction,
            expected_cpu_count=len(manifest.cores.delegated_cpus),
            grace_seconds=float(frequency_validation.get("grace_seconds", 0.0)),
            tail_grace_seconds=float(frequency_validation.get("tail_grace_seconds", 0.0)),
        )
        frequency_summary.update({
            "accepted": frequency_verdict.accepted,
            "factor_id": frequency_verdict.factor_id,
            "message": frequency_verdict.message,
        })
        metadata = merge_metadata(
            metadata,
            {"frequency_trace_validation": frequency_summary},
            context="RUN-E01",
        )
    # ARC-94 (segunda ronda): el diccionario fusionado (RUN-06) solo vivía
    # en memoria, en RunResult.metadata -- metadata.json en disco se
    # quedaba con lo que el launcher escribió (samples_collected,
    # push_retries, perf_attach_mode) y nunca con campaign_id/kernel_ref/
    # checksum/frecuencias que el orquestador agrega después. Confirmado en
    # producción: 100% de los metadata.json aceptados en paccaA100 carecían
    # de esos campos. El comando ejecutado (`command`) tampoco se
    # persistía en ningún lado -- se agrega aquí, no solo en RunResult.
    metadata_path = run_dir / "metadata.json"
    _write_json_atomic(metadata_path, {**metadata, "command": list(command)})

    return RunResult(
        run_id=run_id,
        kernel_ref=kernel_ref,
        freq_level_id=freq_level_id,
        repetition_index=repetition_index,
        command=tuple(command),
        exit_code=exit_code,
        timed_out=timed_out,
        success=success,
        elapsed_seconds=elapsed,
        run_dir=run_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        metadata=metadata,
        applied_frequency=applied_frequency,
        applied_gpu_frequency=applied_gpu_frequency,
    )
