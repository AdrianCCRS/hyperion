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
import time
from typing import Any, Callable, Iterable, Mapping

from .catalog import KernelEntry, resolve_exec_command, verify_binary
from .config import HarnessConfig, load_config
from .gpu_shim import compiled_blocking_sync_shim, cuda_lib_dirs
from .metadata_schema import merge_metadata

logger = logging.getLogger(__name__)

# RUN-03: the launcher's own timeout must never be the only guard. This is a
# generous multiple of the catalog's expected runtime so normal variance
# never triggers a false timeout kill.
SAFETY_MARGIN = 3.0

# RUN-04: after killing the process group, poll a few times before declaring
# the cleanup failed instead of giving up on the first check.
_GROUP_GONE_ATTEMPTS = 20
_GROUP_GONE_INTERVAL_SECONDS = 0.05


class RunTimeoutError(RuntimeError):
    """Raised when the measured process group could not be cleaned up."""


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


def build_run_id(campaign_id: str, kernel_ref: str, freq_level_id: str, repetition_index: int) -> str:
    """RUN-02: deterministic run_id, reproducible from the manifest alone."""
    return f"{campaign_id}__{kernel_ref}__{freq_level_id}__rep{repetition_index:02d}"


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
    # ARC-70: kernels GPU (Rodinia u otros del catálogo con device="gpu")
    # necesitan que el Collector muestree NVML -- el launcher ya soporta
    # --enable-gpu/--gpu-interval-ns (ARC-68), solo faltaba conectarlo desde
    # el catálogo. gpu_interval_ns es opcional en el manifiesto; si no se
    # declara, se omite el flag y el launcher usa su propio default (100ms).
    if getattr(entry, "device", "cpu") == "gpu":
        command.append("--enable-gpu")
        gpu_interval_ns = getattr(manifest, "gpu_interval_ns", None)
        if gpu_interval_ns is not None:
            command += ["--gpu-interval-ns", str(gpu_interval_ns)]
    return command


def _resolve_timeout_seconds(entry: KernelEntry, manifest: Any) -> float:
    """RUN-03: expected_runtime_seconds x SAFETY_MARGIN, or the manifest's
    generic run timeout for entries that do not declare a runtime (e.g. some
    calibration kernels)."""
    if entry.expected_runtime_seconds is not None:
        return float(entry.expected_runtime_seconds) * SAFETY_MARGIN
    return float(manifest.timeouts_seconds.run)


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
    """
    harness = harness or load_config().harness

    # CAT-07: re-verify the binary on disk immediately before every run, not
    # only once during preflight.
    if not verify_binary(entry, node_id):
        raise ValueError(f"C02: checksum de {entry.exec_path!r} no coincide antes de ejecutar")

    applied_frequency = None
    if apply_frequency is not None:
        if environment_profile is not None and getattr(environment_profile, "frequency_write_capable", False):
            frequency_level = _resolve_frequency_level(manifest, freq_level_id)
            applied_frequency = apply_frequency(manifest.cores.delegated_cpus, frequency_level, environment_profile)
        else:
            logger.debug(
                "RUN-08: frequency_write_capable=False, omitiendo apply_frequency para %s/%s",
                kernel_ref,
                freq_level_id,
            )

    applied_gpu_frequency = None
    if apply_gpu_frequency is not None and getattr(entry, "device", "cpu") == "gpu":
        if environment_profile is not None and getattr(environment_profile, "gpu_frequency_write_capable", False):
            frequency_level = _resolve_frequency_level(manifest, freq_level_id)
            applied_gpu_frequency = apply_gpu_frequency(frequency_level, environment_profile)
        else:
            logger.debug(
                "ARC-87: gpu_frequency_write_capable=False, omitiendo apply_gpu_frequency para %s/%s",
                kernel_ref,
                freq_level_id,
            )

    run_id = run_id if run_id is not None else build_run_id(
        manifest.campaign_id, kernel_ref, freq_level_id, repetition_index
    )
    run_dir = Path(manifest.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(entry, manifest, run_id, harness, environment_profile)
    timeout_seconds = _resolve_timeout_seconds(entry, manifest)

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
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump({**metadata, "command": list(command)}, metadata_file, indent=2, sort_keys=True, default=str)
        metadata_file.write("\n")

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
