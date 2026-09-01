from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from . import freqctl
from . import runner
from .catalog import KernelEntry
from .runner import RunResult

logger = logging.getLogger(__name__)

# D03/CAL-04: +/-40% around the manifest's declared datasheet peak values.
D03_TOLERANCE_FRACTION = 0.40

# CAL-10: default coefficient-of-variation threshold for reference stability.
DEFAULT_CV_THRESHOLD_PCT = 5.0

# CAL-09: the reference kernel must be repeated at least this many times to
# make a P95/CV% estimate meaningful.
MIN_REFERENCE_REPETITIONS = 5

_CALIBRATION_FREQ_LEVEL_ID = "F0_calibration"


class CalibrationError(RuntimeError):
    """Roofline/reference calibration could not be trusted (CAL-04/06/09)."""


@dataclass(frozen=True)
class RooflineCalibration:
    campaign_id: str
    timestamp: str
    delegated_cpus: str
    bw_pico_bytes_per_s: float
    p_pico_flops_per_s: float
    i_ridge_flops_per_byte: float
    stream_raw_output: str
    ert_raw_output: str
    plausibility_check_passed: bool
    plausibility_message: str = ""
    # ARC-78: a qué nivel de manifest.frequency_levels corresponde este
    # i_ridge -- P_pico escala con el reloj de núcleo, BW_pico no, así que
    # un solo i_ridge por campaña clasifica mal cualquier ventana que no sea
    # del mismo nivel de frecuencia en que se calibró. "" solo aparece en
    # datos persistidos antes de este cambio (una campaña con un único nivel
    # de frecuencia, tratado como si fuera la referencia).
    freq_level_id: str = ""
    # ARC-80: "fp32"/"fp64" cuando este RooflineCalibration es un ridge de
    # GPU (run_gpu_calibration) -- GPU tiene dos ridge points distintos
    # porque los kernels del catálogo usan una u otra precisión (ARC-76).
    # "" para calibraciones de CPU (run_calibration), que no distinguen
    # precisión.
    gpu_precision: str = ""


@dataclass(frozen=True)
class CalibrationReferences:
    node_id: str
    ipc_p95: float
    ips_p95: float
    mpki_p95: float
    miss_rate_p95: float
    repetitions: int
    cv_pct: float
    accepted: bool


def _format_cpu_list(cpus: Sequence[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def _extract_metric(pattern: str, text: str, *, label: str) -> float:
    match = re.search(pattern, text)
    if not match or not match.groups():
        raise CalibrationError(
            f"CAL-02/CAL-03: no se pudo extraer {label} del stdout con el patrón {pattern!r}"
        )
    try:
        return float(match.group(1))
    except ValueError as error:
        raise CalibrationError(f"CAL-02/CAL-03: {label} no es numérico: {match.group(1)!r}") from error


def _check_plausibility(
    bw_pico: float, p_pico: float, datasheet: Mapping[str, float] | None
) -> tuple[bool, str]:
    """CAL-04/D03. An undeclared datasheet is a failed check, not a skipped
    one (ARC-20: never approve a check due to missing data).

    Only meaningful for the campaign's *reference* frequency level
    (native_governor) -- the datasheet declares one nominal peak, not one
    per frequency level. Other levels use
    _check_plausibility_relative_to_reference() instead (ARC-78)."""
    if not datasheet or "bw_pico_bytes_per_s" not in datasheet or "p_pico_flops_per_s" not in datasheet:
        return False, "D03: manifest.hardware_datasheet no declara bw_pico_bytes_per_s/p_pico_flops_per_s"

    def within_range(observed: float, expected: float) -> bool:
        return expected > 0 and abs(observed - expected) / expected <= D03_TOLERANCE_FRACTION

    bw_ok = within_range(bw_pico, datasheet["bw_pico_bytes_per_s"])
    p_ok = within_range(p_pico, datasheet["p_pico_flops_per_s"])
    if bw_ok and p_ok:
        return True, ""
    reasons = []
    if not bw_ok:
        reasons.append(
            f"BW_pico observado={bw_pico:.3e} fuera de ±{D03_TOLERANCE_FRACTION:.0%} "
            f"de {datasheet['bw_pico_bytes_per_s']:.3e}"
        )
    if not p_ok:
        reasons.append(
            f"P_pico observado={p_pico:.3e} fuera de ±{D03_TOLERANCE_FRACTION:.0%} "
            f"de {datasheet['p_pico_flops_per_s']:.3e}"
        )
    return False, "D03: " + "; ".join(reasons)


# ARC-78: margen de ruido de medicion para el chequeo de consistencia
# interna de niveles de frecuencia no-referencia (ver
# _check_plausibility_relative_to_reference).
_RELATIVE_TOLERANCE_FRACTION = 0.05


def _check_plausibility_relative_to_reference(
    bw_pico: float, p_pico: float, reference: "RooflineCalibration"
) -> tuple[bool, str]:
    """ARC-78: D03 para un nivel de frecuencia que NO es la referencia
    (native_governor). El datasheet del manifiesto declara un único pico
    "nominal" -- no hay un pico distinto declarado por cada nivel reducido
    contra el cual comparar, así que exigir el mismo ±40% del datasheet
    sería inventar un número. En su lugar se verifica consistencia física
    interna: un nivel a menor reloj no puede medir un P_pico mayor que el
    de referencia (con un margen de ruido) -- si lo hace, la frecuencia
    pedida no se aplicó de verdad (o se aplicó al revés), y ese es
    precisamente el tipo de calibración implausible que D03 existe para
    bloquear."""
    if bw_pico <= 0:
        return False, f"D03: BW_pico debe ser positivo, se obtuvo {bw_pico}"
    limit = reference.p_pico_flops_per_s * (1.0 + _RELATIVE_TOLERANCE_FRACTION)
    if p_pico > limit:
        return False, (
            f"D03: P_pico observado={p_pico:.3e} supera al de referencia "
            f"({reference.p_pico_flops_per_s:.3e}, nivel {reference.freq_level_id!r}) "
            f"más allá del margen de ruido (±{_RELATIVE_TOLERANCE_FRACTION:.0%}) -- "
            "un nivel de frecuencia más bajo no debería medir más FLOPs/s que la referencia"
        )
    return True, ""


def calibration_filename(freq_level_id: str, gpu_precision: str = "") -> str:
    # ARC-78: "" es el único caso que usa el nombre viejo -- una campaña con
    # un solo nivel de frecuencia procesada por la ruta de compatibilidad de
    # run_calibration() (manifest de prueba sin frequency_levels). Cualquier
    # freq_level_id real (incluida una campaña con un solo nivel REAL) usa
    # el nombre por-nivel, para que load_calibration() siempre sepa cuál
    # archivo pedir sin adivinar.
    # ARC-80: gpu_precision agrega un segundo eje -- GPU tiene un ridge
    # distinto para fp32 y fp64 (ARC-76), así que un mismo freq_level_id
    # puede tener hasta dos archivos de calibración GPU.
    if gpu_precision:
        if not freq_level_id:
            return f"roofline_calibration_gpu_{gpu_precision}.json"
        return f"roofline_calibration_gpu_{gpu_precision}_{freq_level_id}.json"
    if not freq_level_id:
        return "roofline_calibration.json"
    return f"roofline_calibration_{freq_level_id}.json"


def write_calibration(calibration: RooflineCalibration, output_dir: str | Path) -> Path:
    path = Path(output_dir) / calibration_filename(calibration.freq_level_id, calibration.gpu_precision)
    with path.open("w", encoding="utf-8") as calibration_file:
        json.dump(asdict(calibration), calibration_file, indent=2, sort_keys=True)
        calibration_file.write("\n")
    return path


def load_calibration(
    output_dir: str | Path, freq_level_id: str = "", gpu_precision: str = ""
) -> RooflineCalibration:
    """CAL-06: refuses (raises) a calibration that failed its plausibility
    check. postprocess.py must never label a window against an unverified
    I_ridge.

    ARC-78: `freq_level_id` selects WHICH frequency level's i_ridge to load
    -- omitting it only works for campaigns that never declared
    manifest.frequency_levels (test doubles) or legacy single-level
    artifacts written before this change. A real campaign's postprocess
    call always passes the window's own freq_level_id, never the default.

    ARC-80: `gpu_precision` ("fp32"/"fp64") selects the GPU ridge instead of
    the CPU one -- required for any device="gpu" dataset kernel.
    """
    path = Path(output_dir) / calibration_filename(freq_level_id, gpu_precision)
    with path.open(encoding="utf-8") as calibration_file:
        data = json.load(calibration_file)
    calibration = RooflineCalibration(**data)
    if not calibration.plausibility_check_passed:
        raise CalibrationError(
            f"CAL-06: {path} tiene plausibility_check_passed=False: {calibration.plausibility_message}"
        )
    return calibration


def _require_valid_frequency_trace(
    result: Any,
    ref: str,
    freq_level_id: str,
    *,
    label: str,
    require_per_window: bool,
) -> None:
    """ARC-141: `run_single()` ya calcula `frequency_trace_validation` (E01,
    ARC-138) contra las lecturas reales del colector cuando el manifiesto
    declara `frequency_validation.require_per_window`.

    ARC-167 (2026-08-19): a pesar del nombre, esta función ya NO bloquea la
    calibración -- solo registra un `logger.warning()` cuando la traza
    falta o queda rechazada. Downgrade deliberado, decidido explícitamente
    por el usuario tras evidencia real en `paccaA100`: `stream_official`
    (kernel de calibración de BW_pico, corto y con barreras de
    sincronización entre sus 4 fases Copy/Scale/Add/Triad) produce
    dispersión dispersa y real de `scaling_cur_freq` -- no un transitorio
    de arranque (ya cubierto por `frequency_validation.grace_seconds`,
    ARC-166), sino caídas puntuales cerca del final de la corrida cuando
    algunos hilos terminan su fase antes que otros y quedan momentáneamente
    ociosos, diluyendo el promedio APERF/MPERF de esa ventana sin que el
    candado de frecuencia haya cambiado realmente. `npb_mg` (kernel real
    del dataset) corre limpio bajo el mismo nivel F0 (ARC-156) -- este
    patrón es propio de la estructura de `stream_official`/`ert_probe`
    como microbenchmarks de calibración, no algo que se espere en las
    corridas reales del dataset. Crucialmente, `P_pico`/`BW_pico` se
    extraen del `stdout` del propio programa vía regex (`_extract_metric`),
    nunca de esta traza PMU (ARC-156) -- CAL-07 en calibración siempre fue
    una verificación redundante de calidad, no la fuente del número
    calibrado, así que bloquear la campaña completa por su ruido no está
    justificado. `E01` (`runner.py`, corridas reales del dataset) NO
    cambia -- sigue siendo bloqueante ahí, donde sí importa que la
    telemetría por ventana sea confiable para clasificación."""
    frequency_trace = (result.metadata or {}).get("frequency_trace_validation")
    if require_per_window and frequency_trace is None:
        logger.warning(
            "%s: la calibración (%s) en %r no guardó frequency_trace_validation "
            "aunque el manifiesto exige validación por ventana -- CAL-07 ya no bloquea "
            "la calibración (ARC-167), solo se registra como advertencia",
            label, ref, freq_level_id,
        )
        return
    if frequency_trace is not None and (
        not isinstance(frequency_trace, Mapping)
        or not frequency_trace.get("accepted", False)
    ):
        message = (
            frequency_trace.get("message")
            if isinstance(frequency_trace, Mapping) else "resumen de traza malformado"
        )
        logger.warning(
            "%s: la traza de frecuencia de la calibración (%s) en %r no fue válida: %s "
            "-- CAL-07 ya no bloquea la calibración (ARC-167), solo se registra como advertencia",
            label, ref, freq_level_id, message or "traza de frecuencia inválida",
        )


def _measure_bw_and_flops_peak(
    manifest: Any,
    stream_ref: str,
    stream_kernel: KernelEntry,
    ert_ref: str,
    ert_kernel: KernelEntry,
    freq_level_id: str,
    *,
    environment_profile: Any,
    node_id: str | None,
    run_single: Callable[..., RunResult],
    apply_frequency: Callable[[Any, Any, Any], Any] | None = None,
) -> tuple[float, float, str, str]:
    """Runs STREAM (bandwidth) and ERT (FLOPs) once each at `freq_level_id`
    and returns (bw_pico, p_pico, stream_raw, ert_raw) already converted to
    SI base units (ARC-43)."""
    require_frequency_trace = bool(
        (getattr(manifest, "frequency_validation", None) or {}).get("require_per_window", False)
    )
    stream_result = run_single(
        stream_kernel, manifest, stream_ref, freq_level_id, 0,
        environment_profile=environment_profile, node_id=node_id,
        apply_frequency=apply_frequency,
    )
    if not stream_result.success:
        raise CalibrationError(
            f"CAL-02: la calibración de ancho de banda ({stream_ref}) no tuvo éxito en {freq_level_id!r}"
        )
    _require_valid_frequency_trace(
        stream_result, stream_ref, freq_level_id,
        label="CAL-07", require_per_window=require_frequency_trace,
    )

    ert_result = run_single(
        ert_kernel, manifest, ert_ref, freq_level_id, 0,
        environment_profile=environment_profile, node_id=node_id,
        apply_frequency=apply_frequency,
    )
    if not ert_result.success:
        raise CalibrationError(
            f"CAL-03: la calibración de FLOPs ({ert_ref}) no tuvo éxito en {freq_level_id!r}"
        )
    _require_valid_frequency_trace(
        ert_result, ert_ref, freq_level_id,
        label="CAL-07", require_per_window=require_frequency_trace,
    )

    stream_raw = stream_result.stdout_path.read_text(errors="replace")
    ert_raw = ert_result.stdout_path.read_text(errors="replace")

    # ARC-43: el regex captura la unidad nativa que la suite imprime (STREAM
    # reporta MB/s, no B/s) -- convertir a unidad SI base antes de guardar
    # como bw_pico_bytes_per_s/p_pico_flops_per_s, o i_ridge = p_pico/bw_pico
    # queda sesgado por el cociente entre los dos prefijos (p.ej. GFLOP/s
    # sobre MB/s da un i_ridge 1000x menor que el flops/byte real).
    bw_pico = _extract_metric(stream_kernel.bandwidth_stdout_pattern, stream_raw, label="BW_pico")
    bw_pico *= getattr(stream_kernel, "bandwidth_stdout_unit_multiplier", 1.0)
    p_pico = _extract_metric(ert_kernel.flops_stdout_pattern, ert_raw, label="P_pico")
    p_pico *= getattr(ert_kernel, "flops_stdout_unit_multiplier", 1.0)
    return bw_pico, p_pico, stream_raw, ert_raw


def run_calibration(
    manifest: Any,
    catalog: Mapping[str, KernelEntry],
    *,
    environment_profile: Any = None,
    node_id: str | None = None,
    run_single: Callable[..., RunResult] = runner.run_single,
    apply_frequency: Callable[[Any, Any, Any], Any] | None = None,
) -> RooflineCalibration:
    """CAL-01..05: run STREAM (bandwidth) and ERT (FLOPs) once each per
    frequency level, extract their peaks from stdout (never PMU counters),
    compute I_ridge, and check D03 in this same function so an implausible
    calibration can never silently reach windows.csv (CAL-04: D03 failing is
    a blocking exception).

    ARC-78: calibra UNA VEZ POR CADA nivel de `manifest.frequency_levels`,
    no una sola vez para toda la campaña. P_pico escala con el reloj de
    núcleo pero BW_pico casi no cambia (el reloj de memoria es un dominio
    aparte que este proyecto no toca -- ver
    docs/retoma/pacca/Consolidacion_Kernels_Dataset_Fase1.md sección 0), así
    que i_ridge = P_pico/BW_pico se desplaza con la frecuencia. Usar un solo
    i_ridge fijo para toda la campaña clasificaría mal cualquier ventana de
    un nivel distinto al que se calibró -- especialmente peligroso para
    kernels borderline como `rodinia_lud` (ARC-76/77), que puede cruzar la
    frontera compute/memory-bound solo por el cambio de reloj.

    Se persiste un archivo `roofline_calibration_<freq_level_id>.json` por
    nivel (`write_calibration`/`load_calibration`, ARC-78); `postprocess.py`
    elige el que corresponde a la ventana que está clasificando.

    El nivel `native_governor` (siempre exactamente uno, MAN-10) se procesa
    primero y actúa como referencia: se valida contra
    `manifest.hardware_datasheet` (D03 tal como existía antes de este
    cambio). Los demás niveles no tienen un pico declarado en el datasheet
    contra el cual comparar (el datasheet declara un único pico nominal, no
    uno por frecuencia) -- se valida en cambio que no midan más FLOPs/s que
    la referencia (`_check_plausibility_relative_to_reference`), lo único
    que se puede afirmar sin inventar un número.

    Objetos "manifest" sin `frequency_levels` (dobles de prueba,
    `SimpleNamespace`) caen en una ruta de compatibilidad de un solo nivel
    sintético -- comportamiento idéntico al de antes de ARC-78, ningún test
    existente necesitó cambiar su fixture por este motivo. Un `Manifest`
    real siempre declara `frequency_levels` (MAN-10 exige al menos uno), así
    que nunca toma esta ruta.
    """
    bandwidth_entry: tuple[str, KernelEntry] | None = None
    flops_entry: tuple[str, KernelEntry] | None = None
    for kernel_ref in manifest.calibration:
        entry = catalog[kernel_ref]
        if entry.reports_bandwidth_stdout:
            bandwidth_entry = (kernel_ref, entry)
        if entry.reports_flops_stdout:
            flops_entry = (kernel_ref, entry)
    if bandwidth_entry is None or flops_entry is None:
        raise CalibrationError(
            "CAL-02/CAL-03: manifest.calibration debe referenciar una entrada con "
            "reports_bandwidth_stdout y otra con reports_flops_stdout"
        )

    stream_ref, stream_kernel = bandwidth_entry
    ert_ref, ert_kernel = flops_entry

    levels = getattr(manifest, "frequency_levels", None) or (
        SimpleNamespace(id=_CALIBRATION_FREQ_LEVEL_ID, mode=None, fraction=None),
    )
    native_level = next((lvl for lvl in levels if getattr(lvl, "mode", None) == "native_governor"), levels[0])
    ordered_levels = [native_level] + [lvl for lvl in levels if lvl is not native_level]

    reference_calibration: RooflineCalibration | None = None
    for level in ordered_levels:
        can_apply_frequency = (
            apply_frequency is not None
            and environment_profile is not None
            and getattr(environment_profile, "frequency_write_capable", False)
        )
        if can_apply_frequency:
            # Conserva la transición explícita una vez por nivel (ARC-78).
            # _measure_bw_and_flops_peak también pasa el callable a cada
            # run_single para que el objetivo quede en metadata y la traza
            # FRQ-10 pueda validarse en STREAM y ERT por separado.
            applied_here = apply_frequency(manifest.cores.delegated_cpus, level, environment_profile)
            # ARC-161: espera activa opcional (manifest.frequency_settle) --
            # sin esto, ert_probe (corre en decenas de ms) puede medir bajo
            # el techo de frecuencia anterior en vez del nivel pedido
            # mientras el hardware todavía decae hacia el nuevo objetivo.
            freqctl.settle_if_configured(
                manifest.cores.delegated_cpus, applied_here, environment_profile,
                settle_config=getattr(manifest, "frequency_settle", None),
            )
        elif getattr(level, "mode", None) not in (None, "native_governor"):
            # RUN-09 (ARC-102): mismo principio que runner.run_single -- esta
            # función tiene su PROPIA lógica de aplicación de frecuencia
            # (arriba), separada de la de run_single, y nunca heredó su
            # guard: _measure_bw_and_flops_peak() llama a run_single() sin
            # pasarle apply_frequency, así que el guard de runner.py nunca
            # se activa aquí. Sin capacidad real de escritura, un nivel de
            # calibración "fixed" (no native_governor ni el nivel sintético
            # de compatibilidad con mode=None) no debe medirse en silencio a
            # la frecuencia nativa y persistirse como
            # roofline_calibration_<level.id>.json igual: eso desplazaría el
            # ridge point de ese nivel y contaminaría el etiquetado de toda
            # ventana clasificada contra él, sin ninguna señal de que pasó.
            raise CalibrationError(
                f"RUN-09: nivel de calibración {level.id!r} (mode={getattr(level, 'mode', None)!r}) "
                "requiere escritura real de frecuencia, pero no hay apply_frequency "
                "disponible o frequency_write_capable=False -- no se calibra "
                "silenciosamente a la frecuencia nativa."
            )

        bw_pico, p_pico, stream_raw, ert_raw = _measure_bw_and_flops_peak(
            manifest, stream_ref, stream_kernel, ert_ref, ert_kernel, level.id,
            environment_profile=environment_profile, node_id=node_id, run_single=run_single,
            apply_frequency=apply_frequency if can_apply_frequency else None,
        )
        if bw_pico <= 0:
            raise CalibrationError(f"CAL-04: BW_pico debe ser positivo, se obtuvo {bw_pico}")

        i_ridge = p_pico / bw_pico
        if level is native_level:
            passed, message = _check_plausibility(bw_pico, p_pico, manifest.hardware_datasheet)
        else:
            assert reference_calibration is not None  # native_level always processed first
            passed, message = _check_plausibility_relative_to_reference(bw_pico, p_pico, reference_calibration)

        calibration = RooflineCalibration(
            campaign_id=manifest.campaign_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            delegated_cpus=_format_cpu_list(manifest.cores.delegated_cpus),
            bw_pico_bytes_per_s=bw_pico,
            p_pico_flops_per_s=p_pico,
            i_ridge_flops_per_byte=i_ridge,
            stream_raw_output=stream_raw,
            ert_raw_output=ert_raw,
            plausibility_check_passed=passed,
            plausibility_message=message,
            freq_level_id=level.id if level.id != _CALIBRATION_FREQ_LEVEL_ID else "",
        )
        # Always persisted, even when the check fails: the artifact is the
        # evidence needed to investigate D03, load_calibration() is what
        # refuses to use it (CAL-06).
        write_calibration(calibration, manifest.output_dir)

        if level is native_level:
            reference_calibration = calibration
        if not passed:
            raise CalibrationError(message)

    assert reference_calibration is not None
    return reference_calibration


def run_gpu_calibration(
    manifest: Any,
    catalog: Mapping[str, KernelEntry],
    *,
    environment_profile: Any = None,
    node_id: str | None = None,
    run_single: Callable[..., RunResult] = runner.run_single,
    apply_frequency: Callable[[Any, Any, Any], Any] | None = None,
    apply_gpu_frequency: Callable[[Any, Any], Any] | None = None,
) -> dict[str, RooflineCalibration]:
    """ARC-80: calibra el ridge point de GPU -- fp32 y fp64 por separado
    (ARC-76: la mayoría de Rodinia es FP32, lavaMD es FP64, no hay un solo
    ridge universal) -- una vez por cada nivel de `manifest.frequency_levels`
    (mismo principio que `run_calibration()` ya aplica a CPU, ARC-78).

    Las fuentes de calibración se declaran explícitamente en
    `manifest.gpu["calibration"]` (lista de kernel_ref, mismo patrón que
    `manifest.calibration` para CPU) -- nunca se infieren escaneando el
    catálogo por `role=="calibration" and device=="gpu"`, porque
    `gpu_dgemm_calibration` también es un kernel de calibración de GPU que
    reporta FLOPs (referencia informativa de cuBLAS, ARC-76) y colisionaría
    con `gpu_ert_probe_fp64` si se buscara por esos dos criterios solamente.

    Si `manifest.gpu` no declara `calibration` (campañas que no incluyen
    kernels de GPU), no hace nada y devuelve `{}` -- a diferencia de
    `run_calibration()` (CPU), esta calibración es opcional.
    """
    gpu_config = getattr(manifest, "gpu", None) or {}
    gpu_calibration_refs = gpu_config.get("calibration") if isinstance(gpu_config, Mapping) else None
    if not gpu_calibration_refs:
        return {}

    bandwidth_entry: tuple[str, KernelEntry] | None = None
    flops_entries: dict[str, tuple[str, KernelEntry]] = {}
    for kernel_ref in gpu_calibration_refs:
        if kernel_ref not in catalog:
            raise CalibrationError(
                f"CAL-GPU-00: manifest.gpu.calibration referencia un kernel_ref "
                f"inexistente: {kernel_ref!r}"
            )
        entry = catalog[kernel_ref]
        if entry.reports_bandwidth_stdout:
            bandwidth_entry = (kernel_ref, entry)
        if entry.reports_flops_stdout:
            precision = entry.gpu_precision
            if precision not in ("fp32", "fp64"):
                raise CalibrationError(
                    f"CAL-GPU-00: {kernel_ref!r} en manifest.gpu.calibration reporta "
                    "FLOPs pero no declara gpu_precision='fp32'/'fp64' válido"
                )
            flops_entries[precision] = (kernel_ref, entry)
    if bandwidth_entry is None or "fp32" not in flops_entries or "fp64" not in flops_entries:
        raise CalibrationError(
            "CAL-GPU-01: manifest.gpu.calibration debe referenciar un kernel con "
            "reports_bandwidth_stdout y dos con reports_flops_stdout "
            "(uno por cada gpu_precision: fp32 y fp64)"
        )
    stream_ref, stream_kernel = bandwidth_entry

    # ARC-129: manifest.gpu_frequency_levels, cuando existe, es el ÚNICO eje
    # que importa para calibrar el ridge de GPU -- el ridge (P_pico_gpu/
    # BW_pico_gpu) depende del reloj de GPU, no del de CPU (mismo principio
    # que el comentario de más abajo ya establecía para el pineo de núcleos).
    # Con un producto cartesiano CPU x GPU real, repetir esta calibración
    # una vez por cada nivel de CPU además de GPU no mediría nada físicamente
    # distinto -- solo multiplicaría el costo sin agregar señal. Ausente
    # (None, todo manifiesto anterior a este cambio) cae al comportamiento
    # de siempre: un ridge por cada nivel de frequency_levels (CPU).
    gpu_levels = getattr(manifest, "gpu_frequency_levels", None)
    levels = gpu_levels or getattr(manifest, "frequency_levels", None) or (
        SimpleNamespace(id=_CALIBRATION_FREQ_LEVEL_ID, mode=None, fraction=None),
    )
    native_level = next((lvl for lvl in levels if getattr(lvl, "mode", None) == "native_governor"), levels[0])
    ordered_levels = [native_level] + [lvl for lvl in levels if lvl is not native_level]

    # ARC-102: a diferencia de run_calibration() (eje CPU puro), esta
    # calibración de GPU no exige capacidad de escritura de frecuencia de
    # CPU para un nivel "fixed" -- el pineo de núcleos aquí es incidental
    # (mide FLOPs/BW de la GPU, no de la CPU), y test_arc87_run_gpu_
    # calibration_fija_el_reloj_de_gpu_por_nivel ya fija ese contrato
    # deliberadamente (ARC-87: solo el reloj de GPU es el que determina si
    # el "ridge point por nivel" mide algo distinto entre niveles). El guard
    # RUN-09 real de esta función va en el eje de GPU, más abajo.
    is_fixed_level = lambda lvl: getattr(lvl, "mode", None) not in (None, "native_governor")  # noqa: E731

    reference: dict[str, RooflineCalibration] = {}
    for level in ordered_levels:
        # ARC-129: si gpu_levels existe, `level` recorre gpu_frequency_levels
        # (no frequency_levels) -- pinnear CPU contra ESE id sería un error
        # de eje (dos listas de ids distintas), y físicamente incidental de
        # todas formas (comentario de arriba). Sin gpu_frequency_levels
        # declarado, comportamiento idéntico al de siempre.
        if (
            gpu_levels is None
            and apply_frequency is not None
            and environment_profile is not None
            and getattr(environment_profile, "frequency_write_capable", False)
        ):
            apply_frequency(manifest.cores.delegated_cpus, level, environment_profile)

        # ARC-87: sin fijar también el reloj de GPU en cada nivel, un
        # "ridge point por nivel" para GPU no mediría nada distinto entre
        # niveles -- el reloj físico de la GPU seguiría siendo el mismo en
        # los 6 (REF+F0..F4), solo con una etiqueta de nivel distinta.
        if (
            apply_gpu_frequency is not None
            and environment_profile is not None
            and getattr(environment_profile, "gpu_frequency_write_capable", False)
        ):
            apply_gpu_frequency(level, environment_profile)
        elif is_fixed_level(level):
            # RUN-09 (ARC-102): sin esto, un nivel "fixed" sin
            # gpu_frequency_write_capable mediría el ridge point de GPU al
            # mismo reloj nativo en los 6 niveles -- el problema exacto que
            # el comentario de arriba explica, ocurriendo en silencio.
            raise CalibrationError(
                f"RUN-09: nivel de calibración {level.id!r} (mode={getattr(level, 'mode', None)!r}) "
                "requiere escritura real de frecuencia de GPU, pero no hay apply_gpu_frequency "
                "disponible o gpu_frequency_write_capable=False."
            )

        # BW se mide una sola vez por nivel, compartida entre fp32/fp64 --
        # el reloj de memoria no es parte del espacio DVFS de este proyecto
        # (ARC-77, sección 1 de Diseno_Politica_DVFS_CPU_GPU.md), así que no
        # hay ninguna razón física para que cambie entre precisiones.
        stream_result = run_single(
            stream_kernel, manifest, stream_ref, level.id, 0,
            environment_profile=environment_profile, node_id=node_id,
        )
        if not stream_result.success:
            raise CalibrationError(
                f"CAL-GPU-02: la calibración de ancho de banda GPU ({stream_ref}) "
                f"no tuvo éxito en {level.id!r}"
            )
        stream_raw = stream_result.stdout_path.read_text(errors="replace")
        bw_pico = _extract_metric(stream_kernel.bandwidth_stdout_pattern, stream_raw, label="BW_pico_gpu")
        bw_pico *= getattr(stream_kernel, "bandwidth_stdout_unit_multiplier", 1.0)
        if bw_pico <= 0:
            raise CalibrationError(f"CAL-GPU-02: BW_pico_gpu debe ser positivo, se obtuvo {bw_pico}")

        for precision, (ert_ref, ert_kernel) in flops_entries.items():
            ert_result = run_single(
                ert_kernel, manifest, ert_ref, level.id, 0,
                environment_profile=environment_profile, node_id=node_id,
            )
            if not ert_result.success:
                raise CalibrationError(
                    f"CAL-GPU-03: la calibración de FLOPs GPU ({ert_ref}, {precision}) "
                    f"no tuvo éxito en {level.id!r}"
                )
            ert_raw = ert_result.stdout_path.read_text(errors="replace")
            p_pico = _extract_metric(ert_kernel.flops_stdout_pattern, ert_raw, label=f"P_pico_gpu_{precision}")
            p_pico *= getattr(ert_kernel, "flops_stdout_unit_multiplier", 1.0)
            if p_pico <= 0:
                raise CalibrationError(f"CAL-GPU-03: P_pico_gpu debe ser positivo, se obtuvo {p_pico}")

            # ARC-80: no existe un "datasheet" de GPU declarado en el
            # manifiesto (D03/CAL-04 son específicos de CPU) -- inventar un
            # rango ±40% sin una fuente real sería peor que no chequear
            # nada. El único chequeo honesto disponible es positividad, ya
            # aplicado arriba a bw_pico/p_pico.
            calibration = RooflineCalibration(
                campaign_id=manifest.campaign_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                delegated_cpus=_format_cpu_list(manifest.cores.delegated_cpus),
                bw_pico_bytes_per_s=bw_pico,
                p_pico_flops_per_s=p_pico,
                i_ridge_flops_per_byte=p_pico / bw_pico,
                stream_raw_output=stream_raw,
                ert_raw_output=ert_raw,
                plausibility_check_passed=True,
                plausibility_message=(
                    "CAL-GPU: sin datasheet de GPU declarado en el manifiesto -- "
                    "solo se verificó que bw_pico/p_pico sean positivos."
                ),
                freq_level_id=level.id if level.id != _CALIBRATION_FREQ_LEVEL_ID else "",
                gpu_precision=precision,
            )
            write_calibration(calibration, manifest.output_dir)
            if level is native_level:
                reference[precision] = calibration

    return reference


def _read_final_cpu_counters(run_dir: Path) -> tuple[int, int, int, int] | None:
    """Last CPU row of samples.csv: perf counters are cumulative since
    IOC_RESET, so the last sample approximates the whole run's totals."""
    samples_path = run_dir / "samples.csv"
    try:
        with samples_path.open(newline="", encoding="utf-8") as samples_file:
            reader = csv.DictReader(samples_file)
            last_cpu_row = None
            for row in reader:
                if row.get("tag") == "CPU":
                    last_cpu_row = row
            if last_cpu_row is None:
                return None
            return (
                int(last_cpu_row["instructions"]),
                int(last_cpu_row["cycles"]),
                int(last_cpu_row["cache_references"]),
                int(last_cpu_row["cache_misses"]),
            )
    except (OSError, KeyError, ValueError):
        return None


def _run_metrics(result: RunResult) -> dict[str, float] | None:
    counters = _read_final_cpu_counters(result.run_dir)
    if counters is None:
        return None
    instructions, cycles, cache_references, cache_misses = counters
    if instructions <= 0 or cycles <= 0 or result.elapsed_seconds <= 0:
        return None
    return {
        "ipc": instructions / cycles,
        "ips": instructions / result.elapsed_seconds,
        "mpki": (cache_misses / instructions) * 1000.0,
        "miss_rate": (cache_misses / cache_references) if cache_references > 0 else 0.0,
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _cv_pct(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance ** 0.5 / mean) * 100.0


def build_calibration_references(
    calibration_runs: Sequence[RunResult],
    node_id: str,
    *,
    cv_threshold_pct: float = DEFAULT_CV_THRESHOLD_PCT,
) -> CalibrationReferences:
    """CAL-09/CAL-10: >=5 repetitions of a reference kernel -> P95 of
    IPC/IPS/MPKI/MissRate plus a stability check.

    The dataclass carries a single cv_pct/accepted pair for four metrics;
    cv_pct here is the worst (max) of the four per-metric coefficients of
    variation, so instability in any one signal disqualifies the reference
    instead of being averaged away.
    """
    if len(calibration_runs) < MIN_REFERENCE_REPETITIONS:
        raise CalibrationError(
            f"CAL-09: build_calibration_references requiere >={MIN_REFERENCE_REPETITIONS} "
            f"repeticiones, se dieron {len(calibration_runs)}"
        )

    metrics_by_key: dict[str, list[float]] = {"ipc": [], "ips": [], "mpki": [], "miss_rate": []}
    for result in calibration_runs:
        if not result.success:
            raise CalibrationError(f"CAL-09: la repetición {result.run_id} de referencia no tuvo éxito")
        metrics = _run_metrics(result)
        if metrics is None:
            raise CalibrationError(
                f"CAL-09: la repetición {result.run_id} no tiene contadores CPU utilizables en samples.csv"
            )
        for key, value in metrics.items():
            metrics_by_key[key].append(value)

    cv_pct = max(_cv_pct(values) for values in metrics_by_key.values())
    return CalibrationReferences(
        node_id=node_id,
        ipc_p95=_percentile(metrics_by_key["ipc"], 95),
        ips_p95=_percentile(metrics_by_key["ips"], 95),
        mpki_p95=_percentile(metrics_by_key["mpki"], 95),
        miss_rate_p95=_percentile(metrics_by_key["miss_rate"], 95),
        repetitions=len(calibration_runs),
        cv_pct=cv_pct,
        accepted=cv_pct <= cv_threshold_pct,
    )


def run_calibration_references(
    entry: KernelEntry,
    manifest: Any,
    kernel_ref: str,
    *,
    node_id: str,
    repetitions: int = MIN_REFERENCE_REPETITIONS,
    environment_profile: Any = None,
    cv_threshold_pct: float = DEFAULT_CV_THRESHOLD_PCT,
    run_single: Callable[..., RunResult] = runner.run_single,
    apply_frequency: Callable[[Any, Any, Any], Any] | None = None,
) -> CalibrationReferences:
    """Runs the reference kernel `repetitions` times and delegates to
    build_calibration_references(). Persists calibration_references.json
    regardless of `accepted` (CAL-10/D04 is a warning, not a hard stop).

    ARC-94: `run_calibration()`/`run_gpu_calibration()` barren un nivel de
    `frequency_levels` por vez y terminan con el ÚLTIMO nivel fixed
    todavía aplicado (típicamente F4, el último en orden del manifiesto) --
    sin volver a fijar el governor nativo aquí, antes de medir, estas
    repeticiones de referencia (que fijan IPC/MPKI P95 para
    ipc_relative/mpki_relative de TODA la campaña) quedarían contaminadas
    por esa frecuencia pinneada. `apply_frequency`, si se da, vuelve a
    aplicar el nivel `native_governor` del manifiesto antes de la primera
    repetición -- mismo contrato de 3 argumentos (cpus, level, env) que
    `run_calibration` ya usa, con el snapshot original ligado por el
    llamador vía `functools.partial` (ver campaign.py)."""
    if repetitions < MIN_REFERENCE_REPETITIONS:
        raise ValueError(f"CAL-09: repetitions debe ser >={MIN_REFERENCE_REPETITIONS}")

    levels = getattr(manifest, "frequency_levels", None)
    if (
        apply_frequency is not None
        and environment_profile is not None
        and getattr(environment_profile, "frequency_write_capable", False)
        and levels
    ):
        native_level = next((lvl for lvl in levels if getattr(lvl, "mode", None) == "native_governor"), None)
        if native_level is not None:
            apply_frequency(manifest.cores.delegated_cpus, native_level, environment_profile)

    runs = [
        run_single(
            entry, manifest, kernel_ref, _CALIBRATION_FREQ_LEVEL_ID, repetition,
            environment_profile=environment_profile, node_id=node_id,
        )
        for repetition in range(1, repetitions + 1)
    ]
    require_frequency_trace = bool(
        (getattr(manifest, "frequency_validation", None) or {}).get("require_per_window", False)
    )
    for result in runs:
        _require_valid_frequency_trace(
            result, kernel_ref, _CALIBRATION_FREQ_LEVEL_ID,
            label="CAL-07", require_per_window=require_frequency_trace,
        )
    references = build_calibration_references(runs, node_id, cv_threshold_pct=cv_threshold_pct)
    write_calibration_references(references, manifest.output_dir)
    if not references.accepted:
        logger.warning(
            "CAL-10/D04: calibration_references cv_pct=%.2f%% supera el umbral %.2f%% (node_id=%s)",
            references.cv_pct, cv_threshold_pct, node_id,
        )
    return references


def write_calibration_references(references: CalibrationReferences, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "calibration_references.json"
    with path.open("w", encoding="utf-8") as references_file:
        json.dump(asdict(references), references_file, indent=2, sort_keys=True)
        references_file.write("\n")
    return path


def load_calibration_references(output_dir: str | Path) -> CalibrationReferences:
    path = Path(output_dir) / "calibration_references.json"
    with path.open(encoding="utf-8") as references_file:
        return CalibrationReferences(**json.load(references_file))
