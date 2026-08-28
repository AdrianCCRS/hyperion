from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .catalog import KernelEntry, verify_binary

# VAL-XX ids refer to docs/retoma/Guia_Maestra_Fase1_DVFS.md section 12.10.
# The factor_id values (I04, C02, C03, E06-E08, I07, D03) are the same ones
# preflight.py checks before a campaign starts; validation.py re-checks the
# run-level ones that can only be known AFTER a run finished (defense in
# depth: a preflight pass does not guarantee the condition held throughout
# the run).


@dataclass(frozen=True)
class Verdict:
    """Outcome of validating one completed run."""

    accepted: bool
    factor_id: str | None
    message: str


def validate_cpu_frequency_trace(
    samples_path: str | Path,
    *,
    require_per_window: bool,
    expected_khz: int | None,
    tolerance_fraction: float | None,
    expected_cpu_count: int | None = None,
    grace_seconds: float = 0.0,
    tail_grace_seconds: float = 0.0,
) -> tuple[Verdict, dict[str, Any]]:
    """E01/FRQ-10: valida el reloj medido en los ticks reales del PMU.

    Para REF no existe un objetivo numérico: solo se exige que la lectura
    por ventana esté presente. Para un nivel fixed, todas las lecturas de
    todos los CPUs delegados deben caer dentro de la tolerancia declarada
    respecto al valor aplicado. ``expected_cpu_count=None`` conserva el
    contrato legado de validar solo ``scaling_cur_freq_khz``; producción
    pasa siempre la cantidad de CPUs delegados y exige la columna multi-CPU
    de ARC-145.

    ARC-166: ``grace_seconds`` excluye de la comprobación de TOLERANCIA (no
    de los chequeos estructurales de integridad -- ``missing``/
    ``count_mismatches``/``primary_mismatches`` siguen exigidos desde la
    primera muestra) las lecturas tomadas dentro de esa ventana desde el
    primer tick CPU de la traza. Confirmado reproducible en `paccaA100`
    (2/2 corridas idénticas): un CPU delegado específico tarda ~10-11ms en
    engancharse al candado justo después de que arranca el kernel real,
    pese a que el warm-up previo a la corrida (ARC-165) ya había confirmado
    el asentamiento de los 6 CPUs -- mismo patrón que el `warmup_seconds`
    ya establecido a nivel de kernel para excluir el arranque de las
    ventanas de calidad PMU, aplicado aquí a la traza de frecuencia.
    Default 0.0 preserva el comportamiento anterior byte a byte.

    ARC-169: ``tail_grace_seconds`` excluye, de la misma comprobación de
    TOLERANCIA (nunca de los chequeos estructurales), las lecturas dentro
    de esa ventana antes del ÚLTIMO tick CPU de la traza. Confirmado en la
    campaña final (`pacca_cpu_final_attempt03_20260820`): kernels con una
    única región paralela larga (sin las barreras internas repetidas de
    `stream_official`) pueden diluir el mismo promedio APERF/MPERF
    dependiente de actividad justo en la unión final de esa región, cuando
    un hilo termina su parte del trabajo antes que sus pares y queda
    ocioso mientras los espera -- mismo mecanismo que ARC-167 documentó
    para `stream_official`, pero concentrado al final en vez de disperso
    entre fases. Default 0.0 preserva el comportamiento anterior byte a
    byte.

    ARC-174: el ``Verdict`` devuelto por esta función ya NO falla por una
    desviación de tolerancia -- una sola muestra fuera de rango invalidaba
    hasta ahora TODA la corrida (miles de ventanas descartadas por una),
    pese a que la causa física documentada en ARC-167/169 es dilución
    APERF/MPERF localizada, no una falla real de actuación. El ``Verdict``
    ahora solo puede fallar por integridad ESTRUCTURAL (traza vacía,
    lecturas ausentes, cantidad de CPUs distinta de la esperada,
    incoherencia entre la columna escalar y el primer CPU de la columna
    multi-CPU) -- exactamente las cuatro causas ya cubiertas por el bloque
    ``require_per_window`` de más abajo, sin cambios. El diagnóstico de
    tolerancia (``mismatched_samples``, rango observado, cobertura) se
    conserva íntegro en ``summary`` para quien todavía lo consulte (p.ej.
    la advertencia CAL-07 de ``calibration.py``), pero ya no determina
    ``accepted``. ``summary["structural_valid"]`` y
    ``summary["tolerance_all_within"]`` separan explícitamente ambos ejes
    para que ningún caller confunda uno con otro leyendo solo
    ``accepted``. La clasificación POR VENTANA que reemplaza este gate de
    tolerancia vive en ``classify_frequency_window()`` (usada por
    ``postprocess.build_windows()``), no aquí -- esta función sigue siendo
    el único gate de integridad estructural, evaluado sobre la corrida
    completa antes de que exista windows.csv.
    """
    with open(samples_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("tag") == "CPU"]

    observed: list[int] = []
    missing = 0
    count_mismatches = 0
    primary_mismatches = 0
    spreads: list[int] = []
    grace_ns = grace_seconds * 1e9
    tail_grace_ns = tail_grace_seconds * 1e9
    t0_ns: int | None = None
    t_end_ns: int | None = None
    if tail_grace_ns > 0:
        for row in reversed(rows):
            try:
                t_end_ns = int(row.get("timestamp_ns"))
                break
            except (TypeError, ValueError):
                continue
    excluded_by_grace = 0
    for row in rows:
        try:
            ts_ns: int | None = int(row.get("timestamp_ns"))
        except (TypeError, ValueError):
            ts_ns = None
        if t0_ns is None and ts_ns is not None:
            t0_ns = ts_ns
        within_grace = (
            grace_ns > 0 and ts_ns is not None and t0_ns is not None
            and (ts_ns - t0_ns) < grace_ns
        )
        within_tail_grace = (
            tail_grace_ns > 0 and ts_ns is not None and t_end_ns is not None
            and (t_end_ns - ts_ns) < tail_grace_ns
        )
        within_grace = within_grace or within_tail_grace

        if expected_cpu_count is None:
            raw = row.get("scaling_cur_freq_khz")
            try:
                value = int(raw)
                if value <= 0:
                    raise ValueError
                if within_grace:
                    excluded_by_grace += 1
                else:
                    observed.append(value)
            except (TypeError, ValueError):
                missing += 1
            continue

        raw_all = row.get("scaling_cur_freq_khz_all")
        parts = raw_all.split(";") if raw_all else []
        if len(parts) != expected_cpu_count:
            count_mismatches += 1

        row_observed: list[int] = []
        for cpu_index in range(expected_cpu_count):
            raw = parts[cpu_index] if cpu_index < len(parts) else None
            try:
                value = int(raw)
                if value <= 0:
                    raise ValueError
                row_observed.append(value)
                if within_grace:
                    excluded_by_grace += 1
                else:
                    observed.append(value)
            except (TypeError, ValueError):
                missing += 1

        if len(row_observed) >= 2 and not within_grace:
            spreads.append(max(row_observed) - min(row_observed))

        # El launcher escribe el mismo valor en la columna escalar y en el
        # slot 0. Si difieren, el archivo no conserva una traza internamente
        # coherente y no se puede elegir silenciosamente una de las dos.
        if parts:
            try:
                primary = int(row.get("scaling_cur_freq_khz"))
                first = int(parts[0])
                if primary <= 0 or first <= 0 or primary != first:
                    primary_mismatches += 1
            except (TypeError, ValueError):
                primary_mismatches += 1

    summary = {
        "cpu_samples": len(rows),
        "observed_samples": len(observed),
        "missing_samples": missing,
        "expected_cpu_count": expected_cpu_count,
        "cpu_count_mismatch_samples": count_mismatches,
        "primary_mismatch_samples": primary_mismatches,
        "expected_khz": expected_khz,
        "tolerance_fraction": tolerance_fraction,
        "observed_min_khz": min(observed) if observed else None,
        "observed_max_khz": max(observed) if observed else None,
        "observed_spread_max_khz": max(spreads) if spreads else None,
        "grace_seconds": grace_seconds,
        "tail_grace_seconds": tail_grace_seconds,
        "excluded_by_grace_samples": excluded_by_grace,
    }
    if require_per_window and (not rows or missing > 0 or count_mismatches > 0 or primary_mismatches > 0):
        summary["structural_valid"] = False
        summary["tolerance_all_within"] = None
        return Verdict(
            False, "E01",
            "traza de frecuencia incompleta/incoherente: "
            f"{missing} lecturas ausentes, {count_mismatches} ventanas con cantidad de CPUs distinta de "
            f"{expected_cpu_count}, {primary_mismatches} ventanas con CPU representativo inconsistente "
            f"({len(rows)} ventanas CPU)",
        ), summary

    summary["structural_valid"] = True

    if expected_khz is not None:
        if tolerance_fraction is None:
            # ARC-174: config incompleta (nivel fixed sin tolerance_fraction
            # declarada) no es lo mismo que integridad estructural de la
            # traza -- pero sigue siendo un rechazo total legítimo, porque
            # sin tolerance_fraction no hay forma de evaluar NADA (ni por
            # corrida ni por ventana) contra este nivel.
            summary["tolerance_all_within"] = None
            return Verdict(False, "E01", "nivel fixed sin tolerance_fraction declarada"), summary
        if rows and not observed:
            # ARC-166/ARC-169: toda la corrida cayó dentro de grace_seconds
            # y/o tail_grace_seconds -- no hay ninguna muestra fuera de esas
            # ventanas que confirme el candado. Silenciar esto (mismatches=[]
            # sobre una lista vacía "pasa" por vacuidad) reintroduciría
            # exactamente el riesgo que E01 existe para prevenir: aceptar
            # una corrida sin ninguna confirmación real de frecuencia. Este
            # caso degenerado (100% de la traza excluida por configuración
            # de gracia) sigue rechazando la corrida entera -- a diferencia
            # de la tolerancia por muestra (ARC-174), aquí no hay NINGÚN
            # dato con el que construir una clasificación por ventana.
            summary["tolerance_all_within"] = None
            return Verdict(
                False, "E01",
                f"toda la traza ({len(rows)} ventanas CPU) cae dentro de grace_seconds="
                f"{grace_seconds}s/tail_grace_seconds={tail_grace_seconds}s -- ninguna muestra "
                "disponible fuera de esas ventanas para confirmar el candado",
            ), summary
        tolerance_khz = expected_khz * tolerance_fraction
        summary["tolerance_khz_effective"] = tolerance_khz
        mismatches = [value for value in observed if abs(value - expected_khz) > tolerance_khz]
        summary["mismatched_samples"] = len(mismatches)
        # ARC-174: ya no se rechaza aquí -- ver docstring. El diagnóstico se
        # conserva; classify_frequency_window() decide, por ventana, si cada
        # una individualmente es utilizable.
        summary["tolerance_all_within"] = len(mismatches) == 0
    else:
        summary["mismatched_samples"] = 0
        summary["tolerance_all_within"] = None

    return Verdict(True, None, "ok"), summary


@dataclass(frozen=True)
class FrequencyWindowClassification:
    """ARC-174: veredicto de frecuencia POR VENTANA -- el reemplazo de
    tolerancia agregada de ``validate_cpu_frequency_trace()``. Ninguno de
    estos estados puede, por sí solo, rechazar la corrida completa (eso
    sigue siendo exclusivo del chequeo estructural de arriba)."""

    status: str | None
    outlier_cpu_count: int | None
    in_tolerance_cpu_count: int | None
    below_tolerance_cpu_count: int | None
    above_tolerance_cpu_count: int | None
    min_khz: int | None
    max_khz: int | None
    max_relative_error: float | None


def classify_frequency_window(
    raw_scaling_cur_freq_khz_all: str | None,
    *,
    is_native_governor: bool,
    expected_khz: int | None,
    tolerance_fraction: float | None,
    within_grace: bool,
) -> FrequencyWindowClassification:
    """ARC-174: clasifica UNA ventana (no la corrida) usando la misma
    columna multi-CPU (``scaling_cur_freq_khz_all``) que ya se captura hoy
    -- ningún dato nuevo, solo una decisión más fina sobre datos que ya
    existían. Reemplaza el gate agregado de tolerancia de
    ``validate_cpu_frequency_trace()`` (ARC-174).

    El spread (max-min) NO es la señal usada aquí -- puede ser cero aunque
    los 6 CPUs delegados estén igualmente lejos del objetivo (hallazgo
    explícito antes de implementar esto). Cada CPU se evalúa individualmente
    contra ``expected_khz``/``tolerance_fraction``.

    Cuatro estados posibles:

    - ``"not_applicable_native"``: SOLO para REF (``is_native_governor``
      explícito, nunca inferido de ``expected_khz is None`` -- ese campo
      también es None cuando la actuación de frecuencia está desactivada
      por completo, un caso distinto que no debe leerse como REF).
    - ``"observation_unverified_grace"``: la ventana cae dentro de
      ``grace_seconds``/``tail_grace_seconds`` (decidido por el caller vía
      ``within_grace``, quien conoce ``t_start_ns``/``t_end_ns`` del run) --
      tiene precedencia sobre ``observation_unreliable`` incluso si además
      hay desviación de tolerancia.
    - ``"observation_unreliable"``: ningun CPU confirma el objetivo, o al
      menos uno lo excede por encima de la tolerancia.
    - ``"valid"``: al menos un CPU confirma el objetivo y ninguno lo excede.
      Lecturas inferiores no invalidan por si solas: en paccaA100
      ``scaling_cur_freq`` es APERF/MPERF dependiente de actividad y los
      nucleos ociosos reportan menos aunque min=max este correctamente
      escrito. Job 6696 demostro el caso: GEMM N64/N96 ocupa 1-2 de los 6
      nucleos; exigir seis lecturas iguales rechazo 18 corridas con miles de
      ventanas PMU sanas. Un sobre-reloj si invalida porque no se explica por
      ociosidad; todas las lecturas bajas tambien invalidan porque ninguna
      confirma actuacion bajo carga.

    Estado ``None`` (fail-closed, ARC-174): cuando falta ``expected_khz``/
    ``tolerance_fraction`` (validación desactivada o config incompleta) o
    no hay ninguna lectura de CPU utilizable en esta ventana -- nunca se
    confunde con REF ni se asume "válida" por defecto.
    """
    values: list[int] = []
    if raw_scaling_cur_freq_khz_all:
        for part in raw_scaling_cur_freq_khz_all.split(";"):
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                values.append(value)

    min_khz = min(values) if values else None
    max_khz = max(values) if values else None

    if is_native_governor:
        return FrequencyWindowClassification(
            "not_applicable_native", None, None, None, None, min_khz, max_khz, None,
        )

    if expected_khz is None or tolerance_fraction is None or not values:
        return FrequencyWindowClassification(None, None, None, None, None, min_khz, max_khz, None)

    max_relative_error = max(abs(value - expected_khz) / expected_khz for value in values)
    tolerance_khz = expected_khz * tolerance_fraction
    lower_khz = expected_khz - tolerance_khz
    upper_khz = expected_khz + tolerance_khz
    below_tolerance_cpu_count = sum(1 for value in values if value < lower_khz)
    above_tolerance_cpu_count = sum(1 for value in values if value > upper_khz)
    in_tolerance_cpu_count = len(values) - below_tolerance_cpu_count - above_tolerance_cpu_count
    outlier_cpu_count = below_tolerance_cpu_count + above_tolerance_cpu_count

    if within_grace:
        return FrequencyWindowClassification(
            "observation_unverified_grace", outlier_cpu_count, in_tolerance_cpu_count,
            below_tolerance_cpu_count, above_tolerance_cpu_count,
            min_khz, max_khz, max_relative_error,
        )

    status = (
        "valid"
        if in_tolerance_cpu_count > 0 and above_tolerance_cpu_count == 0
        else "observation_unreliable"
    )
    return FrequencyWindowClassification(
        status, outlier_cpu_count, in_tolerance_cpu_count,
        below_tolerance_cpu_count, above_tolerance_cpu_count,
        min_khz, max_khz, max_relative_error,
    )


def validate_run(
    run_result: Any,
    kernel_entry: KernelEntry,
    *,
    foreign_processes: Any = None,
    governor: Any = None,
    external_load: Any = None,
    run_id_seen: Iterable[str] = (),
    node_id: str | None = None,
) -> Verdict:
    """Accept/reject one completed run (RunResult from runner.run_single).

    VAL-07: deterministic order — I04 first, then C02/C03, then E06-E08,
    then the rest (I07 here; D03 is campaign-wide and handled separately by
    validate_campaign_calibration, VAL-05).

    `foreign_processes`/`governor`/`external_load` are the CheckResult
    objects preflight.check_foreign_processes/check_governor/
    check_external_load already know how to produce (E06/E07/E08); this
    function does not read sysfs or /proc itself, it only applies their
    verdicts in the required order. Any of the three may be omitted (None)
    when the caller did not re-check that condition for this run.

    VAL-08: per-window quality flags (I01 no_freq_reading, I02 low
    running_ratio, I03 sampling jitter, warmup_excluded, intensity_undefined)
    are never consulted here -- a single bad window never flips this
    verdict, only its own quality_status in windows.csv (postprocess.py).

    ARC-94: this is only the FIRST of two acceptance stages. This function
    can only see run-level metadata that exists before windows.csv is
    built, so it can reject a run that never produced any telemetry at all
    (I04/C02/C03/E06-E08/I07) -- it CANNOT reject a run that ran fine but
    produced too few usable windows or no label at all, because
    windows.csv does not exist yet at this point. The caller (campaign.py)
    must run postprocess.py after an accepted verdict from THIS function,
    then call validate_windows() on the result for the final accept/reject
    decision -- an accepted Verdict from validate_run() alone is
    provisional, not final.
    """
    metadata = run_result.metadata

    # I04 (VAL-01): samples_collected==0 or push_retries>0 -> immediate
    # rejection, no window-level repair is possible for a run that never
    # produced usable telemetry.
    samples_collected = metadata.get("samples_collected")
    push_retries = metadata.get("push_retries")
    if samples_collected == 0:
        return Verdict(False, "I04", "samples_collected=0: la corrida no produjo telemetría")
    if push_retries is not None and push_retries > 0:
        return Verdict(False, "I04", f"push_retries={push_retries}: el ring de telemetría se llenó")

    # C02 (VAL-03): the binary actually executed still matches the catalog.
    # runner.run_single() already re-verifies this before launching
    # (CAT-07); this is a second, independent check after the fact.
    if not verify_binary(kernel_entry, node_id):
        return Verdict(False, "C02", f"checksum de {kernel_entry.exec_path!r} no coincide con el catálogo")

    # C03 (VAL-04): success_check against the real result, already applied
    # by runner.run_single() and stored on RunResult.success.
    if not run_result.success:
        return Verdict(False, "C03", "success_check no se cumplió")

    # E01/FRQ-10 (ARC-138): runner.py calcula este resultado sobre todas
    # las lecturas reales del colector y lo conserva en metadata. Rechazar
    # aquí, en vez de lanzar una excepción desde runner, garantiza que
    # campaign.py escriba verdict.json y preserve los crudos de la corrida.
    frequency_trace = metadata.get("frequency_trace_validation")
    if frequency_trace is not None and not frequency_trace.get("accepted", False):
        return Verdict(
            False,
            frequency_trace.get("factor_id") or "E01",
            frequency_trace.get("message") or "traza de frecuencia inválida",
        )

    # E06-E08 (contamination / governor drift / external load), in that
    # order, whichever ones the caller supplied.
    for check in (foreign_processes, governor, external_load):
        if check is not None and not check.passed:
            return Verdict(False, check.factor_id, check.message)

    # I07 (VAL-02): run_id duplicate, even if preflight's own I07 check
    # (check_run_id_unique) somehow missed it (e.g. a resumed campaign that
    # replays combinations).
    if run_result.run_id in set(run_id_seen):
        return Verdict(False, "I07", f"run_id duplicado: {run_result.run_id}")

    return Verdict(True, None, "ok")


# ARC-129: mismo piso de ruido ya establecido y justificado en
# scripts/pacca/measure_warmup.py (ARC-86, min_mean_floor=5.0 para GPU) --
# reusado aquí, no un umbral nuevo inventado, aunque viva en un módulo
# distinto (measure_warmup.py es una herramienta de calibración de catálogo
# offline, no algo que validation.py pueda importar directamente).
_GPU_UTIL_NOISE_FLOOR_PCT = 5.0


def validate_windows(
    windows_path: str | Path,
    *,
    target_windows_per_repetition: int,
    device: str,
    gpu_idle_power_mw_by_level: Mapping[str, float] | None = None,
    gpu_active_power_margin_mw: Mapping[str, float] | float | None = None,
) -> Verdict:
    """VAL-09 (ARC-94): segunda etapa de aceptación, DESPUÉS de que
    postprocess.py escribió windows.csv -- validate_run() por sí solo solo
    conoce metadata a nivel de corrida (samples_collected, checksum,
    success_check); nunca miraba si la corrida realmente produjo ventanas
    útiles. Antes de este cambio, una corrida podía quedar accepted=true
    con cero ventanas quality_status="ok" (CPU) o "gpu_telemetry" (GPU),
    cero etiquetas, o menos muestras de las que
    `target_windows_per_repetition` exige -- ese parámetro se validaba en
    el manifiesto pero nunca se usaba para decidir nada después (hallazgo
    de auditoría externa, ver docs/orchestator/agents/
    Registro_Cambios_Fuera_Plan_Original.md ARC-94).

    ARC-129: para GPU, quality_status=="gpu_telemetry" por sí solo no
    distingue una ventana con señal real de una en el piso de ruido del
    sensor -- antes de este cambio, una corrida donde nvidia-smi/NVML
    reportaba 0-2% de utilización en cada muestra (GPU esencialmente ociosa,
    p.ej. un kernel demasiado corto para el intervalo de muestreo) podía
    contar como "usable" igual que una con actividad de cómputo real, ambas
    con el mismo quality_status. Se exige gpu_util_pct >=
    _GPU_UTIL_NOISE_FLOOR_PCT (mismo piso ya establecido y justificado en
    measure_warmup.py, ARC-86 -- no un umbral nuevo). Filas con
    gpu_util_pct vacío/no numérico se tratan como sin señal (no cuentan),
    nunca se asume "0" en silencio ni se descarta el chequeo.

    ARC-174: para CPU, una ventana además debe tener
    ``frequency_quality_status`` en ``{"valid", "not_applicable_native"}``
    para contar como usable -- ``"observation_unreliable"``,
    ``"observation_unverified_grace"`` y el estado vacío (fail-closed) se
    excluyen del conteo, sin rechazar la corrida completa por eso (esa es
    justamente la diferencia con el gate agregado que existía antes,
    ARC-138/166/169, reemplazado en ``validate_cpu_frequency_trace()``).
    Las filas GPU no tienen esta columna poblada (permanecen sin cambios,
    ver ``build_windows()``) -- esta condición nunca se aplica a
    ``device=="gpu"``.

    ARC-185: el piso de ``gpu_util_pct`` tiene sesgo dependiente de la
    frecuencia -- ``utilization.gpu`` de NVML es una FRACCIÓN DE TIEMPO
    (cuánto del intervalo de muestreo hubo algún kernel corriendo), y con
    reloj más lento el mismo trabajo tarda más, así que un kernel
    genuinamente ocioso puede cruzar el piso del 5 % en los niveles bajos
    sin haber hecho más trabajo real. Medido en vivo: rodinia_lud (ocioso
    por diseño de la prueba) pasa de 0.0 % en REF/F0 a 3.5 % en F4.

    Cuando se proveen ``gpu_idle_power_mw_by_level`` y
    ``gpu_active_power_margin_mw``, el criterio pasa a ser potencia sobre
    la línea de reposo (vatios reales de NVML, no una fracción de tiempo
    con ruido de muestreo): ``gpu_power_mw - idle(nivel) >= margen(nivel)``.
    Un kernel ocioso da ~0 W de exceso en CUALQUIER nivel de reloj, así que
    la COMPARACIÓN es invariante a la frecuencia por construcción -- pero
    el margen NO puede ser un solo número para toda la campaña.

    ARC-189 (encontrado el día siguiente de introducir ARC-185, con datos
    reales de tres kernels confirmados activos por otras vías --
    heartwall, gaussian, dgemm_n4096): el exceso de potencia de trabajo
    GPU REAL escala con el reloj casi tan fuerte como la propia potencia de
    reposo. A F0 (1410 MHz) el exceso mínimo observado en ventanas con
    ``gpu_util_pct >= 50`` fue ~9.5-12.7 W; al mismo kernel, en el MISMO
    régimen de actividad, a F4 (210 MHz) el exceso cae a ~1.3-3.8 W. Un
    margen de 20000 mW (el valor recomendado en la primera versión de
    ARC-185, calibrado solo contra el RUIDO de una sonda en reposo, nunca
    contra trabajo real) habría rechazado el 100 % de las ventanas activas
    en F4 de los TRES kernels de referencia -- exactamente el defecto que
    ARC-185 pretendía corregir, pero en la dirección contraria y aplicado a
    datos genuinamente buenos, no solo a los ociosos. Por eso el margen
    debe ser ``gpu_active_power_margin_mw: Mapping[str, float]``, una
    entrada por nivel, igual que la línea de reposo.

    Por compatibilidad se acepta también un único ``float`` (se aplica a
    todos los niveles); pasar un solo número para TODA la rejilla de
    F0 a F4 es casi con certeza un error de calibración, no una elección
    válida, dado lo anterior -- se admite la forma escalar solo para no
    romper un caller que declare un único nivel.

    Sin ``gpu_idle_power_mw_by_level``/``gpu_active_power_margin_mw``
    (``None``, el default) el comportamiento es IDÉNTICO al anterior --
    piso de utilización fijo -- para no alterar ninguna campaña ya en cola
    que no declare el campo nuevo del manifiesto.
    """
    with open(windows_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable_status = "gpu_telemetry" if device == "gpu" else "ok"
    usable_rows = [row for row in rows if row.get("quality_status") == usable_status]
    if device == "gpu":
        use_power_criterion = (
            gpu_idle_power_mw_by_level is not None and gpu_active_power_margin_mw is not None
        )
        if use_power_criterion:
            # ARC-189: margen por nivel. Un float suelto se trata como el
            # mismo margen en todos los niveles (compatibilidad), pero eso
            # es casi con certeza un error de calibración -- ver docstring.
            margin_by_level = (
                gpu_active_power_margin_mw
                if isinstance(gpu_active_power_margin_mw, Mapping)
                else None
            )
            flat_margin = (
                gpu_active_power_margin_mw
                if not isinstance(gpu_active_power_margin_mw, Mapping)
                else None
            )

            def _has_gpu_signal(row: Mapping[str, Any]) -> bool:
                level = row.get("gpu_freq_level_id")
                idle = gpu_idle_power_mw_by_level.get(level) if level else None
                if idle is None:
                    return False  # nivel sin línea de reposo medida: fail-closed, no se asume 0
                margin = margin_by_level.get(level) if margin_by_level is not None else flat_margin
                if margin is None:
                    return False  # nivel sin margen declarado: fail-closed, igual que sin línea de reposo
                try:
                    power = float(row.get("gpu_power_mw") or "nan")
                except ValueError:
                    return False
                return (power - idle) >= margin
        else:
            def _has_gpu_signal(row: Mapping[str, Any]) -> bool:
                try:
                    return float(row.get("gpu_util_pct") or "nan") >= _GPU_UTIL_NOISE_FLOOR_PCT
                except ValueError:
                    return False
        usable_rows = [row for row in usable_rows if _has_gpu_signal(row)]
    else:
        usable_rows = [
            row for row in usable_rows
            if row.get("frequency_quality_status") in ("valid", "not_applicable_native")
        ]
    if len(usable_rows) < target_windows_per_repetition:
        return Verdict(
            False, "I10",
            f"{len(usable_rows)} ventanas '{usable_status}' logradas, por debajo de "
            f"target_windows_per_repetition={target_windows_per_repetition}",
        )
    has_label = any(row.get("phase_label_train") not in (None, "", "None") for row in usable_rows)
    if not has_label:
        return Verdict(False, "I11", "ninguna ventana usable tiene phase_label_train calculado")
    return Verdict(True, None, "ok")


_FREQUENCY_QUALITY_STATUSES: tuple[str, ...] = (
    "valid", "observation_unreliable", "observation_unverified_grace", "not_applicable_native",
)


def summarize_frequency_quality(windows_path: str | Path) -> dict[str, Any]:
    """ARC-174: artefacto concreto de cobertura/calidad de actuación de
    frecuencia para UNA corrida -- separado a propósito de ``Verdict``
    (inclusión en el dataset) porque una corrida puede incluirse
    legítimamente (>=50 ventanas válidas) mientras conserva una fracción
    grande de ventanas ``observation_unreliable`` que nadie debe perder de
    vista solo porque la corrida "pasó". Pensado para escribirse como
    ``frequency_quality_summary.json`` junto a ``windows.csv``.

    Solo considera filas ``quality_status=="ok"`` (ventanas de CPU reales,
    nunca las GPU passthrough ni las excluidas por otro motivo como
    warmup/pmu_degraded) -- coherente con lo que ``validate_windows()``
    ya cuenta como universo candidato antes del filtro de frecuencia.

    ``longest_unreliable_streak`` cuenta la secuencia consecutiva más larga
    (por ``window_index``) de ``"observation_unreliable"`` -- una dispersa
    entre ventanas aisladas es mucho menos preocupante que una racha larga
    concentrada, y esa distinción se pierde si solo se reporta el conteo
    total.
    """
    with open(windows_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    candidate_rows = [row for row in rows if row.get("quality_status") == "ok"]
    candidate_rows.sort(key=lambda row: int(row.get("window_index") or 0))

    counts: dict[str, int] = {status: 0 for status in _FREQUENCY_QUALITY_STATUSES}
    counts["unset"] = 0
    for row in candidate_rows:
        status = row.get("frequency_quality_status")
        if status in counts:
            counts[status] += 1
        else:
            counts["unset"] += 1

    total_candidates = len(candidate_rows)
    valid_count = counts["valid"] + counts["not_applicable_native"]
    fraction_valid = (valid_count / total_candidates) if total_candidates else None

    longest_unreliable_streak = 0
    current_streak = 0
    for row in candidate_rows:
        if row.get("frequency_quality_status") == "observation_unreliable":
            current_streak += 1
            longest_unreliable_streak = max(longest_unreliable_streak, current_streak)
        else:
            current_streak = 0

    return {
        "total_candidate_windows": total_candidates,
        "frequency_quality_counts": counts,
        "valid_window_count": valid_count,
        "fraction_valid": fraction_valid,
        "longest_unreliable_streak": longest_unreliable_streak,
    }


def validate_campaign_calibration(calibration: Any) -> Verdict:
    """VAL-05: D03 (calibración no plausible) rechaza TODA la campaña, no una
    corrida. In practice calibration.run_calibration() already raises before
    the matrix starts and load_calibration() refuses an invalid file
    (CAL-04/CAL-06/POST-15), so this function is the explicit, single place
    campaign.py can call to state that gate instead of re-deriving it.
    """
    if not getattr(calibration, "plausibility_check_passed", False):
        message = getattr(calibration, "plausibility_message", "D03: calibración no plausible")
        return Verdict(False, "D03", message)
    return Verdict(True, None, "ok")


def write_verdict(verdict: Verdict, run_dir: str | Path) -> Path:
    """VAL-06: rejected runs are NEVER deleted. This only ever adds a
    verdict.json next to the run's other artifacts (samples.csv,
    metadata.json, windows.csv); nothing in this module removes a run
    directory or any file inside it.
    """
    path = Path(run_dir) / "verdict.json"
    with path.open("w", encoding="utf-8") as verdict_file:
        json.dump(
            {"accepted": verdict.accepted, "factor_id": verdict.factor_id, "message": verdict.message},
            verdict_file, indent=2, sort_keys=True,
        )
        verdict_file.write("\n")
    return path


def load_verdict(run_dir: str | Path) -> Verdict:
    path = Path(run_dir) / "verdict.json"
    with path.open(encoding="utf-8") as verdict_file:
        data = json.load(verdict_file)
    return Verdict(accepted=data["accepted"], factor_id=data.get("factor_id"), message=data.get("message", ""))
