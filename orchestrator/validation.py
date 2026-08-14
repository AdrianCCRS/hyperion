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
    """
    with open(windows_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable_status = "gpu_telemetry" if device == "gpu" else "ok"
    usable_rows = [row for row in rows if row.get("quality_status") == usable_status]
    if device == "gpu":
        def _has_gpu_signal(row: Mapping[str, Any]) -> bool:
            try:
                return float(row.get("gpu_util_pct") or "nan") >= _GPU_UTIL_NOISE_FLOOR_PCT
            except ValueError:
                return False
        usable_rows = [row for row in usable_rows if _has_gpu_signal(row)]
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
