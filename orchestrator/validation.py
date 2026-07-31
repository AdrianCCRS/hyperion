from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

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

    VAL-08: window-level rejects (I01 no_freq_reading, I02 low running_ratio,
    I03 sampling jitter, warmup_excluded, intensity_undefined) are never
    consulted here. They only ever affect a single row's quality_status in
    windows.csv (postprocess.py); a bad window never flips this verdict.
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
    if not verify_binary(kernel_entry):
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
