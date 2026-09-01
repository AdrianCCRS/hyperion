from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

# MET-XX ids refer to docs/retoma/Guia_Maestra_Fase1_DVFS.md section 12.11.


@dataclass(frozen=True)
class FactorRow:
    """One row of the campaign report's rejection table (MET-04)."""

    factor_id: str  # "accepted", or the rejection factor_id (C02, I04, ...)
    count: int
    percentage: float


def build_factor_table(verdicts: Sequence[Any]) -> list[FactorRow]:
    """MET-04: counts and percentages by factor_id, summing to EXACTLY 100%.

    Naively rounding every row to 2 decimals can drift a few hundredths off
    100 once several rows are summed; the last row (alphabetically) absorbs
    whatever remainder is left instead of letting that drift show up.
    `verdicts` are anything with `.accepted`/`.factor_id`
    (validation.Verdict works directly).
    """
    if not verdicts:
        return []

    buckets: dict[str, int] = {}
    for verdict in verdicts:
        key = "accepted" if verdict.accepted else (verdict.factor_id or "unknown")
        buckets[key] = buckets.get(key, 0) + 1

    total = len(verdicts)
    keys = sorted(buckets)
    rows: list[FactorRow] = []
    accumulated_pct = 0.0
    for index, key in enumerate(keys):
        count = buckets[key]
        if index == len(keys) - 1:
            percentage = round(100.0 - accumulated_pct, 2)
        else:
            percentage = round(count / total * 100.0, 2)
            accumulated_pct += percentage
        rows.append(FactorRow(key, count, percentage))
    return rows


def calibration_stability_warning(calibration_references: Any, threshold_pct: float = 5.0) -> str | None:
    """MET-05: a visible warning when cv_pct exceeds the threshold (D04,
    non-blocking). Returns None when there is nothing to warn about, or when
    no calibration_references was supplied at all."""
    if calibration_references is None:
        return None
    cv_pct = getattr(calibration_references, "cv_pct", None)
    if cv_pct is None or cv_pct <= threshold_pct:
        return None
    return (
        f"ADVERTENCIA (D04): calibration_references.cv_pct={cv_pct:.2f}% "
        f"supera el umbral {threshold_pct:.2f}% -- las referencias P95 pueden ser inestables"
    )


def _cv_pct(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance ** 0.5 / abs(mean)) * 100.0


def overhead_stats(overhead_pct_values: Sequence[float] | None) -> dict[str, Any]:
    """CAM-08: mean and CV of instrumentation overhead across all
    baseline/telemetry pairs the campaign actually ran. Empty input means
    "no pair ran this campaign" (e.g. every combination was resumed from a
    previous accepted verdict, CAM-03), not "zero overhead"."""
    values = list(overhead_pct_values or [])
    if not values:
        return {"overhead_pct_mean": None, "overhead_pct_cv": None, "overhead_pct_samples": 0}
    return {
        "overhead_pct_mean": sum(values) / len(values),
        "overhead_pct_cv": _cv_pct(values),
        "overhead_pct_samples": len(values),
    }


def overhead_stability_warning(overhead_pct_values: Sequence[float] | None, threshold_pct: float = 10.0) -> str | None:
    """F4.4: overhead baseline-vs-telemetry debe ser estable (CV < 10%).
    Advertencia no bloqueante -- un CV alto no invalida las corridas ya
    aceptadas, pero señala que el overhead de instrumentación no es
    predecible entre repeticiones."""
    stats = overhead_stats(overhead_pct_values)
    cv = stats["overhead_pct_cv"]
    if cv is None or cv <= threshold_pct:
        return None
    return (
        f"ADVERTENCIA (F4.4): overhead_pct_cv={cv:.2f}% "
        f"supera el umbral {threshold_pct:.2f}% -- el overhead de instrumentación no es estable entre corridas"
    )


def build_report(
    *,
    campaign_id: str,
    verdicts: Sequence[Any],
    calibration_references: Any = None,
    total_core_hours: float = 0.0,
    cv_threshold_pct: float = 5.0,
    overhead_pct_values: Sequence[float] | None = None,
    overhead_cv_threshold_pct: float = 10.0,
) -> dict[str, Any]:
    """Assembles the campaign report as a plain dict, ready to serialize."""
    factor_table = build_factor_table(verdicts)
    return {
        "campaign_id": campaign_id,
        "total_runs": len(verdicts),
        "factor_table": [asdict(row) for row in factor_table],
        "factor_table_percentage_sum": round(sum(row.percentage for row in factor_table), 2) if factor_table else 0.0,
        "total_core_hours": total_core_hours,
        "calibration_stability_warning": calibration_stability_warning(calibration_references, cv_threshold_pct),
        **overhead_stats(overhead_pct_values),
        "overhead_stability_warning": overhead_stability_warning(overhead_pct_values, overhead_cv_threshold_pct),
    }


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "campaign_report.json"
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    return path
