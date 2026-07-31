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


def build_report(
    *,
    campaign_id: str,
    verdicts: Sequence[Any],
    calibration_references: Any = None,
    total_core_hours: float = 0.0,
    cv_threshold_pct: float = 5.0,
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
    }


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "campaign_report.json"
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    return path
