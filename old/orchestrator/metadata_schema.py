from __future__ import annotations

from typing import Any, Mapping

# MET-XX ids refer to docs/retoma/Guia_Maestra_Fase1_DVFS.md section 12.11.
# This module is the single place that defines (a) the disjoint-merge
# primitive every other module uses to combine launcher/orchestrator/campaign
# metadata (MET-01), and (b) what "complete" metadata means at each level, so
# report.py and tests can check it without re-deriving the field lists.

# Fields every individual run's metadata.json must carry (runner.py's merge
# of launcher metadata + orchestrator fields, RUN-06/MET-07).
RUN_METADATA_REQUIRED_KEYS = frozenset({
    "run_id", "campaign_id", "kernel_ref", "kernel_suite", "kernel_role",
    "freq_level_id", "repetition_index", "node_id", "binary_checksum",
})

# Fields campaign_metadata.json must carry (campaign.py, CAM-02/CAM-05/MET-02/MET-06).
CAMPAIGN_METADATA_REQUIRED_KEYS = frozenset({
    "campaign_id", "seed", "run_ids_in_order", "accepted_run_ids",
    "rejected_run_ids", "skipped_run_ids", "total_core_hours",
    "frequency_restored_verified",
})

# Fields every single row of windows.csv must carry (postprocess.py, POST-14/MLT-01/MET-07).
WINDOW_TRACEABILITY_KEYS = frozenset({
    "run_id", "kernel_ref", "node_id", "roofline_calibration_ref",
    "node_profile_ref", "calibration_ref", "binary_checksum",
})


class MetadataCollisionError(ValueError):
    """Raised by merge_metadata() when two metadata sources define the same key."""


def merge_metadata(base: Mapping[str, Any], extra: Mapping[str, Any], *, context: str = "") -> dict[str, Any]:
    """MET-01: the only sanctioned way to combine two metadata dicts.

    Never ``{**base, **extra}``: a silent overwrite there would hide a real
    bug (e.g. the launcher and the orchestrator both trying to own
    "run_id"). Any key present in both raises instead of picking a winner.
    """
    collisions = set(base) & set(extra)
    if collisions:
        raise MetadataCollisionError(f"{context}: colisión de claves de metadata: {sorted(collisions)}")
    merged = dict(base)
    merged.update(extra)
    return merged


def validate_run_metadata(metadata: Mapping[str, Any]) -> list[str]:
    """Returns the RUN_METADATA_REQUIRED_KEYS missing from `metadata` (empty = complete)."""
    return sorted(RUN_METADATA_REQUIRED_KEYS - set(metadata))


def validate_campaign_metadata(metadata: Mapping[str, Any]) -> list[str]:
    """Returns the CAMPAIGN_METADATA_REQUIRED_KEYS missing from `metadata`."""
    return sorted(CAMPAIGN_METADATA_REQUIRED_KEYS - set(metadata))


def validate_window_traceability(window_row: Mapping[str, Any]) -> list[str]:
    """Returns the WINDOW_TRACEABILITY_KEYS missing from one windows.csv row."""
    return sorted(WINDOW_TRACEABILITY_KEYS - set(window_row))
