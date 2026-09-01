from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import metadata_schema


def test_met01_merge_metadata_combina_sin_colision():
    merged = metadata_schema.merge_metadata({"a": 1}, {"b": 2}, context="TEST")
    assert merged == {"a": 1, "b": 2}


def test_met01_merge_metadata_rechaza_colision_en_vez_de_pisar():
    with pytest.raises(metadata_schema.MetadataCollisionError, match="TEST"):
        metadata_schema.merge_metadata({"a": 1}, {"a": 2}, context="TEST")


def test_met01_merge_metadata_error_es_value_error():
    # Compatibilidad con codigo que capture ValueError generico.
    assert issubclass(metadata_schema.MetadataCollisionError, ValueError)


def test_validate_run_metadata_reporta_claves_faltantes():
    incomplete = {"run_id": "r", "campaign_id": "c"}
    missing = metadata_schema.validate_run_metadata(incomplete)
    assert missing  # hay claves faltantes
    assert "node_id" in missing
    assert "run_id" not in missing


def test_validate_run_metadata_completa_no_reporta_nada():
    complete = {key: "x" for key in metadata_schema.RUN_METADATA_REQUIRED_KEYS}
    assert metadata_schema.validate_run_metadata(complete) == []


def test_validate_campaign_metadata_completa_no_reporta_nada():
    complete = {key: "x" for key in metadata_schema.CAMPAIGN_METADATA_REQUIRED_KEYS}
    assert metadata_schema.validate_campaign_metadata(complete) == []


def test_met07_validate_window_traceability_completa_no_reporta_nada():
    complete = {key: "x" for key in metadata_schema.WINDOW_TRACEABILITY_KEYS}
    assert metadata_schema.validate_window_traceability(complete) == []


def test_met07_validate_window_traceability_detecta_ausencia():
    missing = metadata_schema.validate_window_traceability({"run_id": "r"})
    assert "node_profile_ref" in missing
