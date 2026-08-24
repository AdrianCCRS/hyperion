"""Regresión del bug encontrado el 2026-08-24: `reprocess_run()` leía
`metadata.get("repetition", 1)`, pero la clave real en `metadata.json` es
`repetition_index` (`metadata_schema.RUN_METADATA_REQUIRED_KEYS`). El
`.get()` con el nombre equivocado nunca fallaba -- simplemente caía
siempre al valor por defecto `1`, así que TODAS las corridas reprocesadas
con esta herramienta (incluida `pacca_cpu_final_attempt03_20260820_arc174`,
540 corridas) quedaron con la columna `repetition` de `windows.csv`
constante en 1, sin importar la repetición real (1 a 10) que fuera.

No afecta al runner en vivo (`campaign.py` -> `postprocess.py`), que pasa
`repetition=item.combination.repetition_index` directo desde el objeto
`Combination` en memoria, nunca releído de `metadata.json` -- por eso
ninguna campaña en cola (6431, 6412, 6471-6477) estaba en riesgo.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.manifest import FrequencyLevel
from orchestrator.schemas.tools import reprocess_frequency_quality_v2 as reprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_campaign import _catalog, _kernel_entry, _manifest  # noqa: E402
from test_postprocess import SAMPLES_HEADER, _cpu_row, _energy_row, _write_samples  # noqa: E402


def _run_dir_with_metadata(tmp_path: Path, *, repetition_index: int, binary_checksum: str) -> Path:
    run_dir = tmp_path / "runs" / "camp01__npb_ep__F0__rep{:02d}".format(repetition_index)
    run_dir.mkdir(parents=True)

    metadata = {
        "run_id": run_dir.name,
        "campaign_id": "camp01",
        "kernel_ref": "npb_ep",
        "kernel_suite": "npb",
        "kernel_role": "dataset",
        "freq_level_id": "F0",
        "repetition_index": repetition_index,
        "node_id": "test-node",
        "binary_checksum": binary_checksum,
        "freq_khz_requested": 3200000,
        "freq_khz_applied": 3200000,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    rows = [
        _cpu_row(repetition=1, ts=0, instructions=1_000_000, cycles=2_000_000,
                 cache_references=10_000, cache_misses=100,
                 time_enabled=1_000_000_000, time_running=1_000_000_000),
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=2_000_000, cycles=4_000_000,
                 cache_references=20_000, cache_misses=200,
                 time_enabled=2_000_000_000, time_running=2_000_000_000),
        _energy_row(repetition=1, ts=0, pkg_delta_uj=0),
        _energy_row(repetition=1, ts=1_000_000_000, pkg_delta_uj=1_000_000),
    ]
    _write_samples(run_dir / "samples.csv", rows)
    return run_dir


@pytest.fixture
def _stubbed_heavy_dependencies(monkeypatch):
    """`reprocess_run` también consulta calibración/perfil de nodo y el
    validador estructural de frecuencia -- se sustituyen por dobles
    mínimos porque esta prueba existe para verificar UNA cosa (qué
    repetición termina en windows.csv), no para reejercitar esos otros
    mecanismos, que ya tienen su propia cobertura en test_postprocess.py
    y test_validation.py."""
    monkeypatch.setattr(
        reprocess.calibration_module, "load_calibration",
        lambda output_dir, freq_level_id: SimpleNamespace(i_ridge_flops_per_byte=1.0),
    )
    monkeypatch.setattr(
        reprocess.node_profile_module, "load_node_profile",
        lambda output_dir: SimpleNamespace(cache_line_size_bytes=64),
    )
    monkeypatch.setattr(
        reprocess.validation_module, "validate_cpu_frequency_trace",
        lambda *a, **k: (
            reprocess.validation_module.Verdict(accepted=True, factor_id=None, message=""),
            {"structural_valid": True},
        ),
    )


def test_reprocess_run_usa_repetition_index_de_metadata_no_el_default(
    tmp_path, _stubbed_heavy_dependencies
):
    entry = _kernel_entry(tmp_path, "npb_ep")
    catalog = {"npb_ep": entry}
    manifest = _manifest(
        tmp_path,
        kernels=("npb_ep",),
        frequency_levels=(FrequencyLevel("F0", "fixed", fraction=1.0),),
    )
    run_dir = _run_dir_with_metadata(
        tmp_path, repetition_index=7, binary_checksum=entry.binary_checksum,
    )
    derived_root = tmp_path / "derived"

    result = reprocess.reprocess_run(run_dir, manifest=manifest, catalog=catalog, derived_root=derived_root)

    assert not result["skipped"], result.get("reason")
    windows_path = derived_root / run_dir.name / "windows.csv"
    assert windows_path.exists()

    import csv
    with windows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "windows.csv salio vacio"
    for row in rows:
        assert row["repetition"] == "7", (
            f"esperaba repetition=7 (de metadata.json.repetition_index), "
            f"salio {row['repetition']!r} -- volvio el bug del default a 1"
        )


def test_reprocess_run_no_confia_en_la_clave_vieja_repetition(
    tmp_path, _stubbed_heavy_dependencies
):
    """Si metadata.json trajera la clave vieja e incorrecta ``repetition``
    (nunca escrita por el runner real, pero cubre el caso), el reprocesador
    NO debe usarla -- solo ``repetition_index`` es la fuente de verdad."""
    entry = _kernel_entry(tmp_path, "npb_ep")
    catalog = {"npb_ep": entry}
    manifest = _manifest(
        tmp_path,
        kernels=("npb_ep",),
        frequency_levels=(FrequencyLevel("F0", "fixed", fraction=1.0),),
    )
    run_dir = _run_dir_with_metadata(
        tmp_path, repetition_index=3, binary_checksum=entry.binary_checksum,
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["repetition"] = 999  # clave vieja, no debe ganarle a repetition_index
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    derived_root = tmp_path / "derived"

    reprocess.reprocess_run(run_dir, manifest=manifest, catalog=catalog, derived_root=derived_root)

    import csv
    with (derived_root / run_dir.name / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["repetition"] == "3" for row in rows)
