"""F1-XDEV-002: pruebas de repostprocess_campaign.py.

Verifican que: (1) encuentra run_dirs por el run_id real (build_matrix +
build_run_id), nunca por heurística de nombre; (2) usa el catálogo dado
(nunca manifest.warmup_seconds_override, salvo que se pida explícitamente);
(3) una corrida sin samples.csv se reporta 'skipped', nunca se fabrica;
(4) --dry-run no llama a run_postprocess; (5) un fallo real de una corrida
se reporta 'error', no se oculta ni detiene el resto; (6) el veredicto
accepted/rejected se recalcula sobre el windows.csv YA corregido y
verdict.json queda actualizado, sin borrar ni mover la corrida.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import repostprocess_campaign as rc
from fase1_telemetria import runner as runner_module
from fase1_telemetria import validation as validation_module
from common.hpc.catalog import KernelEntry
from common.hpc.manifest import Cores, FrequencyLevel, Manifest, Timeouts


def _kernel_entry(tmp_path: Path, kernel_id: str, **overrides) -> KernelEntry:
    binary = tmp_path / f"{kernel_id}.bin"
    binary.write_bytes(f"#!/bin/sh\necho {kernel_id}\n".encode())
    checksum = f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}"
    defaults = dict(
        id=kernel_id, suite="npb", role="dataset", exec_path=str(binary), binary_checksum=checksum,
        phase_label_hint="compute_bound", size_variant="S", expected_runtime_seconds=1,
        warmup_seconds=1.7, success_check={"type": "exit_code"}, estimated_memory_bytes=1024,
    )
    defaults.update(overrides)
    return KernelEntry(**defaults)


def _manifest(tmp_path: Path, **overrides) -> Manifest:
    output_dir = tmp_path / "runs"
    output_dir.mkdir(exist_ok=True)
    defaults = dict(
        campaign_id="camp01", environment_tier="local", seed=42, output_dir=output_dir, overwrite=True,
        catalog_path=tmp_path / "catalog.yaml", calibration=(), kernels=("npb_ep",),
        frequency_levels=(FrequencyLevel("REF", "native_governor"),), repetitions_per_combination=1,
        target_windows_per_repetition=10, interval_ns=1_000_000, running_ratio_min=0.9,
        cores=Cores((2, 3, 4, 5), 0, 1, None), smt_policy="all_threads", cgroup_path=None,
        perf_enabled=True, rapl={"enabled": False}, gpu={}, timeouts_seconds=Timeouts(5, 5, 5),
        hardware_datasheet=None,
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def _run_id(manifest, kernel_ref="npb_ep", level="REF", rep=1):
    return runner_module.build_run_id(manifest.campaign_id, kernel_ref, level, rep)


def _write_windows_csv(path: Path, *, n_ok_rows: int, device: str = "cpu") -> None:
    """Un windows.csv real y mínimo que validate_windows() puede evaluar de
    verdad (mismo patrón que test_campaign.py: quality_status/phase_label_
    train/gpu_util_pct/frequency_quality_status)."""
    status = "gpu_telemetry" if device == "gpu" else "ok"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["quality_status", "phase_label_train", "gpu_util_pct", "frequency_quality_status"],
        )
        writer.writeheader()
        for _ in range(n_ok_rows):
            writer.writerow({
                "quality_status": status, "phase_label_train": "compute_bound",
                "gpu_util_pct": "50" if device == "gpu" else "",
                "frequency_quality_status": "" if device == "gpu" else "valid",
            })


def _fake_run_postprocess(n_ok_rows_by_warmup=None, n_ok_rows=10, calls=None):
    """Fábrica de un run_postprocess falso que escribe un windows.csv real.

    `n_ok_rows_by_warmup`, si se da, es un dict {warmup_seconds: n_filas} --
    simula que corregir el warmup deja menos (o más) filas usables, para
    poder probar que el veredicto se recalcula de verdad.
    """
    def fake(run_dir, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        entry = kwargs.get("kernel_entry")
        device = getattr(entry, "device", "cpu")
        n = n_ok_rows
        if n_ok_rows_by_warmup is not None:
            n = n_ok_rows_by_warmup[kwargs["warmup_seconds"]]
        windows_path = Path(run_dir) / "windows.csv"
        _write_windows_csv(windows_path, n_ok_rows=n, device=device)
        return windows_path
    return fake


def test_reprocesa_solo_las_corridas_con_samples_csv(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=0.3)}

    # Una sola combinación (1 kernel x 1 nivel x 1 repetición) -> 1 run_id.
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")

    calls = []
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess(calls=calls))

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert len(results) == 1
    assert results[0]["status"] == "reprocessed"
    assert len(calls) == 1
    # F1-XDEV-002: usa entry.warmup_seconds del catálogo dado, no un valor fijo.
    assert calls[0]["warmup_seconds"] == 0.3
    assert calls[0]["kernel_entry"] is catalog["npb_ep"]
    assert calls[0]["node_id"] == "pacca"
    # Con 10/10 filas "ok" (== target_windows_per_repetition), el veredicto
    # recalculado también queda aceptado.
    assert results[0]["verdict_accepted"] is True


def test_corrida_sin_samples_csv_se_reporta_skipped_no_se_fabrica(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep")}
    # ningún run_dir creado -- la corrida fue rechazada o nunca se hizo

    calls = []
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess(calls=calls))

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert results[0]["status"] == "skipped"
    assert "sin samples.csv" in results[0]["reason"]
    assert calls == []


def test_ignora_warmup_seconds_override_por_defecto(tmp_path, monkeypatch):
    """El paso 1 (recolección) usó warmup_seconds_override=0.0; el
    re-postproceso debe usar el catálogo YA CORREGIDO, no repetir el forzado."""
    manifest = _manifest(tmp_path, warmup_seconds_override=0.0)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=2.4)}  # valor calibrado
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")

    calls = []
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess(calls=calls))

    rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert calls[0]["warmup_seconds"] == 2.4  # NO 0.0


def test_use_manifest_warmup_override_explicito(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, warmup_seconds_override=0.0)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=2.4)}
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")

    calls = []
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess(calls=calls))

    rc.repostprocess_campaign(manifest, catalog, node_id="pacca", ignore_manifest_override=False)
    assert calls[0]["warmup_seconds"] == 0.0


def test_dry_run_no_llama_a_run_postprocess(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=1.1)}
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")

    called = []
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", lambda *a, **k: called.append(1))

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca", dry_run=True)
    assert called == []
    assert results[0]["status"] == "would_reprocess"
    assert results[0]["warmup_seconds_would_use"] == 1.1


def test_fallo_de_una_corrida_se_reporta_error_no_detiene_las_demas(tmp_path, monkeypatch):
    manifest = _manifest(
        tmp_path, kernels=("npb_ep", "npb_bt"),
        frequency_levels=(FrequencyLevel("REF", "native_governor"),),
    )
    catalog = {
        "npb_ep": _kernel_entry(tmp_path, "npb_ep"),
        "npb_bt": _kernel_entry(tmp_path, "npb_bt"),
    }
    for k in ("npb_ep", "npb_bt"):
        d = manifest.output_dir / _run_id(manifest, kernel_ref=k)
        d.mkdir(parents=True)
        (d / "samples.csv").write_text("dummy\n")

    real_fake = _fake_run_postprocess()

    def flaky_run_postprocess(run_dir, **kw):
        if kw["kernel_ref"] == "npb_bt":
            raise ValueError("samples.csv corrupto")
        return real_fake(run_dir, **kw)

    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", flaky_run_postprocess)

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    by_kernel = {r["run_id"].split("__")[1]: r for r in results}
    assert by_kernel["npb_ep"]["status"] == "reprocessed"
    assert by_kernel["npb_bt"]["status"] == "error"
    assert "corrupto" in by_kernel["npb_bt"]["reason"]


def test_kernel_ref_no_esta_en_el_catalogo_dado_se_salta(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    catalog = {}  # catálogo distinto, no incluye npb_ep
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", lambda *a, **k: pytest.fail("no debería llamarse"))

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert results[0]["status"] == "skipped"
    assert "no está en el catálogo" in results[0]["reason"]


def test_filtra_por_only_kernels(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, kernels=("npb_ep", "npb_bt"))
    catalog = {
        "npb_ep": _kernel_entry(tmp_path, "npb_ep"),
        "npb_bt": _kernel_entry(tmp_path, "npb_bt"),
    }
    for k in ("npb_ep", "npb_bt"):
        d = manifest.output_dir / _run_id(manifest, kernel_ref=k)
        d.mkdir(parents=True)
        (d / "samples.csv").write_text("dummy\n")
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess())

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca", only_kernels=["npb_ep"])
    assert len(results) == 1
    assert "npb_ep" in results[0]["run_id"]


# ----------------------------------------------------------- re-validación


def test_veredicto_se_recalcula_y_verdict_json_se_actualiza(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=0.0)}
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")
    # veredicto provisional en disco, como lo dejaría la campaña en vivo.
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess(n_ok_rows=10))

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert results[0]["verdict_accepted"] is True
    assert results[0]["verdict_changed"] is False
    on_disk = validation_module.load_verdict(run_dir)
    assert on_disk.accepted is True


def test_corregir_el_warmup_puede_hacer_que_una_corrida_al_limite_se_rechace(tmp_path, monkeypatch):
    """El caso que motivó la re-validación: con warmup=0 (provisional) la
    corrida alcanzaba las 10 ventanas objetivo; con el warmup calibrado
    (0.6s) el mismo samples.csv real solo deja 4 -- por debajo del objetivo
    -- así que debe pasar de accepted a rejected, y quedar registrado."""
    manifest = _manifest(tmp_path, target_windows_per_repetition=10)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=0.6)}
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")
    # Al recolectar (warmup_seconds_override=0.0) la campaña en vivo vio 10
    # ventanas "ok" y aceptó la corrida.
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    monkeypatch.setattr(
        rc.postprocess_module, "run_postprocess",
        _fake_run_postprocess(n_ok_rows_by_warmup={0.6: 4}),
    )

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert results[0]["verdict_accepted"] is False
    assert results[0]["verdict_changed"] is True
    on_disk = validation_module.load_verdict(run_dir)
    assert on_disk.accepted is False  # verdict.json quedó actualizado
    # VAL-06: nunca se borra ni se mueve la corrida.
    assert (run_dir / "samples.csv").exists()
    assert (run_dir / "windows.csv").exists()


def test_verdict_changed_es_false_sin_verdict_previo(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    catalog = {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=0.0)}
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")
    # sin verdict.json previo -- p.ej. windows.csv generado fuera del flujo de campaign.py

    monkeypatch.setattr(rc.postprocess_module, "run_postprocess", _fake_run_postprocess(n_ok_rows=10))

    results = rc.repostprocess_campaign(manifest, catalog, node_id="pacca")
    assert results[0]["verdict_changed"] is False
    assert results[0]["verdict_accepted"] is True


def test_main_cli_reporta_veredictos_cambiados(tmp_path, monkeypatch, capsys):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "campaign.yaml"
    catalog_path = tmp_path / "catalog.yaml"
    monkeypatch.setattr(rc.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(rc.catalog_module, "load_catalog",
                        lambda path: {"npb_ep": _kernel_entry(tmp_path, "npb_ep", warmup_seconds=0.6)})
    run_dir = manifest.output_dir / _run_id(manifest)
    run_dir.mkdir(parents=True)
    (run_dir / "samples.csv").write_text("dummy\n")
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)
    monkeypatch.setattr(rc.postprocess_module, "run_postprocess",
                        _fake_run_postprocess(n_ok_rows_by_warmup={0.6: 2}))

    rc_code = rc.main(["--manifest", str(manifest_path), "--node-id", "pacca"])
    out = capsys.readouterr().out
    assert "VEREDICTO CAMBIÓ" in out
    assert "revisar a mano" in out
    assert rc_code == 0  # no hubo 'error', solo un cambio de veredicto (no bloquea)
