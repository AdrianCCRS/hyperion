import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import cli


def test_diagnose_reenvia_argumentos_a_diagnostics_main(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.diagnostics_module, "main", lambda argv: calls.append(argv) or 0)

    exit_code = cli.main([
        "diagnose", "--manifest", "camp.yaml", "--output-dir", "out", "--use-allowed-cpus",
    ])

    assert exit_code == 0
    assert calls == [["--manifest", "camp.yaml", "--output-dir", "out", "--use-allowed-cpus"]]


def test_calibrate_dispara_calibracion_node_profile_y_referencias(monkeypatch, tmp_path):
    manifest = _fake_manifest(tmp_path)
    catalog = {"npb_ep": object()}
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: catalog)
    monkeypatch.setattr(cli, "_detect_environment", lambda manifest, config: "ENV")

    calls = {}
    monkeypatch.setattr(cli.calibration_module, "run_calibration", lambda *a, **k: (
        calls.setdefault("run_calibration", (a, k)),
        _fake_roofline(),
    )[1])
    monkeypatch.setattr(cli.node_profile_module, "build_node_profile", lambda *a, **k: (
        calls.setdefault("build_node_profile", (a, k)), "PROFILE"
    )[1])
    monkeypatch.setattr(cli.node_profile_module, "write_node_profile", lambda *a, **k: calls.setdefault("write_node_profile", (a, k)))
    monkeypatch.setattr(cli.calibration_module, "run_calibration_references", lambda *a, **k: (
        calls.setdefault("run_calibration_references", (a, k)), _fake_references()
    )[1])

    exit_code = cli.main([
        "calibrate", "--manifest", "camp.yaml", "--node-id", "felix-sc3", "--reference-kernel-ref", "npb_ep",
    ])

    assert exit_code == 0
    assert calls["run_calibration"][1]["node_id"] == "felix-sc3"
    assert calls["build_node_profile"][1]["node_id"] == "felix-sc3"
    assert calls["run_calibration_references"][1]["node_id"] == "felix-sc3"


def test_run_campaign_pasa_los_argumentos_correctos(monkeypatch, tmp_path):
    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: {})
    monkeypatch.setattr(cli, "_detect_environment", lambda manifest, config: "ENV")
    monkeypatch.setattr(cli.node_profile_module, "build_node_profile", lambda *a, **k: "PROFILE")
    monkeypatch.setattr(cli.preflight_module, "run_campaign_preflight", lambda *a, **k: [_passing_check()])

    calls = {}

    def fake_run_campaign(manifest_arg, catalog_arg, env_arg, **kwargs):
        calls["args"] = (manifest_arg, catalog_arg, env_arg, kwargs)
        return _fake_campaign_result()

    monkeypatch.setattr(cli.campaign_module, "run_campaign", fake_run_campaign)

    exit_code = cli.main([
        "run-campaign", "--manifest", "camp.yaml", "--node-id", "felix-sc3",
        "--reference-kernel-ref", "npb_ep", "--campaign-timeout-seconds", "3600",
    ])

    assert exit_code == 0
    _, _, env_arg, kwargs = calls["args"]
    assert env_arg == "ENV"
    assert kwargs["node_id"] == "felix-sc3"
    assert kwargs["reference_kernel_ref"] == "npb_ep"
    assert kwargs["campaign_timeout_seconds"] == 3600.0


def test_run_campaign_arc142_sale_con_codigo_1_si_hay_rechazadas(monkeypatch, tmp_path):
    """ARC-142: run_campaign() no lanza excepción por una combinación
    rechazada -- es un veredicto normal, no un fallo del proceso. Sin este
    chequeo, un script/CI que solo mira el exit code no distingue una
    campaña con rechazos de una 100% aceptada."""
    from types import SimpleNamespace

    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: {})
    monkeypatch.setattr(cli, "_detect_environment", lambda manifest, config: "ENV")
    monkeypatch.setattr(cli.node_profile_module, "build_node_profile", lambda *a, **k: "PROFILE")
    monkeypatch.setattr(cli.preflight_module, "run_campaign_preflight", lambda *a, **k: [_passing_check()])

    def result_con_rechazadas():
        progress = SimpleNamespace(
            accepted_run_ids=["r1"], rejected_run_ids=["r2"], skipped_run_ids=[],
            run_ids_in_order=["r1", "r2"], total_core_hours=1.0, frequency_restored_verified=True,
        )
        return SimpleNamespace(progress=progress)

    monkeypatch.setattr(cli.campaign_module, "run_campaign", lambda *a, **k: result_con_rechazadas())

    exit_code = cli.main([
        "run-campaign", "--manifest", "camp.yaml", "--node-id", "felix-sc3",
        "--reference-kernel-ref", "npb_ep",
    ])

    assert exit_code == 1


def test_run_campaign_arc142_sale_con_codigo_1_si_la_matriz_quedo_incompleta(monkeypatch, tmp_path):
    """ARC-142: aunque no haya ninguna rechazada, procesar menos run_ids que
    los planeados en run_ids_in_order (matriz recortada) tampoco debe salir
    con 0."""
    from types import SimpleNamespace

    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: {})
    monkeypatch.setattr(cli, "_detect_environment", lambda manifest, config: "ENV")
    monkeypatch.setattr(cli.node_profile_module, "build_node_profile", lambda *a, **k: "PROFILE")
    monkeypatch.setattr(cli.preflight_module, "run_campaign_preflight", lambda *a, **k: [_passing_check()])

    def result_incompleto():
        progress = SimpleNamespace(
            accepted_run_ids=["r1"], rejected_run_ids=[], skipped_run_ids=[],
            run_ids_in_order=["r1", "r2"], total_core_hours=1.0, frequency_restored_verified=True,
        )
        return SimpleNamespace(progress=progress)

    monkeypatch.setattr(cli.campaign_module, "run_campaign", lambda *a, **k: result_incompleto())

    exit_code = cli.main([
        "run-campaign", "--manifest", "camp.yaml", "--node-id", "felix-sc3",
        "--reference-kernel-ref", "npb_ep",
    ])

    assert exit_code == 1


def test_run_campaign_sale_con_codigo_1_si_no_verifico_restauracion(monkeypatch, tmp_path):
    from types import SimpleNamespace

    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: {})
    monkeypatch.setattr(cli, "_detect_environment", lambda manifest, config: "ENV")
    monkeypatch.setattr(cli.node_profile_module, "build_node_profile", lambda *a, **k: "PROFILE")
    monkeypatch.setattr(cli.preflight_module, "run_campaign_preflight", lambda *a, **k: [_passing_check()])
    progress = SimpleNamespace(
        accepted_run_ids=["r1"], rejected_run_ids=[], skipped_run_ids=[],
        run_ids_in_order=["r1"], total_core_hours=1.0, frequency_restored_verified=False,
    )
    monkeypatch.setattr(
        cli.campaign_module, "run_campaign", lambda *a, **k: SimpleNamespace(progress=progress),
    )

    exit_code = cli.main([
        "run-campaign", "--manifest", "camp.yaml", "--node-id", "felix-sc3",
        "--reference-kernel-ref", "npb_ep",
    ])

    assert exit_code == 1


def test_run_campaign_arc45_corre_preflight_automaticamente_y_bloquea_si_falla(monkeypatch, tmp_path):
    """ARC-45: cmd_run_campaign debe invocar run_campaign_preflight() por su
    cuenta -- antes solo se corría a mano por fuera del CLI (scripts/*)."""
    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: {})
    monkeypatch.setattr(cli, "_detect_environment", lambda manifest, config: "ENV")
    monkeypatch.setattr(cli.node_profile_module, "build_node_profile", lambda *a, **k: "PROFILE")

    preflight_calls = {}

    def fake_preflight(manifest_arg, env_arg, catalog_arg, **kwargs):
        preflight_calls["args"] = (manifest_arg, env_arg, catalog_arg, kwargs)
        return [_failing_check()]

    monkeypatch.setattr(cli.preflight_module, "run_campaign_preflight", fake_preflight)

    run_campaign_calls = []
    monkeypatch.setattr(
        cli.campaign_module, "run_campaign",
        lambda *a, **k: run_campaign_calls.append((a, k)) or _fake_campaign_result(),
    )

    exit_code = cli.main([
        "run-campaign", "--manifest", "camp.yaml", "--node-id", "felix-sc3",
        "--reference-kernel-ref", "npb_ep",
    ])

    assert exit_code == 1
    assert run_campaign_calls == []
    assert preflight_calls["args"][3]["node_profile"] == "PROFILE"


def test_postprocess_resuelve_el_kernel_desde_el_catalogo(monkeypatch, tmp_path):
    from types import SimpleNamespace

    manifest = _fake_manifest(tmp_path)
    entry = SimpleNamespace(warmup_seconds=0.5)
    monkeypatch.setattr(cli.manifest_module, "load", lambda path: manifest)
    monkeypatch.setattr(cli.catalog_module, "load_catalog", lambda path: {"npb_ep": entry})

    calls = {}

    def fake_run_postprocess(run_dir, **kwargs):
        calls["run_dir"] = run_dir
        calls["kwargs"] = kwargs
        return Path(run_dir) / "windows.csv"

    monkeypatch.setattr(cli.postprocess_module, "run_postprocess", fake_run_postprocess)

    exit_code = cli.main([
        "postprocess", "--manifest", "camp.yaml", "--run-dir", str(tmp_path / "run_x"),
        "--run-id", "run_x", "--repetition", "1", "--kernel-ref", "npb_ep",
        "--node-id", "felix-sc3", "--freq-level-id", "REF",
    ])

    assert exit_code == 0
    assert calls["kwargs"]["kernel_entry"] is entry
    assert calls["kwargs"]["node_id"] == "felix-sc3"


def test_report_lee_metadata_y_veredictos(monkeypatch, tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "campaign_metadata.json").write_text(
        '{"campaign_id": "camp01", "accepted_run_ids": ["r1"], "rejected_run_ids": [], "total_core_hours": 1.0}'
    )
    run_dir = campaign_dir / "r1"
    run_dir.mkdir()
    cli.validation_module.write_verdict(cli.validation_module.Verdict(True, None, "ok"), run_dir)

    written = {}
    monkeypatch.setattr(cli.report_module, "write_report", lambda data, output_dir: written.setdefault("data", data) or Path(output_dir) / "campaign_report.json")

    exit_code = cli.main(["report", "--campaign-dir", str(campaign_dir)])

    assert exit_code == 0
    assert written["data"]["campaign_id"] == "camp01"
    assert written["data"]["total_runs"] == 1


def test_arc142_report_cuenta_skipped_run_ids(monkeypatch, tmp_path):
    """ARC-142: skipped_run_ids (MET-06) son corridas aceptadas en una
    sesión anterior de run_campaign() -- cmd_report las ignoraba, subcontando
    total_runs/factor_table para cualquier campaña reanudada."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "campaign_metadata.json").write_text(json.dumps({
        "campaign_id": "camp01", "accepted_run_ids": ["r2"], "rejected_run_ids": [],
        "skipped_run_ids": ["r1"], "total_core_hours": 1.0,
    }))
    for run_id in ("r1", "r2"):
        run_dir = campaign_dir / run_id
        run_dir.mkdir()
        cli.validation_module.write_verdict(cli.validation_module.Verdict(True, None, "ok"), run_dir)

    written = {}
    monkeypatch.setattr(cli.report_module, "write_report", lambda data, output_dir: written.setdefault("data", data) or Path(output_dir) / "campaign_report.json")

    exit_code = cli.main(["report", "--campaign-dir", str(campaign_dir)])

    assert exit_code == 0
    assert written["data"]["total_runs"] == 2


def test_cam08_report_propaga_overhead_pct_values(monkeypatch, tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "campaign_metadata.json").write_text(
        json.dumps({
            "campaign_id": "camp01", "accepted_run_ids": [], "rejected_run_ids": [],
            "total_core_hours": 1.0, "overhead_pct_values": [5.0, 50.0, 5.0],
        })
    )

    written = {}
    monkeypatch.setattr(cli.report_module, "write_report", lambda data, output_dir: written.setdefault("data", data) or Path(output_dir) / "campaign_report.json")

    exit_code = cli.main(["report", "--campaign-dir", str(campaign_dir)])

    assert exit_code == 0
    assert written["data"]["overhead_pct_samples"] == 3
    assert written["data"]["overhead_stability_warning"] is not None


def _fake_manifest(tmp_path: Path):
    from types import SimpleNamespace
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        cores=SimpleNamespace(delegated_cpus=(2, 3)),
        output_dir=output_dir, running_ratio_min=0.9, rapl={"enabled": False},
        catalog_path=tmp_path / "catalog.yaml",
    )


def _fake_roofline():
    from types import SimpleNamespace
    return SimpleNamespace(plausibility_check_passed=True)


def _fake_references():
    from types import SimpleNamespace
    return SimpleNamespace(accepted=True)


def _passing_check():
    from orchestrator.preflight import CheckResult
    return CheckResult("I07", "Directorio de campaña", True, True, {}, "ok")


def _failing_check():
    from orchestrator.preflight import CheckResult
    return CheckResult("D01", "Turbo/HWP deshabilitado", False, True, {}, "turbo sigue habilitado")


def _fake_campaign_result():
    from types import SimpleNamespace
    progress = SimpleNamespace(
        accepted_run_ids=["r1"], rejected_run_ids=[], skipped_run_ids=[],
        run_ids_in_order=["r1"], total_core_hours=1.0, frequency_restored_verified=True,
    )
    return SimpleNamespace(progress=progress)
