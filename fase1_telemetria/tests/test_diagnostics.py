from pathlib import Path
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import diagnostics
from common.hpc.environment import EnvironmentProfile


def test_diagnostico_carga_componentes_y_serializa_contexto(tmp_path, monkeypatch):
    manifest = SimpleNamespace(
        campaign_id="auditoria",
        environment_tier="hpc_sc3",
        cores=SimpleNamespace(delegated_cpus=(4, 5)),
        catalog_path=tmp_path / "catalog.yaml",
    )
    environment = EnvironmentProfile(
        "hpc_sc3", True, ["package-0"], True, "acpi-cpufreq",
        [1500000, 2450000], 1, {4: [4, 132]}, False, False,
    )
    environment.delegated_cpus = [8, 9]
    environment.numa_cpu_map = {0: [8, 9]}
    environment.delegated_cpu_numa_nodes = {8: 0, 9: 0}
    environment.perf_events_available = ["cpu-cycles"]
    monkeypatch.setattr(diagnostics, "load", lambda _: manifest)
    monkeypatch.setattr(diagnostics, "load_catalog", lambda _: {"stream": object()})
    monkeypatch.setattr(diagnostics, "load_config", lambda _: object())
    monkeypatch.setattr(diagnostics, "detect_environment", lambda cpus, config: environment)
    monkeypatch.setattr(diagnostics, "compute_matrix_size", lambda _: 6)
    monkeypatch.setattr(diagnostics, "_runtime_context", lambda: {"effective_cpus": [8, 9]})

    artifact = diagnostics.create_startup_diagnostic(
        tmp_path / "campaign.yaml", tmp_path / "report", delegated_cpus="8-9"
    )

    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["manifest"]["declared_delegated_cpus"] == [4, 5]
    assert report["environment"]["delegated_cpus"] == [8, 9]
    assert report["catalog"] == {"entries": ["stream"], "entry_count": 1, "loaded": True}
    assert report["runtime"]["effective_cpus"] == [8, 9]


def test_cli_usa_los_cpus_permitidos(tmp_path, monkeypatch, capsys):
    artifact = tmp_path / "startup_diagnostic.json"
    captured = {}

    def crear(manifest, output_dir, *, config_path, delegated_cpus):
        captured.update(manifest=manifest, output_dir=output_dir, config_path=config_path, delegated_cpus=delegated_cpus)
        return artifact

    monkeypatch.setattr(diagnostics, "create_startup_diagnostic", crear)
    monkeypatch.setattr(diagnostics.os, "sched_getaffinity", lambda _: {7, 3})

    assert diagnostics.main(["--manifest", "campaign.yaml", "--output-dir", "report", "--use-allowed-cpus"]) == 0
    assert captured["delegated_cpus"] == "3,7"
    assert capsys.readouterr().out.strip() == str(artifact)
