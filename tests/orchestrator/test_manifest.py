from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import manifest


@pytest.fixture
def catalogo():
    return {
        "stream": SimpleNamespace(reports_bandwidth_stdout=True, reports_flops_stdout=False),
        "ert": SimpleNamespace(reports_bandwidth_stdout=False, reports_flops_stdout=True),
        "npb_a": SimpleNamespace(reports_bandwidth_stdout=False, reports_flops_stdout=False),
        "npb_b": SimpleNamespace(reports_bandwidth_stdout=False, reports_flops_stdout=False),
        "npb_c": SimpleNamespace(reports_bandwidth_stdout=False, reports_flops_stdout=False),
    }


@pytest.fixture
def campaign(tmp_path):
    return {
        "campaign_id": "prueba",
        "environment_tier": "local",
        "seed": 123,
        "output_dir": str(tmp_path / "salida"),
        "overwrite": False,
        "catalog_path": "catalog.yaml",
        "calibration": [{"kernel_ref": "stream"}, {"kernel_ref": "ert"}],
        "kernels": [{"kernel_ref": "npb_a"}, {"kernel_ref": "npb_b"}, {"kernel_ref": "npb_c"}],
        "frequency_levels": [
            {"id": "F0", "mode": "fixed", "fraction": 1.0},
            {"id": "F1", "mode": "fixed", "fraction": 0.5},
            {"id": "F2", "mode": "fixed", "fraction": 0.0},
            {"id": "REF", "mode": "native_governor"},
        ],
        "repetitions_per_combination": 5,
        "target_windows_per_repetition": 50,
        "interval_ns": 1_000_000,
        "running_ratio_min": 0.9,
        "cores": {"delegated_cpus": "2-5", "collector_cpu": 0, "consumer_cpu": 1, "numa_node_pin": 0},
        "smt_policy": "all_threads",
        "cgroup_path": None,
        "perf_enabled": True,
        "rapl": {"enabled": True, "domains": ["package"]},
        "gpu": {"enabled": False},
        "timeouts_seconds": {"ready": 15, "run": 300, "shutdown": 10},
    }


def cargar(tmp_path, monkeypatch, catalogo, campaign):
    ruta = tmp_path / "campaign.yaml"
    ruta.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    monkeypatch.setattr(manifest, "load_catalog", lambda _: catalogo)
    return manifest.load(ruta)


def test_man_t01_manifest_valido(tmp_path, monkeypatch, catalogo, campaign):
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.campaign_id == "prueba"
    assert resultado.cores.delegated_cpus == (2, 3, 4, 5)
    assert resultado.timeouts_seconds.run == 300


def test_man_t02_hpc_requiere_cgroup(tmp_path, monkeypatch, catalogo, campaign):
    campaign["environment_tier"] = "hpc_sc3"
    campaign.pop("cgroup_path")
    with pytest.raises(manifest.ManifestValidationError, match="cgroup_path"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_manifest_requiere_politica_smt(tmp_path, monkeypatch, catalogo, campaign):
    campaign.pop("smt_policy")
    with pytest.raises(manifest.ManifestValidationError, match="smt_policy"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t03_minimo_repeticiones(tmp_path, monkeypatch, catalogo, campaign):
    campaign["repetitions_per_combination"] = 2
    with pytest.raises(manifest.ManifestValidationError, match="mínimo es 3"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t04_directorio_existente_es_i07(tmp_path, monkeypatch, catalogo, campaign):
    Path(campaign["output_dir"]).mkdir()
    with pytest.raises(manifest.ManifestValidationError, match="I07"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t05_seed_ausente(tmp_path, monkeypatch, catalogo, campaign):
    campaign.pop("seed")
    with pytest.raises(manifest.ManifestValidationError, match="seed"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


@pytest.mark.parametrize("seed", [3.14, "abc", True])
def test_man_t05_seed_debe_ser_entero(tmp_path, monkeypatch, catalogo, campaign, seed):
    campaign["seed"] = seed
    with pytest.raises(manifest.ManifestValidationError, match="seed"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t06_solapamiento_de_cores(tmp_path, monkeypatch, catalogo, campaign):
    campaign["cores"]["collector_cpu"] = 3
    with pytest.raises(manifest.ManifestValidationError, match="solaparse"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t07_calibracion_incompleta(tmp_path, monkeypatch, catalogo, campaign):
    campaign["calibration"] = [{"kernel_ref": "stream"}]
    with pytest.raises(manifest.ManifestValidationError, match="I_ridge no es calculable"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t08_roles_solapados(tmp_path, monkeypatch, catalogo, campaign):
    campaign["calibration"].append({"kernel_ref": "npb_a"})
    with pytest.raises(manifest.ManifestValidationError, match="repetido entre roles"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t09_referencia_inexistente(tmp_path, monkeypatch, catalogo, campaign):
    campaign["kernels"] = [{"kernel_ref": "ausente"}]
    with pytest.raises(manifest.ManifestValidationError, match="kernel_ref inexistente"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t10_fraccion_fuera_de_rango(tmp_path, monkeypatch, catalogo, campaign):
    campaign["frequency_levels"][0]["fraction"] = 1.5
    with pytest.raises(manifest.ManifestValidationError, match="fraction"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t10_fraccion_negativa(tmp_path, monkeypatch, catalogo, campaign):
    campaign["frequency_levels"][0]["fraction"] = -0.1
    with pytest.raises(manifest.ManifestValidationError, match="fraction"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


@pytest.mark.parametrize(
    "levels, mensaje",
    [
        ([{"id": "F0", "mode": "fixed", "fraction": 1.0}], "exactamente un nivel native_governor"),
        (
            [
                {"id": "REF0", "mode": "native_governor"},
                {"id": "REF1", "mode": "native_governor"},
            ],
            "exactamente un nivel native_governor",
        ),
        (
            [
                {"id": "F0", "mode": "ondemand", "fraction": 1.0},
                {"id": "REF", "mode": "native_governor"},
            ],
            "debe ser fixed o native_governor",
        ),
    ],
)
def test_man_t10_modos_de_frecuencia(tmp_path, monkeypatch, catalogo, campaign, levels, mensaje):
    campaign["frequency_levels"] = levels
    with pytest.raises(manifest.ManifestValidationError, match=mensaje):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


@pytest.mark.parametrize(
    "field, value",
    [("running_ratio_min", 0.0), ("running_ratio_min", 1.1), ("interval_ns", 0), ("interval_ns", -1)],
)
def test_man_t11_ratio_e_intervalo_validos(tmp_path, monkeypatch, catalogo, campaign, field, value):
    campaign[field] = value
    with pytest.raises(manifest.ManifestValidationError, match=field):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t11_tamano_de_matriz_y_log_baseline(tmp_path, monkeypatch, catalogo, campaign, caplog):
    caplog.set_level("INFO", logger="orchestrator.manifest")
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert manifest.compute_matrix_size(resultado) == 60
    assert "×2 por baseline" in caplog.text
    assert "120 corridas" in caplog.text
