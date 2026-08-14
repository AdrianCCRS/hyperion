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
    # ARC-129: ausente en el YAML (campaign fixture no lo declara) -> None,
    # nunca se infiere -- preserva el comportamiento acoplado de siempre.
    assert resultado.gpu_frequency_levels is None


def test_arc129_gpu_frequency_levels_ausente_es_none(tmp_path, monkeypatch, catalogo, campaign):
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.gpu_frequency_levels is None


def test_arc129_gpu_frequency_levels_se_parsea_igual_que_frequency_levels(tmp_path, monkeypatch, catalogo, campaign):
    campaign["gpu_frequency_levels"] = [
        {"id": "GREF", "mode": "native_governor"},
        {"id": "GF0", "mode": "fixed", "fraction": 1.0},
    ]
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert [level.id for level in resultado.gpu_frequency_levels] == ["GREF", "GF0"]
    assert resultado.gpu_frequency_levels[1].fraction == 1.0


def test_arc129_gpu_frequency_levels_exige_exactamente_un_native_governor(tmp_path, monkeypatch, catalogo, campaign):
    campaign["gpu_frequency_levels"] = [{"id": "GF0", "mode": "fixed", "fraction": 1.0}]
    with pytest.raises(manifest.ManifestValidationError) as excinfo:
        cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert excinfo.value.rule_id == "MAN-10"


def test_arc129_gpu_frequency_levels_vacia_falla(tmp_path, monkeypatch, catalogo, campaign):
    campaign["gpu_frequency_levels"] = []
    with pytest.raises(manifest.ManifestValidationError) as excinfo:
        cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert excinfo.value.rule_id == "MAN-10"


def test_man12_kernel_calibration_en_kernels_falla(tmp_path, monkeypatch, catalogo, campaign):
    """ARC-94: campaign_pacca_gpu_ref.yaml mezclaba 4 kernels role=="calibration"
    directamente en `kernels:` -- MAN-12 lo bloquea ahora."""
    catalogo["npb_a"] = SimpleNamespace(reports_bandwidth_stdout=False, reports_flops_stdout=False, role="calibration")
    with pytest.raises(manifest.ManifestValidationError, match="MAN-12"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man12_kernel_dataset_en_calibration_falla(tmp_path, monkeypatch, catalogo, campaign):
    catalogo["stream"].role = "dataset"
    with pytest.raises(manifest.ManifestValidationError, match="MAN-12"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man12_roles_correctos_no_falla(tmp_path, monkeypatch, catalogo, campaign):
    catalogo["stream"].role = "calibration"
    catalogo["ert"].role = "calibration"
    catalogo["npb_a"].role = "dataset"
    catalogo["npb_b"].role = "dataset"
    catalogo["npb_c"].role = "dataset"
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.campaign_id == "prueba"


def test_arc73_projected_bytes_y_core_hours_ausentes_por_defecto(tmp_path, monkeypatch, catalogo, campaign):
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.projected_campaign_bytes is None
    assert resultado.remaining_core_hours is None
    assert resultado.projected_core_hours is None
    # ARC-102: ausente -> run_campaign() usa su propio default (1.0), nunca
    # se infiere aquí.
    assert resultado.load_threshold is None


def test_arc73_projected_bytes_y_core_hours_se_leen_del_manifiesto(tmp_path, monkeypatch, catalogo, campaign):
    campaign["projected_campaign_bytes"] = 200_000_000
    campaign["remaining_core_hours"] = 1000.0
    campaign["projected_core_hours"] = 10.0
    campaign["load_threshold"] = 2.5
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.projected_campaign_bytes == 200_000_000
    assert resultado.remaining_core_hours == 1000.0
    assert resultado.projected_core_hours == 10.0
    assert resultado.load_threshold == pytest.approx(2.5)


@pytest.mark.parametrize("field", ["projected_campaign_bytes", "remaining_core_hours", "projected_core_hours", "load_threshold"])
def test_arc73_projected_bytes_y_core_hours_negativos_fallan(tmp_path, monkeypatch, catalogo, campaign, field):
    campaign[field] = -1
    with pytest.raises(manifest.ManifestValidationError, match="MAN-00"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_man_t02_hpc_sc3_no_requiere_cgroup(tmp_path, monkeypatch, catalogo, campaign):
    # MAN-01/ARC-41: cgroup_path es opcional en TODOS los tiers, incluido
    # hpc_sc3 -- E06 (Cpus_allowed real) ya no depende de cgroups, y en
    # clústeres sin delegación de cgroup (felix) no hay forma de crear un
    # hijo vacío para el workload de todas formas.
    campaign["environment_tier"] = "hpc_sc3"
    campaign["cgroup_path"] = None
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.environment_tier == "hpc_sc3"
    assert resultado.cgroup_path is None


def test_man_t02_cgroup_path_sigue_siendo_campo_obligatorio_aunque_sea_null(tmp_path, monkeypatch, catalogo, campaign):
    # No se aceptan claves ausentes por descuido (MAN-00): cgroup_path debe
    # declararse explícitamente como null, no simplemente omitirse.
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


def test_arc88_gpu_interval_ns_ausente_por_defecto(tmp_path, monkeypatch, catalogo, campaign):
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.gpu_interval_ns is None


def test_arc88_gpu_interval_ns_se_lee_del_manifiesto(tmp_path, monkeypatch, catalogo, campaign):
    campaign["gpu_interval_ns"] = 5_000_000
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.gpu_interval_ns == 5_000_000


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_arc88_gpu_interval_ns_invalido_falla(tmp_path, monkeypatch, catalogo, campaign, value):
    campaign["gpu_interval_ns"] = value
    with pytest.raises(manifest.ManifestValidationError, match="gpu_interval_ns"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)


def test_arc116_uncore_ausente_por_defecto(tmp_path, monkeypatch, catalogo, campaign):
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.uncore == {}


def test_arc116_uncore_se_lee_del_manifiesto(tmp_path, monkeypatch, catalogo, campaign):
    campaign["uncore"] = {"enabled": True}
    resultado = cargar(tmp_path, monkeypatch, catalogo, campaign)
    assert resultado.uncore == {"enabled": True}


def test_arc116_uncore_invalido_falla(tmp_path, monkeypatch, catalogo, campaign):
    campaign["uncore"] = "not-an-object"
    with pytest.raises(manifest.ManifestValidationError, match="uncore"):
        cargar(tmp_path, monkeypatch, catalogo, campaign)
