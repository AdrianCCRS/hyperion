"""F1-XDEV-003 / brecha G: pruebas del generador de manifiestos definitivos.

Verifican que: (1) se niega sin lista congelada de kernels; (2) el manifiesto
generado carga con el parser real; (3) la rejilla MHz se resuelve a fracciones
contra el rango del nodo; (4) sin datos del nodo queda marcado pendiente de
verificación y el gate falla.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria.campaigns import generate_final_manifest as gen
from common.hpc import manifest as manifest_module

_TEMPLATE = (Path(__file__).resolve().parents[1]
             / "catalog/campaigns/campaign_pacca_phase_coverage_cpu_screen.yaml")
_GPU_TEMPLATE = (Path(__file__).resolve().parents[1]
                 / "catalog/campaigns/campaign_pacca_phase_coverage_gpu_screen.yaml")
_CATALOG = str(Path(__file__).resolve().parents[1] / "catalog/catalog.yaml")


def test_se_niega_sin_lista_de_kernels():
    with pytest.raises(ValueError, match="lista congelada"):
        gen.generate(_TEMPLATE, [], device="cpu", campaign_id="x")


def test_cpu_resuelve_mhz_a_fraccion_contra_el_nodo(tmp_path):
    m = gen.generate(
        _TEMPLATE, ["npb_bt", "dgemm_n2048"], device="cpu",
        campaign_id="pacca_cpu_final_v1",
        cpu_range_khz=(800_000, 3_200_000),
    )
    lvls = m["frequency_levels"]
    assert lvls[0] == {"id": "REF", "mode": "native_governor"}
    fixed = [l for l in lvls if l["mode"] == "fixed"]
    assert len(fixed) == len(gen.CPU_GRID_MHZ_DEFAULT)
    # 3200 MHz -> tope del rango -> fraction 1.0 ; 800 MHz -> 0.0
    assert fixed[0]["fraction"] == 1.0
    assert fixed[-1]["fraction"] == 0.0
    assert m["metadata"]["frequency_grid_status"] == "resolved_against_node"
    ok, _ = gen.verify_grid_against_node(m)
    assert ok is True


def test_manifiesto_generado_carga_con_el_parser_real(tmp_path):
    m = gen.generate(
        _TEMPLATE, ["npb_bt", "npb_cg", "dgemm_n2048"], device="cpu",
        campaign_id="pacca_cpu_final_v1", cpu_range_khz=(800_000, 3_200_000),
        catalog_path=_CATALOG,
    )
    p = tmp_path / "campaign_final.yaml"
    p.write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True))
    loaded = manifest_module.load(p)
    assert loaded.campaign_id == "pacca_cpu_final_v1"
    ids = {l.id for l in loaded.frequency_levels}
    assert "REF" in ids
    assert len([l for l in loaded.frequency_levels if l.mode == "fixed"]) == len(gen.CPU_GRID_MHZ_DEFAULT)


def test_gpu_usa_gpu_frequency_levels_y_snap_a_soportados(tmp_path):
    supported = [210, 510, 810, 1110, 1170, 1230, 1290, 1350, 1410]
    m = gen.generate(
        _GPU_TEMPLATE, ["rodinia_lud", "rodinia_gaussian"], device="gpu",
        campaign_id="pacca_gpu_final_v1", gpu_supported_clocks_mhz=supported,
        catalog_path=_CATALOG,
    )
    assert m["frequency_levels"] == [{"id": "REF", "mode": "native_governor"}]
    assert "gpu_frequency_levels" in m
    used = [pt["used_mhz"] for pt in m["metadata"]["frequency_grid_resolved_points"]]
    assert set(used) <= set(supported)
    p = tmp_path / "gpu_final.yaml"
    p.write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True))
    loaded = manifest_module.load(p)
    assert loaded.gpu_frequency_levels is not None


def test_sin_datos_del_nodo_queda_pendiente_y_gate_falla():
    m = gen.generate(_TEMPLATE, ["npb_bt"], device="cpu", campaign_id="x")
    assert m["metadata"]["frequency_grid_status"] == "assumed_range_pending_node_verification"
    ok, msg = gen.verify_grid_against_node(m)
    assert ok is False
    assert "resolver" in msg


def test_main_cli(tmp_path):
    kf = tmp_path / "frozen.txt"
    kf.write_text("# familias congeladas del cribado\nnpb_bt\nnpb_cg\ndgemm_n2048\n")
    out = tmp_path / "final.yaml"
    rc = gen.main([
        "--template", str(_TEMPLATE), "--kernels-file", str(kf),
        "--device", "cpu", "--campaign-id", "pacca_cpu_final_v1",
        "--out", str(out), "--cpu-freq-range-khz", "800000", "3200000",
        "--catalog-path", _CATALOG,
    ])
    assert rc == 0
    assert out.exists()
    loaded = manifest_module.load(out)
    refs = {(k if isinstance(k, str) else k.kernel_ref) for k in loaded.kernels}
    assert refs == {"npb_bt", "npb_cg", "dgemm_n2048"}
