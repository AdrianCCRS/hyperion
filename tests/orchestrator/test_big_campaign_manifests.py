"""Validacion de los manifiestos de la campana suplementaria "big" (9
config_id nuevos GEMM/Cholesky/FFT en gemm_N8192/N12288/N16384) y de que la
campana full original quedo intacta (bug CAM-09: agregar los kernel_ref big
al manifiesto full ya usado cambia compute_protocol_fingerprint())."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import campaign as campaign_module
from orchestrator import catalog as catalog_module
from orchestrator import manifest as manifest_module

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "orchestrator" / "schemas"
CAMPAIGNS_DIR = SCHEMAS_DIR / "campaigns"

BIG_CPU_KERNELS = (
    "dual_gemm_cpu_N8192", "dual_gemm_cpu_N12288", "dual_gemm_cpu_N16384",
    "dual_fft_cpu_N8192", "dual_fft_cpu_N12288", "dual_fft_cpu_N16384",
    "dual_cholesky_cpu_N8192", "dual_cholesky_cpu_N12288", "dual_cholesky_cpu_N16384",
)
BIG_GPU_KERNELS = tuple(ref.replace("_cpu_", "_gpu_") for ref in BIG_CPU_KERNELS)


def _load(name: str):
    path = CAMPAIGNS_DIR / name
    manifest = manifest_module.load(path)
    catalog = catalog_module.load_catalog(str(manifest.catalog_path))
    return manifest, catalog


def test_manifiesto_cpu_full_no_contiene_kernels_big():
    manifest, _ = _load("campaign_pacca_dual_cpu_full.yaml")
    assert set(manifest.kernels).isdisjoint(BIG_CPU_KERNELS)
    assert len(manifest.kernels) == 68


def test_manifiesto_gpu_full_no_contiene_kernels_big():
    manifest, _ = _load("campaign_pacca_dual_gpu_full.yaml")
    assert set(manifest.kernels).isdisjoint(BIG_GPU_KERNELS)
    assert len(manifest.kernels) == 68


def test_manifiesto_cpu_big_carga_sin_error_y_resuelve_catalogo():
    manifest, catalog = _load("campaign_pacca_dual_cpu_big.yaml")
    assert set(manifest.kernels) == set(BIG_CPU_KERNELS)
    for kernel_ref in manifest.kernels:
        assert kernel_ref in catalog, f"kernel_ref huerfano: {kernel_ref}"


def test_manifiesto_gpu_big_carga_sin_error_y_resuelve_catalogo():
    manifest, catalog = _load("campaign_pacca_dual_gpu_big.yaml")
    assert set(manifest.kernels) == set(BIG_GPU_KERNELS)
    for kernel_ref in manifest.kernels:
        assert kernel_ref in catalog, f"kernel_ref huerfano: {kernel_ref}"


def test_campana_big_tiene_campaign_id_y_output_dir_propios():
    cpu_full, _ = _load("campaign_pacca_dual_cpu_full.yaml")
    cpu_big, _ = _load("campaign_pacca_dual_cpu_big.yaml")
    gpu_full, _ = _load("campaign_pacca_dual_gpu_full.yaml")
    gpu_big, _ = _load("campaign_pacca_dual_gpu_big.yaml")

    assert cpu_big.campaign_id != cpu_full.campaign_id
    assert gpu_big.campaign_id != gpu_full.campaign_id
    assert str(cpu_big.output_dir) != str(cpu_full.output_dir)
    assert str(gpu_big.output_dir) != str(gpu_full.output_dir)


def test_fingerprint_de_la_campana_big_distinto_del_full():
    # La razon de ser de un manifiesto separado: el fingerprint de la
    # campana big DEBE ser distinto del de la campana full (kernel_refs
    # diferentes cuentan en el payload hasheado, ver
    # compute_protocol_fingerprint en orchestrator/campaign.py).
    cpu_full, catalog_full = _load("campaign_pacca_dual_cpu_full.yaml")
    cpu_big, catalog_big = _load("campaign_pacca_dual_cpu_big.yaml")

    fp_full = campaign_module.compute_protocol_fingerprint(cpu_full, catalog_full)
    fp_big = campaign_module.compute_protocol_fingerprint(cpu_big, catalog_big)
    assert fp_full != fp_big


def test_fingerprint_del_full_no_cambio_respecto_a_antes_de_la_campana_big():
    # Confirma que revertir los 9 kernel_ref agregados restauro el
    # fingerprint original -- una prueba de regresion directa contra el bug
    # CAM-09 (agregar kernels a un manifiesto ya usado invalida las corridas
    # aceptadas).
    import subprocess

    original_yaml = subprocess.run(
        ["git", "show", "01a053d~1:orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_full.yaml"],
        check=True, text=True, capture_output=True, cwd=SCHEMAS_DIR.parents[1],
    ).stdout
    current_yaml = (CAMPAIGNS_DIR / "campaign_pacca_dual_cpu_full.yaml").read_text(encoding="utf-8")
    assert original_yaml == current_yaml
