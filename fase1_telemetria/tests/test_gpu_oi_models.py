from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import gpu_oi_models  # noqa: E402


def test_gaussian_fan1_y_fan2_siguen_la_dimension_decreciente():
    fan1_first = gpu_oi_models.work_for_launch(
        "rodinia_gaussian", "_Z4Fan1PfS_ii", 0, {"size": 4096}
    )
    fan2_first = gpu_oi_models.work_for_launch(
        "rodinia_gaussian", "_Z4Fan2PfS_S_iii", 1, {"size": 4096}
    )
    fan1_last = gpu_oi_models.work_for_launch(
        "rodinia_gaussian", "_Z4Fan1PfS_ii", 8190 - 2, {"size": 4096}
    )
    fan2_last = gpu_oi_models.work_for_launch(
        "rodinia_gaussian", "_Z4Fan2PfS_S_iii", 8190 - 1, {"size": 4096}
    )

    assert fan1_first.flops == 4095
    assert fan2_first.flops == 2 * 4095 * 4096 + 2 * 4095
    assert fan1_last.flops == 1
    assert fan2_last.flops == 6
    assert fan1_first.operational_intensity > fan1_last.operational_intensity
    assert fan2_first.operational_intensity > fan2_last.operational_intensity


def test_gaussian_modela_los_8190_lanzamientos_de_n4096():
    for launch_index in range(8190):
        name = "Fan1" if launch_index % 2 == 0 else "Fan2"
        work = gpu_oi_models.work_for_launch(
            "rodinia_gaussian", name, launch_index, {"size": 4096}
        )
        assert work.flops > 0
        assert work.bytes_moved > 0


def test_modelo_falla_cerrado_para_kernel_no_registrado():
    with pytest.raises(KeyError, match="sin modelo analítico"):
        gpu_oi_models.work_for_launch("kernel_inexistente", "kernel", 0, {})


def test_gaussian_rechaza_nombre_o_indice_inconsistente():
    with pytest.raises(ValueError, match="inesperado"):
        gpu_oi_models.work_for_launch("rodinia_gaussian", "otro", 0, {"size": 8})
    with pytest.raises(ValueError, match="excede"):
        gpu_oi_models.work_for_launch("rodinia_gaussian", "Fan1", 14, {"size": 8})


def test_modelos_nativos_y_cutlass_tienen_trabajo_analitico_reproducible():
    axpy = gpu_oi_models.work_for_launch(
        "dual_axpy_gpu_N10000000", "_Z15axpy_kernel_valIddEv", 0, {"size": 4}
    )
    assert axpy.flops == 8
    assert axpy.bytes_moved == 96

    stencil = gpu_oi_models.work_for_launch(
        "dual_stencil_gpu_N1024", "jacobi_kernel", 0, {"size": 4}
    )
    assert stencil.flops == 20
    assert stencil.bytes_moved == 192

    gemm = gpu_oi_models.work_for_launch(
        "gpu_cutlass_simt_dgemm_n4096", "cutlass::gemm", 0, {"size": 4}
    )
    assert gemm.flops == 128
    assert gemm.bytes_moved == 384

    conv = gpu_oi_models.work_for_launch(
        "gpu_cutlass_simt_conv2d", "cutlass::conv", 0,
        {"n": 1, "h": 3, "w": 3, "c": 1, "k": 1, "r": 3, "s": 3, "p": 3, "q": 3},
    )
    assert conv.flops == 162
    assert conv.bytes_moved == 108


def test_spmv_rechaza_nombres_de_kernel_cusparse_desconocidos():
    events = (
        ("memcpy", "[memcpy]", 20),
        ("memcpy", "[memcpy]", 112),
        ("memcpy", "[memcpy]", 224),
        ("memcpy", "[memcpy]", 32),
        ("kernel", "csr_partition_kernel", 0),  # nombre viejo, ya no coincide (job 7588)
        ("kernel", "csrmv_v3_kernel", 0),
        ("memcpy", "[memcpy]", 32),
    )
    with pytest.raises(ValueError, match="kernels internos inesperados"):
        gpu_oi_models.work_for_operation("dual_spmv_gpu_N1000000", events, 0, {"size": 4})


def test_spmv_asigna_el_despacho_completo_a_sus_siete_eventos():
    events = (
        ("memcpy", "[memcpy]", 20),
        ("memcpy", "[memcpy]", 112),
        ("memcpy", "[memcpy]", 224),
        ("memcpy", "[memcpy]", 32),
        ("kernel", "_ZN8cusparse30binary_search_partition_kernelI...", 0),
        ("kernel", "_ZN8cusparse21load_balancing_kernelI...", 0),
        ("memcpy", "[memcpy]", 32),
    )
    work = gpu_oi_models.work_for_operation(
        "dual_spmv_gpu_N1000000", events, 0, {"size": 4}
    )
    assert work.flops == 56
    assert work.bytes_moved == 420


def test_cholesky_asigna_una_sola_factorizacion_a_sus_eventos_internos():
    events = (
        ("memcpy", "[memcpy]", 128),
        ("kernel", "xxtrf4_set_info_ker", 0),
        ("kernel", "getrf_wo_pivot_params", 0),
        ("memcpy", "[memcpy]", 4),
        ("memcpy", "[memcpy]", 128),
    )
    work = gpu_oi_models.work_for_operation(
        "dual_cholesky_gpu_N2048", events, 0, {"size": 4}
    )
    assert work.flops == pytest.approx(64 / 3)
    assert work.bytes_moved == 260


def test_fft_asigna_una_transformada_a_sus_tres_kernels_internos():
    events = (
        ("memcpy", "[memcpy]", 256),
        ("kernel", "regular_fft_factor", 0),
        ("kernel", "regular_fft_factor", 0),
        ("kernel", "vector_fft", 0),
        ("memcpy", "[memcpy]", 256),
    )
    work = gpu_oi_models.work_for_operation(
        "dual_fft_gpu_N4096", events, 0, {"size": 4}
    )
    assert work.flops == 320
    assert work.bytes_moved == 512


def test_kmeans_modela_la_iteracion_de_asignacion_sin_contar_setup():
    events = (
        ("memcpy", "[memcpy]", 40),
        ("memcpy", "[memcpy]", 24),
        ("kernel", "kmeansPoint", 0),
        ("memcpy", "[memcpy]", 40),
    )
    work = gpu_oi_models.work_for_operation(
        "rodinia_kmeans_gpu_N350000_D34_K200", events, 0,
        {"n_points": 10, "n_features": 3, "n_clusters": 2},
    )
    assert work.flops == 180
    assert work.bytes_moved == 224


def test_rajaperf_usa_las_cuentas_publicadas_por_el_suite():
    triad = gpu_oi_models.work_for_launch(
        "gpu_rajaperf_stream_triad", "rajaperf::stream::triad<256>", 0,
        {"elements": 4},
    )
    assert (triad.flops, triad.bytes_moved) == (8, 96)

    jacobi = gpu_oi_models.work_for_launch(
        "gpu_rajaperf_jacobi_2d", "poly_jacobi_2D_1", 0, {"side": 4},
    )
    assert (jacobi.flops, jacobi.bytes_moved) == (20, 128)

    heat = gpu_oi_models.work_for_launch(
        "gpu_rajaperf_heat_3d", "poly_heat_3D_1", 0, {"side": 4},
    )
    assert heat.flops == 120
    assert heat.bytes_moved == 320

    gemm = gpu_oi_models.work_for_launch(
        "gpu_rajaperf_gemm", "poly_gemm", 0, {"ni": 2, "nj": 2, "nk": 3},
    )
    assert (gemm.flops, gemm.bytes_moved) == (40, 128)


def test_lud_modela_la_secuencia_bloqueada_y_su_trabajo_decreciente():
    first_diag = gpu_oi_models.work_for_launch(
        "rodinia_lud", "lud_diagonal", 0, {"size": 64},
    )
    first_perimeter = gpu_oi_models.work_for_launch(
        "rodinia_lud", "lud_perimeter", 1, {"size": 64},
    )
    first_internal = gpu_oi_models.work_for_launch(
        "rodinia_lud", "lud_internal", 2, {"size": 64},
    )
    last_diag = gpu_oi_models.work_for_launch(
        "rodinia_lud", "lud_diagonal", 9, {"size": 64},
    )
    assert first_diag.flops == last_diag.flops == 2600
    assert first_perimeter.flops == 3 * 7936
    assert first_internal.flops == 9 * 8448
    with pytest.raises(ValueError, match="excede"):
        gpu_oi_models.work_for_launch("rodinia_lud", "lud_diagonal", 10, {"size": 64})


def test_minibude_usa_las_interacciones_y_huella_del_deck():
    work = gpu_oi_models.work_for_launch(
        "minibude_cuda_bm1", "fasten_main<1>", 0,
        {"poses": 2, "proteins": 3, "ligands": 5, "forcefields": 7},
    )
    assert work.flops == 1200
    assert work.bytes_moved == 296


def test_minibude_modela_el_kernel_auxiliar_de_version_sin_perder_cobertura():
    work = gpu_oi_models.work_for_launch(
        "minibude_cuda_bm1", "_Z7versionPi", 0,
        {"poses": 2, "proteins": 3, "ligands": 5, "forcefields": 7},
    )
    assert work.flops == 0
    assert work.bytes_moved == 4
    assert work.model_id == "minibude/cuda-version-int-store-v1"


def test_heartwall_usa_la_cota_de_convolucion_dominante_por_frame():
    work = gpu_oi_models.work_for_launch(
        "rodinia_heartwall", "kernel", 0,
        {"points": 2, "template_side": 3, "search_side": 5, "frame_bytes": 100},
    )
    assert work.flops == 900
    assert work.bytes_moved == 764
