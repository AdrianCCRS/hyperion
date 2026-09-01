"""Tests de derive_kernel_family / leave_one_familia_out (§2.1.1/§2.6 del
plan de realineación) -- añadidos durante la reconstrucción en 4 fases.

No son un test genérico: verifican, contra IDs reales tomados de
fase1_telemetria/catalog/catalog.yaml, que el agrupamiento por familia
algorítmica tiene la granularidad exacta que describe el plan (RAJAPerf
separado por sub-suite, GAP bfs/pr como familias distintas, tamaños/clases/
dispositivo NUNCA definiendo una familia nueva).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase2_clasificador.eval import protocol


@pytest.mark.parametrize("kernel_ref,expected_familia", [
    # dual_* : mismo algoritmo, dispositivo y tamaño son variantes.
    ("dual_gemm_cpu_N64", "dual_gemm"),
    ("dual_gemm_gpu_N16384", "dual_gemm"),
    ("dual_stencil_cpu_N3328", "dual_stencil"),
    # NPB: clase B/C del mismo problema, misma familia.
    ("npb_bt", "npb_bt"),
    ("npb_bt_c", "npb_bt"),
    ("npb_mg_c", "npb_mg"),
    # RAJAPerf: sub-suites separadas, tal como pide el plan.
    ("cpu_rajaperf_stream_add", "rajaperf_stream"),
    ("gpu_rajaperf_stream_copy", "rajaperf_stream"),
    ("cpu_rajaperf_lcals_first_sum", "rajaperf_lcals"),
    ("cpu_rajaperf_polybench_fdtd_2d", "rajaperf_polybench"),
    ("rajaperf_polybench_3mm_omp", "rajaperf_polybench"),
    ("cpu_rajaperf_basic_daxpy", "rajaperf_basic"),
    # RAJAPerf-CUDA sin sub-suite reconocible en el nombre.
    ("gpu_rajaperf_heat_3d", "rajaperf_cuda"),
    ("gpu_rajaperf_reduce3_int", "rajaperf_cuda"),
    # Rodinia: la variante CPU-OpenMP se une a su par GPU (mismo algoritmo).
    ("rodinia_lavamd", "rodinia_lavamd"),
    ("rodinia_lavamd_omp", "rodinia_lavamd"),
    ("rodinia_dwt2d", "rodinia_dwt2d"),
    ("rodinia_dwt2d_s2048", "rodinia_dwt2d"),
    # phasic: el parámetro p es una variante de carga, no un algoritmo.
    ("phasic_p010", "phasic"),
    ("gpu_phasic_p1000", "phasic"),
    ("ptrchase", "ptrchase"),
    # DGEMM: calibración cuBLAS, tamaño y CPU/OpenBLAS son un solo algoritmo.
    ("dgemm_n2048", "dgemm"),
    ("gpu_dgemm_calibration", "dgemm"),
    ("gpu_dgemm_n4096", "dgemm"),
    # GAP: bfs y pr son algoritmos distintos, NO deben compartir familia.
    ("cpu_gap_bfs", "gap_bfs"),
    ("cpu_gap_pr", "gap_pr"),
    # Sin variantes conocidas: familia = kernel_ref sin cambios.
    ("cpu_hpcg", "cpu_hpcg"),
    ("cpu_lulesh", "cpu_lulesh"),
    ("cpu_cholmod", "cpu_cholmod"),
])
def test_familias_esperadas(kernel_ref, expected_familia):
    assert protocol.derive_kernel_family(kernel_ref) == expected_familia


def test_gap_bfs_y_pr_no_comparten_familia():
    """Regresión explícita: agrupar todo GAP-Benchmark como una sola
    familia (por 'suite') fusionaría dos algoritmos distintos (BFS vs
    PageRank) en una sola unidad de validación -- exactamente el error que
    leave_one_familia_out existe para evitar."""
    assert protocol.derive_kernel_family("cpu_gap_bfs") != protocol.derive_kernel_family("cpu_gap_pr")


def test_leave_one_familia_out_agrupa_tamanos_del_mismo_algoritmo():
    df = pd.DataFrame({
        "kernel_ref": [
            "dual_gemm_cpu_N64", "dual_gemm_cpu_N64",
            "dual_gemm_gpu_N16384", "dual_gemm_gpu_N16384",
            "npb_bt", "npb_bt_c",
        ],
        "y": [0, 1, 0, 1, 0, 1],
    })
    folds = list(protocol.leave_one_familia_out(df))
    familias_vistas = sorted(f for _, _, f in folds)
    # dual_gemm_cpu_N64 y dual_gemm_gpu_N16384 son la MISMA familia
    # (dual_gemm) -- deben salir juntos en un solo pliegue, no en dos.
    assert familias_vistas == ["dual_gemm", "npb_bt"]

    dual_gemm_fold = next((tr, te) for tr, te, fam in folds if fam == "dual_gemm")
    idx_train, idx_test = dual_gemm_fold
    assert set(df.iloc[idx_test]["kernel_ref"]) == {
        "dual_gemm_cpu_N64", "dual_gemm_gpu_N16384",
    }
    assert set(df.iloc[idx_train]["kernel_ref"]) == {"npb_bt", "npb_bt_c"}


def test_assert_no_familia_leak_detecta_fuga_entre_tamanos_del_mismo_algoritmo():
    df = pd.DataFrame({
        "kernel_ref": ["dual_gemm_cpu_N64", "dual_gemm_gpu_N16384", "npb_bt"],
    })
    idx_train = np.array([0])       # dual_gemm_cpu_N64 -> familia dual_gemm
    idx_test = np.array([1, 2])     # dual_gemm_gpu_N16384 (misma familia!) + npb_bt
    with pytest.raises(AssertionError, match="fuga de familia"):
        protocol.assert_no_familia_leak(df, idx_train, idx_test)


def test_dataset_original_de_9_kernels_no_comparte_ninguna_familia():
    """El dataset histórico de train_phase.py (9 kernels, ninguno del
    catálogo ampliado) no debe verse afectado en la PARTICIÓN por este
    cambio -- leave_one_familia_out debe producir el mismo número de
    pliegues que leave_one_kernel_out sobre ese conjunto."""
    original_kernels = [
        "npb_bt", "npb_mg", "npb_cg", "npb_sp", "npb_ft", "npb_lu",
        "dgemm_n2048", "rodinia_lavamd_omp", "rajaperf_polybench_3mm_omp",
    ]
    df = pd.DataFrame({"kernel_ref": original_kernels})
    n_folds_kernel = len(list(protocol.leave_one_kernel_out(df)))
    n_folds_familia = len(list(protocol.leave_one_familia_out(df)))
    assert n_folds_kernel == n_folds_familia == len(original_kernels)
