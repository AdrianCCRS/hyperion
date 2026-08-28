from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
DUAL_SOURCES = [
    REPO / "kernels/dgemm/dgemm_bench.c",
    *sorted((REPO / "kernels/dual").glob("*_bench.c")),
    *sorted((REPO / "kernels/dual").glob("*_dispatch.cu")),
]


@pytest.mark.parametrize("source", DUAL_SOURCES, ids=lambda path: path.name)
def test_todo_kernel_dual_publica_contrato_cold_y_warm(source):
    text = source.read_text(encoding="utf-8")
    assert '#include "dispatch_timing.h"' in text or '#include "../dual/dispatch_timing.h"' in text
    assert text.count("print_dispatch_timing(") == 1
    for marker_variable in (
        "cold_t0_ns", "setup_complete_ns", "cold_t1_ns", "t0_ns", "t1_ns",
    ):
        assert marker_variable in text


def test_spmv_cpu_y_gpu_comparten_indices_int32():
    cpu = (REPO / "kernels/dual/spmv_cpu_bench.c").read_text(encoding="utf-8")
    gpu = (REPO / "kernels/dual/spmv_gpu_dispatch.cu").read_text(encoding="utf-8")
    assert "int **col_idx" in cpu
    assert "long **col_idx" not in cpu
    assert "CUSPARSE_INDEX_32I" in gpu


def test_spmv_gpu_transfiere_csr_completo_en_cold_y_warm():
    gpu = (REPO / "kernels/dual/spmv_gpu_dispatch.cu").read_text(encoding="utf-8")
    for operand in ("d_row_ptr", "d_col_idx", "d_values"):
        # Una copia en el primer despacho y otra dentro del bucle warm.
        assert gpu.count(f"cudaMemcpy({operand}") == 2
    assert "bytes_per_dispatch" in gpu


@pytest.mark.parametrize("source", sorted((REPO / "kernels/dual").glob("*_gpu_dispatch.cu")), ids=lambda path: path.name)
def test_dual_gpu_configura_blocking_sync_despues_de_cold_t0(source):
    text = source.read_text(encoding="utf-8")
    cold = text.index("long long cold_t0_ns = now_ns();")
    flags = text.index("cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)")
    first_allocation = text.index("cudaMalloc", flags)
    assert cold < flags < first_allocation


def test_cholesky_no_genera_btb_cubico_fuera_de_medicion():
    cpu = (REPO / "kernels/dual/cholesky_cpu_bench.c").read_text(encoding="utf-8")
    gpu = (REPO / "kernels/dual/cholesky_gpu_dispatch.cu").read_text(encoding="utf-8")
    assert "cblas_dgemm" not in cpu
    assert "for (long k = 0; k < n; ++k)" not in gpu
    assert "radius + 1.0" in cpu
    assert "radius + 1.0" in gpu
