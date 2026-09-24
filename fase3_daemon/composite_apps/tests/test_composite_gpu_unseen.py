"""La app B de GPU debe resolverse contra el catalogo REAL, con verdad de fase y con familias fuera del entrenamiento."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import load_catalog
from fase3_daemon.composite_apps.composite_gpu_unseen import COMPUTE_KERNEL, DEFAULT_SEQUENCE, MEMORY_KERNEL
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH, resolve_entries

# familias del entrenamiento del clasificador de GPU (metadata del RF sin reloj, n_families=16)
TRAIN_FAMILIES = {
    "dual_axpy", "dual_cholesky", "dual_fft", "dual_spmv", "dual_stencil", "gpu_cutlass_simt_conv2d",
    "gpu_cutlass_simt_dgemm_n4096", "minibude_cuda_bm1", "rajaperf_cuda", "rajaperf_stream", "rodinia_gaussian",
    "rodinia_heartwall", "rodinia_kmeans_gpu_N350000_D34_K200", "rodinia_lud",
}


def test_secuencia_repite_kernels_con_verdad_de_fase_en_gpu():
    entries = resolve_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), DEFAULT_SEQUENCE)
    hints = [e.phase_label_hint for e in entries]
    assert hints[0] == "memory_bound" and hints[-1] == "compute_bound"
    assert set(hints) == {"memory_bound", "compute_bound"}
    assert all(e.device == "gpu" for e in entries)


def test_los_kernels_no_estan_en_el_entrenamiento():
    assert MEMORY_KERNEL not in TRAIN_FAMILIES and COMPUTE_KERNEL not in TRAIN_FAMILIES
