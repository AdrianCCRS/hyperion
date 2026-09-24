"""La app B de GPU debe resolverse contra el catalogo REAL, con un kernel por fase y fuera de las familias de entrenamiento."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import load_catalog
from fase3_daemon.composite_apps.composite_gpu_unseen import PHASES, build_entries
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH

# familias/suites del entrenamiento del clasificador de GPU (metadata del RF sin reloj, n_families=16)
TRAIN_FAMILIES = {
    "dual_axpy", "dual_cholesky", "dual_fft", "dual_spmv", "dual_stencil", "gpu_cutlass_simt_conv2d",
    "gpu_cutlass_simt_dgemm_n4096", "minibude_cuda_bm1", "rajaperf_cuda", "rajaperf_stream", "rodinia_gaussian",
    "rodinia_heartwall", "rodinia_kmeans_gpu_N350000_D34_K200", "rodinia_lud",
}


def test_una_fase_memory_y_una_compute_en_gpu_con_argumentos_escalados():
    entries = build_entries(load_catalog(str(DEFAULT_CATALOG_PATH)))
    assert [e.phase_label_hint for e in entries] == ["memory_bound", "compute_bound"]
    assert all(e.device == "gpu" for e in entries)
    assert [e.exec_args for e in entries] == [a for _, a, _ in PHASES]


def test_los_kernels_no_estan_en_el_entrenamiento():
    assert not ({kid for kid, _, _ in PHASES} & TRAIN_FAMILIES)
