"""El escenario E debe resolverse contra el catalogo REAL: fases de GPU, dominadas por memoria, con las inéditas fuera del entrenamiento."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import load_catalog
from fase3_daemon.composite_apps.composite_gpu_memdom import PHASES, build_entries
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH

TRAIN_FAMILIES = {
    "dual_axpy", "dual_cholesky", "dual_fft", "dual_spmv", "dual_stencil", "gpu_cutlass_simt_conv2d",
    "gpu_cutlass_simt_dgemm_n4096", "minibude_cuda_bm1", "rajaperf_cuda", "rajaperf_stream", "rodinia_gaussian",
    "rodinia_heartwall", "rodinia_kmeans_gpu_N350000_D34_K200", "rodinia_lud", "gpu_rajaperf_stream_triad",
    "gpu_rajaperf_jacobi_2d", "gpu_rajaperf_heat_3d",
}


@pytest.mark.parametrize("which", ["known", "unseen"])
def test_resuelve_en_el_catalogo_y_es_de_gpu(which):
    entries = build_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), which)
    assert [e.id for e in entries] == [k for k, _, _ in PHASES[which]]
    assert all(e.device == "gpu" for e in entries)
    assert [e.phase_label_hint for e in entries] == [h for _, _, h in PHASES[which]]


@pytest.mark.parametrize("which", ["known", "unseen"])
def test_domina_la_memoria(which):
    assert sum(h == "memory_bound" for _, _, h in PHASES[which]) == 3
    assert sum(h == "compute_bound" for _, _, h in PHASES[which]) == 1


def test_los_inéditos_no_estan_en_el_entrenamiento():
    assert not ({kid for kid, _, _ in PHASES["unseen"]} & TRAIN_FAMILIES)


def test_los_vistos_son_familias_del_entrenamiento_con_verdad_declarada_de_fase2():
    mem = [kid for kid, _, h in PHASES["known"] if h == "memory_bound"]
    assert all(kid.startswith(("dual_stencil", "dual_spmv", "dual_axpy")) for kid in mem)


def test_falla_cerrado_si_falta_un_kernel():
    with pytest.raises(ValueError):
        build_entries({}, "known")
    with pytest.raises(ValueError):
        build_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), "otro")
