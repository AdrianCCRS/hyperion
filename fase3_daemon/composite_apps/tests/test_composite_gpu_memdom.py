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


@pytest.mark.parametrize("which,n_mem", [("known", 3), ("unseen", 2)])
def test_domina_la_memoria(which, n_mem):
    assert sum(h == "memory_bound" for _, _, h in PHASES[which]) == n_mem
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


def test_sensitive_intercala_heartwall_entre_las_fases_de_memoria_de_ea():
    entries = build_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), "sensitive", heartwall_frames=15000, heartwall_repeats=2)
    assert [e.id for e in entries] == ["dual_stencil_gpu_N36864", "rodinia_heartwall", "rodinia_heartwall", "dual_spmv_gpu_N200000000"]
    assert [e.phase_label_hint for e in entries] == ["memory_bound", "compute_bound", "compute_bound", "memory_bound"]
    assert entries[1].exec_args == "data/heartwall/test.avi 15000"
    assert all(e.device == "gpu" for e in entries)
    assert "rodinia_heartwall" in TRAIN_FAMILIES


@pytest.mark.parametrize("frames,repeats", [(0, 1), (20001, 1), (1000, 0)])
def test_sensitive_rechaza_parametros_fuera_de_rango(frames, repeats):
    with pytest.raises(ValueError):
        build_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), "sensitive", heartwall_frames=frames, heartwall_repeats=repeats)
