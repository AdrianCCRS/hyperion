"""La compuesta conjunta debe resolverse contra el catalogo REAL: 2 fases de CPU y 2 de GPU, con verdad de fase."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import load_catalog
from fase3_daemon.composite_apps.composite_joint import build_entries
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH


@pytest.mark.parametrize("which", ["known", "unseen"])
def test_dos_fases_de_cpu_y_dos_de_gpu_con_verdad_de_fase(which):
    entries = build_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), which)
    assert [e.device == "gpu" for e in entries] == [False, False, True, True]
    assert all(e.phase_label_hint in ("compute_bound", "memory_bound") for e in entries)


def test_conjunto_desconocido_falla():
    with pytest.raises(ValueError):
        build_entries({}, "otro")
