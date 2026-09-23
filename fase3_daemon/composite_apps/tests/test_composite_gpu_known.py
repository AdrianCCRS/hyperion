"""La secuencia por defecto de la app A de GPU debe resolverse contra el catalogo REAL con verdad de fase declarada."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import load_catalog
from fase3_daemon.composite_apps.composite_gpu_known import DEFAULT_SEQUENCE
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH, resolve_entries


def test_secuencia_por_defecto_existe_en_el_catalogo_con_verdad_de_fase():
    entries = resolve_entries(load_catalog(str(DEFAULT_CATALOG_PATH)), DEFAULT_SEQUENCE)
    assert [e.phase_label_hint for e in entries] == ["compute_bound", "memory_bound"]
    assert all(e.device == "gpu" for e in entries)
