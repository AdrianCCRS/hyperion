"""Pruebas de `fase3_daemon/composite_apps/composite_unseen.py` (Bloque C,
ítem C2b, Aplicación B). La orquestación (`run_composite`/`resolve_entries`)
ya está probada exhaustivamente en `test_composite_known.py` -- aquí solo
se verifica lo que es propio de este módulo: que trae la secuencia inédita
correcta por default y que resuelve contra el catálogo real."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import load_catalog
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH, resolve_entries
from fase3_daemon.composite_apps.composite_unseen import DEFAULT_SEQUENCE


def test_default_sequence_es_el_par_ineditos():
    assert DEFAULT_SEQUENCE == ("cpu_xsbench_omp", "cpu_rsbench_omp")


def test_default_sequence_distinta_de_aplicacion_a():
    # Requisito del propio diseño (§0.1 Eje 1): A y B deben ser conjuntos
    # de familias disjuntos, o B no prueba nada distinto de A.
    from fase3_daemon.composite_apps.composite_known import DEFAULT_SEQUENCE as KNOWN_SEQUENCE
    assert set(DEFAULT_SEQUENCE).isdisjoint(set(KNOWN_SEQUENCE))


def test_default_sequence_resuelve_contra_el_catalogo_real():
    # No usa mocks: si algun dia estos IDs se renombran o se retiran del
    # catalogo real, este test debe fallar de forma obvia, no quedar
    # verde contra un catalogo sintetico que ya no representa el real.
    catalog = load_catalog(str(DEFAULT_CATALOG_PATH))
    entries = resolve_entries(catalog, DEFAULT_SEQUENCE)
    labels = {e.id: e.phase_label_hint for e in entries}
    assert labels == {"cpu_xsbench_omp": "memory_bound", "cpu_rsbench_omp": "compute_bound"}


def test_default_sequence_tiene_checksum_para_pacca_a100():
    catalog = load_catalog(str(DEFAULT_CATALOG_PATH))
    entries = resolve_entries(catalog, DEFAULT_SEQUENCE)
    for entry in entries:
        assert "pacca-a100" in entry.binary_checksum
