"""Pruebas locales de `verify_gpu_coordination_e2e.py` (Bloque C, ítem C4)
-- solo la lógica de parseo/colapso, determinista y sin subprocesos
reales. El camino feliz (proceso C++ real leyendo lo que Python real
escribe) es intrínsecamente de punta a punta entre dos procesos del
sistema operativo y se verifica en el cluster
(`scripts/pacca/hyp_verify_gpu_coordination_e2e.sbatch`), no aquí."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fase3_daemon.gpu_loop.verify_gpu_coordination_e2e import (
    observed_transitions, parse_probe_output,
)


def test_parse_probe_output_lineas_simples():
    stdout = "100,0\n200,1\n300,1\n"
    assert parse_probe_output(stdout) == [(100, False), (200, True), (300, True)]


def test_parse_probe_output_vacio():
    assert parse_probe_output("") == []
    assert parse_probe_output("\n\n") == []


def test_observed_transitions_colapsa_repeticiones():
    readings = [(1, False), (2, False), (3, True), (4, True), (5, True), (6, False)]
    assert observed_transitions(readings) == [False, True, False]


def test_observed_transitions_sin_lecturas():
    assert observed_transitions([]) == []


def test_observed_transitions_una_sola_lectura():
    assert observed_transitions([(1, True)]) == [True]
