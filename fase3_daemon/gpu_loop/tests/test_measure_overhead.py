"""Pruebas locales de `measure_overhead.py` (Bloque C, ítem C6) -- solo la
lógica pura de agregación/percentiles, con `GpuPhaseDecision` construidos
a mano. El camino feliz (clasificador real + kernel GPU real +
`build_daemon_gpu_loop` real en un hilo) es de punta a punta y se mide en
el cluster (`scripts/pacca/hyp_measure_gpu_overhead.sbatch`), no aquí."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fase3_daemon.gpu_loop.controller import GpuPhaseDecision, GpuPhaseLabel
from fase3_daemon.gpu_loop.measure_overhead import OverheadReport


def _decision(inference_ns=None, actuation_ns=None) -> GpuPhaseDecision:
    return GpuPhaseDecision(
        label=GpuPhaseLabel.COMPUTE_BOUND, target_clock_mhz=0,
        inference_time_ns=inference_ns, actuation_time_ns=actuation_ns,
    )


def test_from_decisions_extrae_solo_valores_no_none():
    decisions = [
        _decision(inference_ns=100, actuation_ns=200),
        _decision(inference_ns=None, actuation_ns=None),  # nunca deberia pasar en produccion, pero no debe romper
        _decision(inference_ns=150, actuation_ns=250),
    ]
    report = OverheadReport.from_decisions(decisions)
    assert report.n_decisions == 3
    assert report.inference_ns == [100, 150]
    assert report.actuation_ns == [200, 250]


def test_percentiles_lista_vacia_no_lanza():
    report = OverheadReport(n_decisions=0, inference_ns=[], actuation_ns=[])
    summary = report.to_summary()
    assert summary["inference_ns"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}


def test_percentiles_un_solo_valor():
    report = OverheadReport(n_decisions=1, inference_ns=[500], actuation_ns=[500])
    summary = report.to_summary()
    assert summary["inference_ns"]["p50"] == 500
    assert summary["inference_ns"]["max"] == 500.0


def test_percentiles_max_es_el_maximo_real():
    values = list(range(1, 101))  # 1..100
    report = OverheadReport(n_decisions=100, inference_ns=values, actuation_ns=values)
    summary = report.to_summary()
    assert summary["inference_ns"]["max"] == 100.0
    # p50 de 1..100 (metodo inclusive) cae cerca del centro
    assert 45 <= summary["inference_ns"]["p50"] <= 55


def test_to_summary_tiene_n_decisions():
    report = OverheadReport(n_decisions=7, inference_ns=[1, 2, 3], actuation_ns=[1, 2, 3])
    assert report.to_summary()["n_decisions"] == 7
