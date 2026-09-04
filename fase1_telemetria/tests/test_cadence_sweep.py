"""F1-GPU-002 Etapa A: pruebas de la comparación de cadencias NVML."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria.gpu_transition.cadence_sweep import compare_cadences, runbook, BASELINE_NS


def _summary(q_ns, *, steps):
    return {
        "probe_interval_ns_requested": q_ns,
        "observed_cadence": {"p50_delta_ns": q_ns, "p95_delta_ns": int(q_ns * 1.3)},
        "signal_step_analysis": {
            signal: {"n_consecutive_changes_lower_bound": value}
            for signal, value in steps.items()
        },
    }


def test_recomienda_10ms_si_conserva_los_escalones():
    summaries = [
        _summary(BASELINE_NS, steps={"power_mw": 10, "util_pct": 8, "sm_clock_mhz": 3}),
        _summary(10_000_000, steps={"power_mw": 9, "util_pct": 8, "sm_clock_mhz": 3}),   # ~90-100%
        _summary(50_000_000, steps={"power_mw": 3, "util_pct": 2, "sm_clock_mhz": 1}),   # colapsa
        _summary(100_000_000, steps={"power_mw": 1, "util_pct": 1, "sm_clock_mhz": 1}),
    ]
    r = compare_cadences(summaries)
    assert r["q_produccion_ns"] == 10_000_000
    assert "10 ms" in r["q_produccion_reason"]


def test_se_queda_en_5ms_si_10ms_ya_pierde_escalones():
    summaries = [
        _summary(BASELINE_NS, steps={"power_mw": 10, "util_pct": 10, "sm_clock_mhz": 4}),
        _summary(10_000_000, steps={"power_mw": 5, "util_pct": 5, "sm_clock_mhz": 2}),  # 50%
    ]
    r = compare_cadences(summaries)
    assert r["q_produccion_ns"] == BASELINE_NS
    assert "ninguna cadencia" in r["q_produccion_reason"]


def test_sin_baseline_no_supera_5ms():
    r = compare_cadences([_summary(10_000_000, steps={"power_mw": 5, "util_pct": 5, "sm_clock_mhz": 2})])
    assert r["q_produccion_ns"] == BASELINE_NS


def test_reporta_retencion_por_senal():
    summaries = [
        _summary(BASELINE_NS, steps={"power_mw": 10, "util_pct": 10, "sm_clock_mhz": 5}),
        _summary(10_000_000, steps={"power_mw": 9, "util_pct": 7, "sm_clock_mhz": 5}),
    ]
    r = compare_cadences(summaries)
    ret = r["per_interval"][10_000_000]["step_retention_vs_5ms"]
    assert ret["power_mw"] == 0.9
    assert ret["util_pct"] == 0.7


def test_mem_util_es_senal_de_decision():
    common = {
        "power_mw": 10,
        "util_pct": 10,
        "sm_clock_mhz": 5,
        "graphics_clock_mhz": 5,
    }
    summaries = [
        _summary(BASELINE_NS, steps={**common, "mem_util_pct": 10}),
        _summary(10_000_000, steps={**common, "mem_util_pct": 2}),
    ]
    r = compare_cadences(summaries)
    assert r["q_produccion_ns"] == BASELINE_NS
    assert "mem_util_pct" in r["signals_used_for_decision"]


def test_temperatura_y_energia_se_reportan_sin_gobernar_la_decision():
    baseline = {
        "power_mw": 10,
        "util_pct": 10,
        "mem_util_pct": 10,
        "sm_clock_mhz": 5,
        "graphics_clock_mhz": 5,
        "temperature_c": 4,
        "energy_mj": 10,
    }
    candidate = {**baseline, "temperature_c": 0, "energy_mj": 1}
    r = compare_cadences([
        _summary(BASELINE_NS, steps=baseline),
        _summary(10_000_000, steps=candidate),
    ])
    assert r["q_produccion_ns"] == 10_000_000
    assert "temperature_c" in r["signals_reported"]
    assert "temperature_c" not in r["signals_used_for_decision"]


def test_runbook_se_genera(tmp_path):
    rb = runbook("cuda_kernel_loop --secs 30", "0", 1275, tmp_path)
    assert rb.exists()
    txt = rb.read_text()
    assert "--dry-run-actuation" in txt
    for q in ("5000000", "10000000", "50000000", "100000000"):
        assert q in txt
