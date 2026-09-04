"""F1-GPU-002: pruebas del agregador de la matriz de transiciones de reloj GPU.

Cubren la regla de derivación conservadora (máximo sobre pares y réplicas, nunca
promedio), el manejo de timeouts / réplicas insuficientes, y el caso sin datos
estables. No requieren GPU ni el binario del probe -- solo dicts de resumen.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria.gpu_transition.aggregate_transition_matrix import (
    aggregate_summaries,
    conservative_transition_ns,
    load_summaries,
)


def _summary(from_clock, to_mhz, result="stable", cub_ns=1_000_000, cmd_lat_ns=3_000_000,
             metrics_valid=True):
    return {
        "from_clock": from_clock,
        "to_clock_mhz": to_mhz,
        "result": result,
        "dry_run_actuation": False,
        "restoration": {"confirmed": True},
        "gpu": {"uuid": "GPU-test", "driver_version": "test-driver"},
        "workload_checksum_sha256": "sha256:test",
        "transition_metrics": {
            "valid": metrics_valid,
            "conservative_upper_bound_ns": cub_ns,
            "command_latency_ns": cmd_lat_ns,
            "t_actuacion_ns": cub_ns,
        },
    }


def test_conservador_es_el_maximo_sobre_replicas_y_pares():
    summaries = [
        _summary("REF", 1410, cub_ns=8_000_000),
        _summary("REF", 1410, cub_ns=12_000_000),   # réplica peor del mismo par
        _summary("REF", 1410, cub_ns=9_000_000),
        _summary("1410", 900, cub_ns=40_000_000),   # otro par, mucho peor
        _summary("1410", 900, cub_ns=38_000_000),
        _summary("1410", 900, cub_ns=41_000_000),
    ]
    rep = conservative_transition_ns(summaries)
    assert rep["n_stable"] == 6
    # máximo global = peor réplica del peor par, NO un promedio
    assert rep["t_transicion_gpu_ns_conservative"] == 41_000_000
    assert rep["worst_pair"] == "1410->900"
    assert rep["per_pair"]["REF->1410"]["conservative_upper_bound_ns_max"] == 12_000_000
    assert rep["per_pair"]["1410->900"]["conservative_upper_bound_ns_max"] == 41_000_000
    assert rep["usable_for_policy"] is True
    assert rep["warnings"] == []


def test_timeout_no_aporta_cota_pero_si_advertencia():
    summaries = [
        _summary("REF", 1410, cub_ns=10_000_000),
        _summary("REF", 1410, cub_ns=11_000_000),
        _summary("REF", 1410, cub_ns=10_500_000),
        _summary("REF", 210, result="timeout", metrics_valid=False),
    ]
    rep = conservative_transition_ns(summaries)
    assert rep["n_timeout"] == 1
    assert rep["n_stable"] == 3
    # el par que sí convergió aporta la cota; el timeout no
    assert rep["t_transicion_gpu_ns_conservative"] == 11_000_000
    assert any("timeout" in w for w in rep["warnings"])
    # el par REF->210 quedó con 0 réplicas estables -> aviso de par
    assert any(w.startswith("par 'REF->210'") for w in rep["warnings"])
    assert rep["usable_for_policy"] is False


def test_replicas_insuficientes_marca_no_usable():
    summaries = [
        _summary("REF", 1410, cub_ns=10_000_000),
        _summary("REF", 1410, cub_ns=12_000_000),   # solo 2 réplicas
    ]
    rep = conservative_transition_ns(summaries)
    assert rep["t_transicion_gpu_ns_conservative"] == 12_000_000
    assert any("solo 2 réplica" in w for w in rep["warnings"])
    assert rep["usable_for_policy"] is False


def test_sin_datos_estables_devuelve_none():
    summaries = [
        _summary("REF", 210, result="timeout", metrics_valid=False),
        _summary("REF", 210, result="workload_inactive", metrics_valid=False),
    ]
    rep = conservative_transition_ns(summaries)
    assert rep["t_transicion_gpu_ns_conservative"] is None
    assert rep["n_stable"] == 0
    assert rep["usable_for_policy"] is False


def test_metrics_no_valido_se_ignora_aunque_result_sea_stable():
    summaries = [
        _summary("REF", 1410, result="stable", metrics_valid=False),
        _summary("REF", 1410, cub_ns=7_000_000),
        _summary("REF", 1410, cub_ns=8_000_000),
        _summary("REF", 1410, cub_ns=9_000_000),
    ]
    rep = conservative_transition_ns(summaries)
    # la fila con metrics.valid=False no cuenta como réplica estable
    assert rep["per_pair"]["REF->1410"]["stable_replicates"] == 3
    assert rep["t_transicion_gpu_ns_conservative"] == 9_000_000


def test_load_summaries_desde_directorio(tmp_path):
    d = tmp_path / "run_a"
    d.mkdir()
    (d / "gpu_clock_transition_summary.json").write_text(
        json.dumps(_summary("REF", 1410, cub_ns=5_000_000)), encoding="utf-8"
    )
    d2 = tmp_path / "run_b"
    d2.mkdir()
    (d2 / "gpu_clock_transition_summary.json").write_text(
        json.dumps(_summary("REF", 1410, cub_ns=6_000_000)), encoding="utf-8"
    )
    loaded = load_summaries([tmp_path])
    assert len(loaded) == 2
    assert all("_source_path" in s for s in loaded)

    rep = aggregate_summaries([tmp_path])
    assert rep["n_stable"] == 2
    assert rep["t_transicion_gpu_ns_conservative"] == 6_000_000


def test_direccion_del_par_importa():
    # REF->1410 y 1410->REF son pares distintos; no se funden.
    summaries = [
        _summary("REF", 1410, cub_ns=5_000_000),
        _summary("1410", 0, cub_ns=20_000_000),
    ]
    rep = conservative_transition_ns(summaries)
    assert set(rep["per_pair"].keys()) == {"REF->1410", "1410->0"}


def test_dry_run_o_restauracion_fallida_bloquean_politica():
    summaries = [_summary("REF", 1410, cub_ns=5_000_000) for _ in range(3)]
    summaries[0]["dry_run_actuation"] = True
    summaries[1]["restoration"] = {"confirmed": False}
    rep = conservative_transition_ns(summaries)
    assert rep["usable_for_policy"] is False
    assert any("dry_run" in warning for warning in rep["warnings"])
    assert any("restauración" in warning for warning in rep["warnings"])


def test_par_requerido_ausente_bloquea_politica():
    summaries = [_summary("REF", 1410) for _ in range(3)]
    rep = conservative_transition_ns(summaries, required_pairs={"REF->1410", "REF->210"})
    assert rep["usable_for_policy"] is False
    assert any("REF->210" in warning for warning in rep["warnings"])
