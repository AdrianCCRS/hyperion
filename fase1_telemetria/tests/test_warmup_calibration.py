"""F1-XDEV-002: pruebas herméticas de la calibración de warmup.

No requieren hardware ni campañas: construyen `windows.csv` sintéticos con un
transitorio conocido y verifican detección, criterio robusto entre >=3
repeticiones, estados explícitos, artefacto y propuesta al catálogo sin pisar.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import warmup_calibration as wc


# --------------------------------------------------------------- fixtures

def _pseudo(i: int) -> float:
    """[-1, 1) determinista, sin depender de random."""
    return ((i * 2654435761) % 10007) / 10007.0 * 2.0 - 1.0


def _write_cpu_windows(path: Path, *, warmup_s: float, span_s: float = 3.0,
                       dt_ms: float = 1.0, ipc_stable: float = 2.0) -> None:
    """Transitorio realista: durante el warmup el IPC fluctúa mucho (CV alto,
    cache llenándose / ramp-up); tras el warmup se asienta (CV bajo). El
    detector de CV busca exactamente el primer punto donde deja de fluctuar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(span_s * 1000 / dt_ms)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_start_ns", "t_end_ns", "ipc", "gpu_power_mw", "quality_status"])
        for i in range(n):
            t = int(i * dt_ms * 1e6)
            elapsed = i * dt_ms / 1000.0
            if elapsed < warmup_s:
                ipc = ipc_stable * (0.55 + 0.45 * _pseudo(i))  # ~[0.2, 2.0], CV alto
            else:
                ipc = ipc_stable * (1.0 + 0.01 * _pseudo(i))   # +-1%, CV bajo
            w.writerow([t, t + int(dt_ms * 1e6), round(ipc, 5), "", "ok"])


def _write_gpu_windows(path: Path, *, warmup_s: float, span_s: float = 6.0,
                       dt_ms: float = 100.0, util_stable: float = 98.0) -> None:
    """Warmup: blips de utilización (H2D, primeros lanzamientos) -> CV alto.
    Régimen: utilización sostenida -> CV bajo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(span_s * 1000 / dt_ms)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_start_ns", "t_end_ns", "ipc", "gpu_power_mw", "gpu_util_pct", "quality_status"])
        for i in range(n):
            t = int(i * dt_ms * 1e6)
            elapsed = i * dt_ms / 1000.0
            if elapsed < warmup_s:
                util = max(0.0, 30.0 + 30.0 * _pseudo(i))   # blips [0, 60], CV alto
            else:
                util = util_stable * (1.0 + 0.005 * _pseudo(i))
            w.writerow([t, t + int(dt_ms * 1e6), "", "250000", round(util, 3), "gpu_telemetry"])


# --------------------------------------------------------------- tests

def test_detecta_transitorio_cpu_y_aplica_margen():
    import tempfile
    d = Path(tempfile.mkdtemp())
    p = d / "windows.csv"
    _write_cpu_windows(p, warmup_s=0.5, span_s=3.0)
    r = wc.analyze_run(p)
    assert r.detected
    assert r.signal == "ipc"
    # instante bruto ~0.5s; propuesto = 1.2x
    assert 0.35 <= r.raw_warmup_s <= 0.75
    assert abs(r.proposed_warmup_s - r.raw_warmup_s * wc.MARGIN) < 1e-6


def test_gpu_usa_gpu_util_pct_y_changepoint_si_hace_falta():
    import tempfile
    d = Path(tempfile.mkdtemp())
    p = d / "windows.csv"
    _write_gpu_windows(p, warmup_s=2.0, span_s=8.0)
    r = wc.analyze_run(p)
    assert r.signal == "gpu_util_pct"
    assert r.detected
    assert 1.5 <= r.raw_warmup_s <= 3.5  # el escalón está en ~2s


def test_criterio_robusto_toma_el_maximo_entre_corridas(tmp_path):
    runs = []
    for i, w_s in enumerate([0.3, 0.9, 0.5]):  # la peor corrida arranca a 0.9s
        p = tmp_path / f"run{i}" / "windows.csv"
        _write_cpu_windows(p, warmup_s=w_s, span_s=4.0)
        runs.append((p, f"camp__k__REF__rep0{i+1}", "REF"))
    res = wc.calibrate_kernel(runs, kernel_ref="k", device="cpu")
    assert res.status == wc.STATUS_MEASURED
    assert res.n_runs_detected == 3
    # el warmup adoptado corresponde a la corrida de 0.9s, no al promedio
    assert res.warmup_seconds >= 0.9 * wc.MARGIN * 0.8
    assert res.raw_warmup_s_max >= 0.7


def test_menos_de_tres_repeticiones_no_es_measured(tmp_path):
    runs = []
    for i in range(2):
        p = tmp_path / f"run{i}" / "windows.csv"
        _write_cpu_windows(p, warmup_s=0.5, span_s=3.0)
        runs.append((p, f"c__k__REF__rep0{i+1}", "REF"))
    res = wc.calibrate_kernel(runs, kernel_ref="k", device="cpu")
    assert res.status == wc.STATUS_INSUFFICIENT
    assert res.warmup_seconds is None
    assert any("3" in n for n in res.notes)


def test_fallback_documentado_exige_razon_y_riesgo(tmp_path):
    p = tmp_path / "run0" / "windows.csv"
    # solo 1 corrida (< 3) -> nunca 'measured'; se admite un fallback documentado
    _write_cpu_windows(p, warmup_s=0.5, span_s=3.0)
    runs = [(p, "c__k__REF__rep01", "REF")]
    try:
        wc.calibrate_kernel(runs, kernel_ref="k", device="cpu", fallback_seconds=2.0)
        assert False, "debió exigir razón/riesgo"
    except ValueError:
        pass
    res = wc.calibrate_kernel(
        runs, kernel_ref="k", device="cpu",
        fallback_seconds=2.0, fallback_reason="heredado de campaña histórica",
        fallback_risk="puede sobre-excluir ~1s de datos válidos",
    )
    assert res.status == wc.STATUS_FALLBACK
    assert res.warmup_seconds == 2.0
    assert any("fallback" in n.lower() for n in res.notes)


def test_gpu_sin_senal_queda_insufficient_no_valor_arbitrario(tmp_path):
    p = tmp_path / "run0" / "windows.csv"
    # solo 4 muestras GPU -> ni CV ni changepoint resuelven
    _write_gpu_windows(p, warmup_s=0.0, span_s=0.4, dt_ms=100.0)
    res = wc.calibrate_kernel([(p, "c__k__REF__rep01", "REF")], kernel_ref="k", device="gpu")
    assert res.status == wc.STATUS_INSUFFICIENT
    assert res.warmup_seconds is None
    assert any("not_suitable" in n or "alargar" in n for n in res.notes)


def test_artefacto_json_y_csv(tmp_path):
    runs = []
    for i in range(3):
        p = tmp_path / f"run{i}" / "windows.csv"
        _write_cpu_windows(p, warmup_s=0.4, span_s=3.0)
        runs.append((p, f"c__k__REF__rep0{i+1}", "REF"))
    res = [wc.calibrate_kernel(runs, kernel_ref="k1", device="cpu")]
    json_path, csv_path = wc.write_artifact(res, tmp_path / "out")
    j = json.loads(json_path.read_text())
    assert j["schema"] == "f1-xdev-002/warmup_calibration/1"
    assert "k1" in j["per_kernel"]
    assert j["proposals"] and j["proposals"][0]["kernel_ref"] == "k1"
    assert csv_path.read_text().splitlines()[0].startswith("kernel_ref,")


def test_propuesta_al_catalogo_no_pisa_sin_apply(tmp_path):
    import yaml
    cat = tmp_path / "catalog.yaml"
    cat.write_text(yaml.safe_dump({"kernels": {
        "k1": {"id": "k1", "warmup_seconds": 1.0, "binary_checksum": "sha256:aaa"},
    }}))
    props = [{"kernel_ref": "k1", "device": "cpu", "proposed_warmup_seconds": 0.6,
              "status": "measured", "binary_checksum": "sha256:aaa"}]
    diff = wc.apply_proposals_to_catalog(cat, props, apply=False)
    assert diff["applied"] is False
    assert diff["diff"][0] == {"kernel_ref": "k1", "action": "update", "from": 1.0,
                               "to": 0.6, "status": "measured"}
    # sin apply, el archivo no cambió
    assert yaml.safe_load(cat.read_text())["kernels"]["k1"]["warmup_seconds"] == 1.0

    wc.apply_proposals_to_catalog(cat, props, apply=True)
    assert yaml.safe_load(cat.read_text())["kernels"]["k1"]["warmup_seconds"] == 0.6
    assert cat.with_suffix(".yaml.bak").exists()


def test_propuesta_se_salta_si_checksum_no_coincide(tmp_path):
    import yaml
    cat = tmp_path / "catalog.yaml"
    cat.write_text(yaml.safe_dump({"kernels": {"k1": {"id": "k1", "warmup_seconds": 1.0,
                                                      "binary_checksum": "sha256:aaa"}}}))
    props = [{"kernel_ref": "k1", "device": "cpu", "proposed_warmup_seconds": 0.6,
              "status": "measured", "binary_checksum": "sha256:DIFERENTE"}]
    diff = wc.apply_proposals_to_catalog(cat, props, apply=True)
    assert diff["diff"][0]["action"] == "skip"
    assert yaml.safe_load(cat.read_text())["kernels"]["k1"]["warmup_seconds"] == 1.0
