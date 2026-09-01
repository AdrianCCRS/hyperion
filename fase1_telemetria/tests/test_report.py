import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import report
from fase1_telemetria.validation import Verdict


def test_met04_tabla_por_factor_id_suma_exactamente_100():
    verdicts = [
        Verdict(True, None, "ok"), Verdict(True, None, "ok"), Verdict(True, None, "ok"),
        Verdict(False, "C02", "checksum"), Verdict(False, "I04", "sin muestras"),
        Verdict(False, "I04", "sin muestras"), Verdict(False, "I04", "sin muestras"),
    ]
    table = report.build_factor_table(verdicts)

    total_pct = sum(row.percentage for row in table)
    assert round(total_pct, 2) == 100.0
    counts = {row.factor_id: row.count for row in table}
    assert counts == {"accepted": 3, "C02": 1, "I04": 3}


def test_met04_tabla_suma_100_con_muchas_filas_pese_al_redondeo():
    # 7 corridas -> 1/7 = 14.285714...%, un caso clasico donde redondear cada
    # fila por separado se desvia de 100.
    verdicts = [Verdict(False, f"F{i}", "x") for i in range(7)]
    table = report.build_factor_table(verdicts)
    assert round(sum(row.percentage for row in table), 2) == 100.0


def test_build_factor_table_vacio():
    assert report.build_factor_table([]) == []


def test_met05_advertencia_visible_si_cv_supera_umbral():
    refs = SimpleNamespace(cv_pct=8.5)
    warning = report.calibration_stability_warning(refs, threshold_pct=5.0)
    assert warning is not None
    assert "8.50" in warning
    assert "D04" in warning


def test_met05_sin_advertencia_si_cv_esta_dentro_del_umbral():
    refs = SimpleNamespace(cv_pct=2.0)
    assert report.calibration_stability_warning(refs, threshold_pct=5.0) is None


def test_met05_sin_advertencia_si_no_hay_referencias():
    assert report.calibration_stability_warning(None) is None


def test_build_report_y_write_report(tmp_path):
    verdicts = [Verdict(True, None, "ok"), Verdict(False, "C02", "checksum")]
    refs = SimpleNamespace(cv_pct=9.0)

    data = report.build_report(
        campaign_id="camp01", verdicts=verdicts, calibration_references=refs,
        total_core_hours=1.5, cv_threshold_pct=5.0,
    )
    assert data["campaign_id"] == "camp01"
    assert data["total_runs"] == 2
    assert data["factor_table_percentage_sum"] == 100.0
    assert data["calibration_stability_warning"] is not None

    path = report.write_report(data, tmp_path)
    assert path.name == "campaign_report.json"
    loaded = json.loads(path.read_text())
    assert loaded == data


def test_cam08_overhead_stats_vacio_no_es_cero():
    stats = report.overhead_stats([])
    assert stats == {"overhead_pct_mean": None, "overhead_pct_cv": None, "overhead_pct_samples": 0}
    assert report.overhead_stats(None) == stats


def test_cam08_overhead_stats_calcula_media_y_cv():
    stats = report.overhead_stats([10.0, 10.0, 10.0])
    assert stats["overhead_pct_mean"] == 10.0
    assert stats["overhead_pct_cv"] == 0.0
    assert stats["overhead_pct_samples"] == 3


def test_cam08_advertencia_f44_si_cv_supera_umbral():
    warning = report.overhead_stability_warning([5.0, 50.0, 5.0], threshold_pct=10.0)
    assert warning is not None
    assert "F4.4" in warning


def test_cam08_sin_advertencia_si_cv_dentro_del_umbral():
    assert report.overhead_stability_warning([10.0, 11.0, 9.0], threshold_pct=10.0) is None
    assert report.overhead_stability_warning([], threshold_pct=10.0) is None


def test_build_report_incluye_estadisticas_de_overhead(tmp_path):
    verdicts = [Verdict(True, None, "ok")]
    data = report.build_report(
        campaign_id="camp01", verdicts=verdicts, overhead_pct_values=[5.0, 50.0, 5.0],
    )
    assert data["overhead_pct_samples"] == 3
    assert data["overhead_stability_warning"] is not None
