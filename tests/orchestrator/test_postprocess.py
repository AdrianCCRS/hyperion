import csv
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import postprocess


SAMPLES_HEADER = [
    "run_id", "repetition", "kernel", "label", "timestamp_ns", "tag",
    "instructions", "cycles", "cache_references", "cache_misses",
    "time_enabled_ns", "time_running_ns",
    "pkg_uj", "dram_uj", "pkg_delta_uj", "dram_delta_uj", "energy_delta_valid",
    "gpu_power_mw", "gpu_util_pct",
]


def _cpu_row(*, repetition, ts, instructions, cycles, cache_references, cache_misses,
             time_enabled, time_running):
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "CPU",
        "instructions": instructions, "cycles": cycles,
        "cache_references": cache_references, "cache_misses": cache_misses,
        "time_enabled_ns": time_enabled, "time_running_ns": time_running,
    }


def _energy_row(*, repetition, ts, pkg_delta_uj, dram_delta_uj=0, valid=True):
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "ENERGY",
        "pkg_delta_uj": pkg_delta_uj, "dram_delta_uj": dram_delta_uj,
        "energy_delta_valid": 1 if valid else 0,
    }


def _write_samples(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=SAMPLES_HEADER, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _context(**overrides) -> postprocess.WindowContext:
    defaults = dict(
        run_id="r", repetition=1, kernel_ref="npb_ep", node_id="felix-sc3",
        phase_label_hint="compute_bound", freq_level_id="REF",
        freq_khz_requested=None, freq_khz_applied=None, freq_khz_observed=2261000,
        binary_checksum="sha256:x", roofline_calibration_ref="cal/roofline_calibration.json",
        node_profile_ref="cal/node_profile.json", calibration_ref="cal/calibration_references.json",
        i_ridge_flops_per_byte=1.0, llc_line_size_bytes=64, run_flops_total=None,
        warmup_seconds=0.0, running_ratio_min=0.9, rapl_enabled=False,
        calibration_references=None,
    )
    defaults.update(overrides)
    return postprocess.WindowContext(**defaults)


def test_arc48_repeticion_de_campana_2_no_deja_windows_csv_vacio(tmp_path):
    # runner.py nunca pasa --repetitions al launcher, asi que
    # samples.csv SIEMPRE tiene "1" en su propia columna "repetition" --
    # sin importar si esta es la repeticion 1, 2 o 3 de la campana
    # (campaign.py). Antes del fix, build_windows() filtraba samples.csv
    # por context.repetition (2 en este test) y nunca encontraba nada,
    # devolviendo una lista vacia -- afecto el 100% de las repeticiones
    # 2 y 3 de los 7 kernels en la primera campana real (F4.4 extendido).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    windows = postprocess.build_windows(samples, _context(repetition=2, run_id="r__rep02"))

    assert len(windows) == 2
    assert windows[0]["repetition"] == 2  # metadata de salida: la repeticion de campana, no la del CSV
    assert windows[1]["quality_status"] != "first_sample_no_delta"  # la segunda fila si tuvo delta


def test_post01_primera_muestra_sin_delta_imputado(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
    ])
    windows = postprocess.build_windows(samples, _context())

    assert len(windows) == 1
    row = windows[0]
    assert row["quality_status"] == "first_sample_no_delta"
    assert row["window_index"] == 0
    assert row["t_start_ns"] is None
    assert row["delta_instructions"] is None
    assert row["ipc"] is None


def test_post04_ventana_normal_usa_delta_t_real(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_500_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000,
                 time_enabled=1_500_000, time_running=1_500_000),
    ])
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["delta_t_ns"] == 1_500_000  # no el --interval-ns nominal
    assert window["delta_instructions"] == 2_000_000
    assert window["ipc"] == pytest.approx(2.0)
    assert window["llc_miss_rate"] == pytest.approx(0.01)
    assert window["mpki"] == pytest.approx(0.5)
    assert window["ips"] == pytest.approx(2_000_000 / (1_500_000 / 1e9))
    assert window["running_ratio"] == pytest.approx(1.0)
    assert window["quality_status"] == "ok"


def test_post02_delta_negativo_marca_pmu_degraded_y_conserva_la_fila(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=1_500_000, cycles=1_100_000,
                 cache_references=100_500, cache_misses=1_050, time_enabled=1_100_000, time_running=1_100_000),
    ])
    windows = postprocess.build_windows(samples, _context())

    window = windows[1]
    assert window["quality_status"] == "pmu_degraded"
    assert window["delta_instructions"] == -500_000  # se conserva el valor crudo, no se oculta
    assert window["ipc"] is None  # no se deriva una tasa de un contador invalido


def test_post03_running_ratio_bajo_marca_pmu_degraded(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000,
                 time_enabled=1_000_000, time_running=500_000),  # running_ratio=0.5
    ])
    windows = postprocess.build_windows(samples, _context(running_ratio_min=0.9))

    window = windows[1]
    assert window["running_ratio"] == pytest.approx(0.5)
    assert window["quality_status"] == "pmu_degraded"


def test_post05_post06_energia_invalida_nunca_se_reporta_como_consumo_real(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _energy_row(repetition=1, ts=1_000_000_000, pkg_delta_uj=0, valid=False),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        _energy_row(repetition=1, ts=1_001_000_000, pkg_delta_uj=500_000, valid=False),
    ])
    windows = postprocess.build_windows(samples, _context(rapl_enabled=True, run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["energy_valid"] is False
    assert window["pkg_delta_uj"] is None  # nunca el 500000 "crudo" invalido
    assert window["power_w"] is None
    assert window["quality_status"] == "energy_invalid"


def test_post05_energia_valida_calcula_power_w(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _energy_row(repetition=1, ts=1_000_000_000, pkg_delta_uj=0, valid=False),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        _energy_row(repetition=1, ts=1_001_000_000, pkg_delta_uj=2_000_000, valid=True),  # 2 J en 1 ms
    ])
    windows = postprocess.build_windows(samples, _context(rapl_enabled=True, run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["energy_valid"] is True
    assert window["pkg_delta_uj"] == 2_000_000
    assert window["power_w"] == pytest.approx(2.0 / 0.001)
    assert window["quality_status"] == "ok"


def test_post07_ventana_en_warmup_se_conserva_pero_se_marca(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_000_500_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=500_000, time_running=500_000),
    ])
    # warmup_seconds=1.0 -> la ventana (t=1_000_000_000 a 1_000_500_000) cae dentro del warmup.
    windows = postprocess.build_windows(samples, _context(warmup_seconds=1.0, run_flops_total=1000.0))

    window = windows[1]
    assert window["quality_status"] == "warmup_excluded"
    assert window in windows  # sigue presente en windows.csv, no se descarta


def test_post08_bytes_movidos_cero_da_nan_y_no_divide_por_cero(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=0,  # sin misses -> bytes_moved_window = 0
                 time_enabled=1_000_000, time_running=1_000_000),
    ])
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["bytes_moved_window"] == 0
    assert math.isnan(window["operational_intensity"])
    assert window["quality_status"] == "intensity_undefined"
    assert window["phase_label_train"] is None


def test_post09_post10_flops_prorateado_y_bytes_con_line_size_del_node_profile(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        _cpu_row(repetition=1, ts=1_002_000_000, instructions=8_000_000, cycles=4_000_000,
                 cache_references=400_000, cache_misses=1_000, time_enabled=2_000_000, time_running=2_000_000),
    ])
    # run_total_instructions = 8_000_000 (ultima fila). Ventana 1: delta=4_000_000 -> mitad del total.
    windows = postprocess.build_windows(
        samples, _context(run_flops_total=1_000_000.0, llc_line_size_bytes=128)
    )

    window1 = windows[1]
    assert window1["flops_window_estimate"] == pytest.approx(500_000.0)  # mitad de 1_000_000
    assert window1["bytes_moved_window"] == 1_000 * 128  # linea real del node_profile, no 64 hardcodeado


def test_post11_phase_label_train_por_roofline_no_por_hint(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=1_000_000, cycles=500_000,
                 cache_references=10_000, cache_misses=1_000,  # muchos bytes movidos
                 time_enabled=1_000_000, time_running=1_000_000),
    ])
    # phase_label_hint dice compute_bound, pero I = flops/bytes sera bajo -> memory_bound.
    windows = postprocess.build_windows(
        samples, _context(phase_label_hint="compute_bound", run_flops_total=1.0, i_ridge_flops_per_byte=1.0)
    )

    window = windows[1]
    assert window["phase_label_hint"] == "compute_bound"
    assert window["phase_label_train"] == "memory_bound"  # I muy bajo (poco flops, muchos bytes)


def test_post12_post13_features_relativas_sin_recorte(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=1_000_000,  # ipc=4.0
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    refs = SimpleNamespace(ipc_p95=2.0, ips_p95=1.0, mpki_p95=1.0, miss_rate_p95=1.0)
    windows = postprocess.build_windows(samples, _context(calibration_references=refs))

    window = windows[1]
    assert window["ipc"] == pytest.approx(4.0)
    assert window["ipc_relative"] == pytest.approx(2.0)  # >1: informacion valida, no recortada


def test_post14_trazabilidad_en_cada_fila(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    windows = postprocess.build_windows(samples, _context())

    for window in windows:
        assert window["node_id"] == "felix-sc3"
        assert window["node_profile_ref"] == "cal/node_profile.json"
        assert window["calibration_ref"] == "cal/calibration_references.json"
        assert window["binary_checksum"] == "sha256:x"


def test_post16_write_windows_csv_escribe_columnas_absolutas_y_relativas(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    refs = SimpleNamespace(ipc_p95=2.0, ips_p95=1.0, mpki_p95=1.0, miss_rate_p95=1.0)
    windows = postprocess.build_windows(samples, _context(calibration_references=refs))
    out_path = postprocess.write_windows_csv(windows, tmp_path / "windows.csv")

    with out_path.open(newline="", encoding="utf-8") as windows_file:
        reader = csv.DictReader(windows_file)
        assert reader.fieldnames == list(postprocess.REQUIRED_OUTPUT_COLUMNS)
        rows = list(reader)
    assert rows[1]["ipc"] == "2.0"
    assert rows[1]["ipc_relative"] == "1.0"


def test_write_windows_csv_rechaza_quality_status_invalido(tmp_path):
    with pytest.raises(ValueError):
        postprocess.write_windows_csv(
            [{"quality_status": "no_es_valido"}], tmp_path / "windows.csv"
        )


def test_post15_run_postprocess_rechaza_calibracion_no_plausible(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    samples = run_dir / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
    ])
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()

    from orchestrator import calibration as calibration_module
    calibration_module.write_calibration(
        calibration_module.RooflineCalibration(
            campaign_id="c", timestamp="t", delegated_cpus="0-3", bw_pico_bytes_per_s=1.0,
            p_pico_flops_per_s=1.0, i_ridge_flops_per_byte=1.0, stream_raw_output="", ert_raw_output="",
            plausibility_check_passed=False, plausibility_message="D03: fuera de rango",
        ),
        cal_dir,
    )

    kernel_entry = SimpleNamespace(phase_label_hint="compute_bound", binary_checksum="sha256:x",
                                    flops_total_stdout_pattern=None)

    with pytest.raises(calibration_module.CalibrationError, match="CAL-06"):
        postprocess.run_postprocess(
            run_dir, run_id="r", repetition=1, kernel_ref="npb_ep", kernel_entry=kernel_entry,
            node_id="felix-sc3", freq_level_id="REF", calibration_dir=cal_dir,
        )


def test_post09_flops_total_directo_tiene_prioridad_sobre_tasa():
    entry = SimpleNamespace(
        flops_total_stdout_pattern=r"TOTAL_FLOPS\s+([0-9.]+)",
        flops_rate_stdout_pattern=r"Mop/s total\s*=\s*([0-9.]+)",
        runtime_seconds_stdout_pattern=r"Time in seconds\s*=\s*([0-9.]+)",
    )
    stdout = "TOTAL_FLOPS 42.0\nMop/s total     =    372.35\nTime in seconds =      0.09\n"
    assert postprocess.extract_run_flops_total(entry, stdout) == 42.0


def test_post09_npb_sin_total_directo_usa_tasa_por_tiempo():
    entry = SimpleNamespace(
        flops_total_stdout_pattern=None,
        flops_rate_stdout_pattern=r"Mop/s total\s*=\s*([0-9.]+)",
        runtime_seconds_stdout_pattern=r"Time in seconds\s*=\s*([0-9.]+)",
    )
    stdout = "Mop/s total     =    372.35\nTime in seconds =      0.09\n"
    resultado = postprocess.extract_run_flops_total(entry, stdout)
    assert resultado == pytest.approx(372.35 * 1e6 * 0.09)


def test_post09_sin_ningun_patron_devuelve_none():
    entry = SimpleNamespace(
        flops_total_stdout_pattern=None, flops_rate_stdout_pattern=None, runtime_seconds_stdout_pattern=None,
    )
    assert postprocess.extract_run_flops_total(entry, "cualquier salida") is None


def test_post09_falta_uno_de_los_dos_patrones_de_tasa_devuelve_none():
    entry = SimpleNamespace(
        flops_total_stdout_pattern=None,
        flops_rate_stdout_pattern=r"Mop/s total\s*=\s*([0-9.]+)",
        runtime_seconds_stdout_pattern=None,
    )
    stdout = "Mop/s total     =    372.35\n"
    assert postprocess.extract_run_flops_total(entry, stdout) is None


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_fixture_fake_samples_cubre_wrap_running_ratio_bajo_bytes_cero_y_warmup():
    """Ejercita tests/orchestrator/fixtures/fake_samples.csv de punta a punta:
    primera muestra, warmup, ventana ok, wrap/reset de contador, running_ratio
    bajo, y bytes_moved_window == 0 -- los cuatro casos que F2.5 pide cubrir."""
    windows = postprocess.build_windows(
        FIXTURES_DIR / "fake_samples.csv",
        _context(
            run_id="fake_run", kernel_ref="npb_ep", warmup_seconds=0.0005,
            running_ratio_min=0.9, run_flops_total=6_000_000.0,
        ),
    )

    statuses = [row["quality_status"] for row in windows]
    assert statuses[0] == "first_sample_no_delta"
    assert "warmup_excluded" in statuses
    assert "pmu_degraded" in statuses  # cubre tanto el wrap como el running_ratio bajo
    assert "intensity_undefined" in statuses  # cache_misses=0 en la ultima ventana
    assert "ok" in statuses

    # POST-02: el wrap se conserva crudo, no se enmascara con un delta positivo falso.
    wrap_window = next(w for w in windows if w["delta_instructions"] == -200_000)
    assert wrap_window["quality_status"] == "pmu_degraded"
