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
    "stalled_cycles_backend", "l2_lines_in_all",
    "fp_scalar_double", "fp_128b_packed_double", "fp_256b_packed_double", "fp_512b_packed_double",
    "time_enabled_ns", "time_running_ns",
    "pkg_uj", "dram_uj", "pkg_delta_uj", "dram_delta_uj", "energy_delta_valid",
    "gpu_power_mw", "gpu_util_pct", "gpu_mem_util_pct", "gpu_sm_clock_mhz", "gpu_energy_mj", "gpu_temperature_c",
]


def _cpu_row(*, repetition, ts, instructions, cycles, cache_references, cache_misses,
             time_enabled, time_running, stalled_cycles_backend=0, l2_lines_in_all=0,
             # ARC-97: "" by default (not 0), matching the launcher's real
             # empty-not-zero convention for a node/PMU that never opened
             # these -- most existing fixtures don't care about FLOPs
             # measurement and must keep exercising the flops_window_estimate
             # fallback path, exactly as they did before this column existed.
             fp_scalar_double="", fp_128b_packed_double="",
             fp_256b_packed_double="", fp_512b_packed_double=""):
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "CPU",
        "instructions": instructions, "cycles": cycles,
        "cache_references": cache_references, "cache_misses": cache_misses,
        "stalled_cycles_backend": stalled_cycles_backend,
        "l2_lines_in_all": l2_lines_in_all,
        "fp_scalar_double": fp_scalar_double,
        "fp_128b_packed_double": fp_128b_packed_double,
        "fp_256b_packed_double": fp_256b_packed_double,
        "fp_512b_packed_double": fp_512b_packed_double,
        "time_enabled_ns": time_enabled, "time_running_ns": time_running,
    }


def _energy_row(*, repetition, ts, pkg_delta_uj, dram_delta_uj=0, valid=True):
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "ENERGY",
        "pkg_delta_uj": pkg_delta_uj, "dram_delta_uj": dram_delta_uj,
        "energy_delta_valid": 1 if valid else 0,
    }


def _gpu_row(*, repetition, ts, gpu_power_mw, gpu_util_pct, gpu_mem_util_pct="",
             gpu_sm_clock_mhz="", gpu_energy_mj="", gpu_temperature_c=""):
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "GPU",
        "gpu_power_mw": gpu_power_mw, "gpu_util_pct": gpu_util_pct,
        "gpu_mem_util_pct": gpu_mem_util_pct,
        "gpu_sm_clock_mhz": gpu_sm_clock_mhz, "gpu_energy_mj": gpu_energy_mj,
        "gpu_temperature_c": gpu_temperature_c,
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


def test_stalled_cycles_backend_delta_y_ratio_se_calculan(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 stalled_cycles_backend=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 stalled_cycles_backend=400_000),
    ])
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["delta_stalled_cycles_backend"] == 400_000
    assert window["stall_backend_ratio"] == pytest.approx(0.4)
    assert window["quality_status"] == "ok"


def test_l2_lines_in_all_delta_y_bytes_moved_l2_proxy_se_calculan(tmp_path):
    # ARC-63: mismo patron que stalled_cycles_backend -- delta crudo y una
    # columna comparable a bytes_moved_window (mismo multiplicador de
    # tamano de linea), pensada como cruce independiente del sesgo de
    # bytes_moved_window (F3.4/ARC-33, cuantificado por kernel en ARC-60).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 l2_lines_in_all=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 l2_lines_in_all=2_000),
    ])
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0, llc_line_size_bytes=64))

    window = windows[1]
    assert window["delta_l2_lines_in_all"] == 2_000
    assert window["bytes_moved_l2_proxy"] == 2_000 * 64
    assert window["quality_status"] == "ok"


def test_l2_lines_in_all_no_soportado_no_afecta_bytes_moved_window(tmp_path):
    # Ausente en el CSV (nodo que no lo abre, ARC-63): debe comportarse como
    # "no medido aqui" -- ni pmu_degraded ni contamina bytes_moved_window,
    # que sigue calculandose exclusivamente con cache_misses.
    samples = tmp_path / "samples.csv"
    old_header = [c for c in SAMPLES_HEADER if c != "l2_lines_in_all"]
    with samples.open("w", newline="", encoding="utf-8") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=old_header, restval="")
        writer.writeheader()
        for row in [
            _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                     cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
            _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                     cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        ]:
            row.pop("l2_lines_in_all", None)
            writer.writerow(row)
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["delta_l2_lines_in_all"] is None
    assert window["bytes_moved_l2_proxy"] is None
    assert window["quality_status"] == "ok"
    assert window["bytes_moved_window"] == 1_000 * 64  # sigue basado en cache_misses, sin cambios


def test_stalled_cycles_backend_no_soportado_en_el_nodo_no_marca_pmu_degraded(tmp_path):
    # ARC-50: algunos nodos (confirmado empiricamente en paccaA100, Ice Lake
    # con kernel RHEL8) no pueden abrir PERF_COUNT_HW_STALLED_CYCLES_BACKEND
    # en absoluto (ENOENT del kernel, no un problema de permisos). El
    # launcher escribe la columna vacia en TODAS las filas de la corrida
    # cuando eso pasa (nunca "0", que se leeria como una medicion real) --
    # incluye tambien samples.csv de antes de este cambio, sin la columna.
    # Ambos casos son "no medido en este nodo", no una anomalia de PMU: no
    # deben marcar pmu_degraded ni impedir que el resto de metricas core
    # (ipc/ips/mpki) se calculen normalmente.
    samples = tmp_path / "samples.csv"
    old_header = [c for c in SAMPLES_HEADER if c != "stalled_cycles_backend"]
    with samples.open("w", newline="", encoding="utf-8") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=old_header, restval="")
        writer.writeheader()
        for row in [
            _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                     cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
            _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                     cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        ]:
            row.pop("stalled_cycles_backend", None)
            writer.writerow(row)
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["delta_stalled_cycles_backend"] is None
    assert window["stall_backend_ratio"] is None
    assert window["quality_status"] == "ok"
    assert window["ipc"] is not None  # el resto de metricas core no se contamina


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


def test_arc56_power_w_usa_el_intervalo_real_de_rapl_no_el_de_la_ventana_cpu(tmp_path):
    # Reproduce el escenario que causaba picos de power_w de decenas de kW:
    # una ventana de CPU anomalamente corta (10 ns, jitter de muestreo)
    # emparejada con un delta de energia que en realidad abarca el
    # intervalo normal de RAPL (aqui 1 ms, entre las dos muestras ENERGY).
    # Antes del fix, power_w dividia pkg_delta_uj por los 10 ns de la
    # ventana CPU -> 2e8 W. Con el fix, usa los 1_000_000 ns reales entre
    # las dos muestras ENERGY que produjeron ese delta -> 2000 W.
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _energy_row(repetition=1, ts=999_000_010, pkg_delta_uj=0, valid=False),
        _cpu_row(repetition=1, ts=1_000_000_010, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        _energy_row(repetition=1, ts=1_000_000_010, pkg_delta_uj=2_000_000, valid=True),  # 2 J, 1 ms real de RAPL
    ])
    windows = postprocess.build_windows(samples, _context(rapl_enabled=True, run_flops_total=1_000_000.0))

    window = windows[1]
    assert window["delta_t_ns"] == 10  # la ventana CPU en si sigue siendo la real, sin tocar
    assert window["energy_valid"] is True
    assert window["pkg_delta_uj"] == 2_000_000
    assert window["power_w"] == pytest.approx(2.0 / 0.001)  # 1 ms real de RAPL, no 10 ns de CPU
    assert window["power_w"] < 1_000_000  # nunca mas los ~2e8 W del calculo roto


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
    # ARC-97: sin columnas fp_* pobladas (nodo sin FP_ARITH_INST_RETIRED),
    # operational_intensity debe seguir cayendo al prorrateo, exactamente
    # como antes de que existiera la medicion directa.
    assert window1["flops_measured_window"] is None
    assert window1["flops_source"] == "estimated"


def test_arc97_flops_medidos_por_hardware_reemplazan_al_prorrateo(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 fp_scalar_double=0, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 # 10 scalar*1 + 10 128B*2 + 10 256B*4 + 10 512B*8 = 10+20+40+80 = 150 flops
                 fp_scalar_double=10, fp_128b_packed_double=10,
                 fp_256b_packed_double=10, fp_512b_packed_double=10),
    ])
    # run_flops_total deliberadamente muy distinto del valor medido, para que
    # el assert de abajo solo pueda pasar si de verdad se prefirio la medida
    # por hardware sobre el prorrateo (que daria 1_000_000.0, no 150.0).
    windows = postprocess.build_windows(
        samples, _context(run_flops_total=1_000_000.0, llc_line_size_bytes=128)
    )

    window1 = windows[1]
    assert window1["flops_measured_window"] == 150.0
    assert window1["flops_source"] == "measured"
    assert window1["flops_window_estimate"] == pytest.approx(1_000_000.0)  # se sigue calculando (auditoria)
    assert window1["operational_intensity"] == pytest.approx(150.0 / (1_000 * 128))  # usa la medida, no el prorrateo


def test_arc97_delta_fp_negativo_marca_pmu_degraded(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 fp_scalar_double=50, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 # fp_scalar_double bajo de 50 a 10: delta negativo, mismo
                 # tratamiento que un contador de nucleo que retrocede.
                 fp_scalar_double=10, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
    ])
    windows = postprocess.build_windows(samples, _context(run_flops_total=1_000_000.0))

    window1 = windows[1]
    assert window1["quality_status"] == "pmu_degraded"
    assert window1["flops_measured_window"] is None  # valid_counters=False lo bloquea


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


def test_arc70_filas_gpu_se_incluyen_como_passthrough_no_ventaneado(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=36324, gpu_util_pct=0),
        _gpu_row(repetition=1, ts=1_000_600_000, gpu_power_mw=36486, gpu_util_pct=15),
    ])
    windows = postprocess.build_windows(samples, _context())

    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert len(gpu_rows) == 2
    # ARC-70: passthrough puro -- ninguno de los campos de ventana de CPU
    # (Roofline, PMU, energía) se calcula ni se infiere para estas filas.
    assert gpu_rows[0]["gpu_power_mw"] == 36324
    assert gpu_rows[0]["gpu_util_pct"] == 0
    assert gpu_rows[0]["t_start_ns"] is None
    assert gpu_rows[0]["delta_t_ns"] is None
    assert gpu_rows[0]["operational_intensity"] is None
    assert gpu_rows[0]["phase_label_train"] is None
    assert gpu_rows[1]["gpu_power_mw"] == 36486
    assert gpu_rows[1]["gpu_util_pct"] == 15
    # Las ventanas de CPU no se contaminan con las columnas de GPU.
    cpu_rows = [w for w in windows if w["quality_status"] != "gpu_telemetry"]
    assert all(w["gpu_power_mw"] is None for w in cpu_rows)


def test_arc94_filas_gpu_excluyen_calentamiento(tmp_path):
    """ARC-94: antes de este cambio, warmup_seconds del catalogo nunca se
    comparaba contra el timestamp de la muestra GPU -- todas las filas
    quedaban 'gpu_telemetry' sin importar si caian antes del calentamiento
    declarado."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        # warmup_seconds=0.2 -> warmup_end_ns = 1_000_000_000 + 200_000_000 = 1_200_000_000
        _gpu_row(repetition=1, ts=1_050_000_000, gpu_power_mw=36324, gpu_util_pct=0),   # dentro del warmup
        _gpu_row(repetition=1, ts=1_199_999_999, gpu_power_mw=36400, gpu_util_pct=1),   # justo dentro
        _gpu_row(repetition=1, ts=1_200_000_000, gpu_power_mw=45000, gpu_util_pct=80),  # justo fuera
        _gpu_row(repetition=1, ts=1_500_000_000, gpu_power_mw=46000, gpu_util_pct=90),  # bien fuera
    ])
    windows = postprocess.build_windows(samples, _context(warmup_seconds=0.2))

    gpu_windows = [w for w in windows if w.get("gpu_power_mw") is not None]
    gpu_windows.sort(key=lambda w: w["t_end_ns"])
    assert [w["quality_status"] for w in gpu_windows] == [
        "warmup_excluded", "warmup_excluded", "gpu_telemetry", "gpu_telemetry",
    ]
    usable = [w for w in gpu_windows if w["quality_status"] == "gpu_telemetry"]
    assert len(usable) == 2
    assert usable[0]["gpu_power_mw"] == 45000


def test_arc95_filas_gpu_calculan_delta_de_energia(tmp_path):
    """ARC-95: gpu_energy_mj es un contador acumulado (igual que pkg_uj de
    RAPL) -- sin un delta por ventana no es insumo utilizable para EDP de
    GPU. Primera fila sin predecesor -> invalida, igual que RAPL."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_100_000, gpu_power_mw=45000, gpu_util_pct=80,
                  gpu_energy_mj=1_000_000),
        _gpu_row(repetition=1, ts=1_000_200_000, gpu_power_mw=45000, gpu_util_pct=80,
                  gpu_energy_mj=1_004_500),
        _gpu_row(repetition=1, ts=1_000_300_000, gpu_power_mw=45000, gpu_util_pct=80,
                  gpu_energy_mj=1_009_000),
    ])
    windows = postprocess.build_windows(samples, _context())
    gpu_rows = sorted(
        (w for w in windows if w["quality_status"] == "gpu_telemetry"),
        key=lambda w: w["t_end_ns"],
    )
    assert gpu_rows[0]["gpu_energy_delta_mj"] is None
    assert gpu_rows[0]["gpu_energy_valid"] is False
    assert gpu_rows[1]["gpu_energy_delta_mj"] == 4500
    assert gpu_rows[1]["gpu_energy_valid"] is True
    assert gpu_rows[2]["gpu_energy_delta_mj"] == 4500
    assert gpu_rows[2]["gpu_energy_valid"] is True


def test_arc95_filas_gpu_energia_invalida_si_el_contador_retrocede(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_100_000, gpu_power_mw=45000, gpu_util_pct=80,
                  gpu_energy_mj=1_000_000),
        _gpu_row(repetition=1, ts=1_000_200_000, gpu_power_mw=45000, gpu_util_pct=80,
                  gpu_energy_mj=500),  # el contador retrocedio (reinicio del driver)
    ])
    windows = postprocess.build_windows(samples, _context())
    gpu_rows = sorted(
        (w for w in windows if w["quality_status"] == "gpu_telemetry"),
        key=lambda w: w["t_end_ns"],
    )
    assert gpu_rows[1]["gpu_energy_delta_mj"] is None
    assert gpu_rows[1]["gpu_energy_valid"] is False


def test_arc94_filas_gpu_propagan_reloj_sm_energia_y_temperatura(tmp_path):
    """ARC-94: GpuSample ganó sm_clock_mhz/energy_mj/temperature_c -- sin
    reloj SM observado no se puede confirmar que un nivel DVFS de GPU se
    mantuvo durante la corrida, sin energía acumulada no hay insumo para
    EDP de GPU, y sin temperatura no se puede detectar contaminación
    térmica."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=45000, gpu_util_pct=80,
                  gpu_sm_clock_mhz=1350, gpu_energy_mj=123456789, gpu_temperature_c=62),
    ])
    windows = postprocess.build_windows(samples, _context())

    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert len(gpu_rows) == 1
    assert gpu_rows[0]["gpu_sm_clock_mhz"] == 1350
    assert gpu_rows[0]["gpu_energy_mj"] == 123456789
    assert gpu_rows[0]["gpu_temperature_c"] == 62


def test_arc94_filas_gpu_sin_metricas_nuevas_quedan_none(tmp_path):
    """Corridas del launcher previas a este cambio (o driver sin soporte)
    no deben romper el postprocesamiento -- las 3 columnas nuevas quedan
    None, no un error."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=45000, gpu_util_pct=80),
    ])
    windows = postprocess.build_windows(samples, _context())

    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert gpu_rows[0]["gpu_sm_clock_mhz"] is None
    assert gpu_rows[0]["gpu_energy_mj"] is None
    assert gpu_rows[0]["gpu_temperature_c"] is None


def test_arc94_filas_gpu_propagan_utilizacion_de_memoria(tmp_path):
    """ARC-94 (segunda ronda): nvmlUtilization_t trae .gpu Y .memory --
    solo se conservaba .gpu. Un kernel con trafico de memoria alto pero
    bajo uso de SM podia parecer 'ocioso' mirando solo gpu_util_pct."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=45000, gpu_util_pct=15,
                  gpu_mem_util_pct=78),
    ])
    windows = postprocess.build_windows(samples, _context())
    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert gpu_rows[0]["gpu_util_pct"] == 15
    assert gpu_rows[0]["gpu_mem_util_pct"] == 78


def test_arc94_filas_gpu_usan_su_propio_archivo_de_calibracion(tmp_path):
    """ARC-94 (segunda ronda): antes de este cambio, roofline_calibration_ref
    de una fila GPU apuntaba al archivo de calibración de CPU (heredado de
    _base_row()) aunque phase_label_train se calculó con
    gpu_i_ridge_flops_per_byte -- la columna de trazabilidad mentía sobre
    qué archivo produjo la etiqueta."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=45000, gpu_util_pct=80),
    ])
    windows = postprocess.build_windows(
        samples,
        _context(
            roofline_calibration_ref="cal/roofline_calibration_REF.json",
            gpu_roofline_calibration_ref="cal/roofline_calibration_REF_fp32.json",
            gpu_operational_intensity=1233.0, gpu_i_ridge_flops_per_byte=3.36,
        ),
    )
    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    cpu_rows = [w for w in windows if w["quality_status"] != "gpu_telemetry"]
    assert gpu_rows[0]["roofline_calibration_ref"] == "cal/roofline_calibration_REF_fp32.json"
    assert all(w["roofline_calibration_ref"] == "cal/roofline_calibration_REF.json" for w in cpu_rows)


def test_arc80_filas_gpu_calculan_phase_label_train_con_ridge_de_gpu(tmp_path):
    # ARC-80: cuando el contexto trae la intensidad operacional (medida
    # offline con ncu) y el ridge de GPU calibrado, las filas GPU SI deben
    # traer una etiqueta -- corrige el error de diseño de ARC-72 (dejarlas
    # sin etiquetar "porque eso es Fase 2", cuando en realidad es Fase 1).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=36324, gpu_util_pct=0),
    ])
    # rodinia_backprop: 0.087 FLOP/byte, muy por debajo de cualquier ridge
    # FP32 realista -- memory_bound sin ambiguedad.
    windows = postprocess.build_windows(
        samples,
        _context(gpu_operational_intensity=0.087, gpu_i_ridge_flops_per_byte=7.28),
    )
    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert len(gpu_rows) == 1
    assert gpu_rows[0]["operational_intensity"] == pytest.approx(0.087)
    assert gpu_rows[0]["i_ridge_used"] == pytest.approx(7.28)
    assert gpu_rows[0]["phase_label_train"] == "memory_bound"


def test_arc80_filas_gpu_compute_bound_con_ridge_de_gpu(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=36324, gpu_util_pct=0),
    ])
    # rodinia_lavamd: 1233 FLOP/byte -- compute_bound sin ambiguedad.
    windows = postprocess.build_windows(
        samples,
        _context(gpu_operational_intensity=1233.0, gpu_i_ridge_flops_per_byte=3.36),
    )
    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert gpu_rows[0]["phase_label_train"] == "compute_bound"


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
            freq_level_id="REF",
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


def test_arc78_run_postprocess_pide_el_ridge_del_propio_freq_level_id(tmp_path, monkeypatch):
    # P_pico escala con el reloj, BW_pico no (ARC-78) -- una ventana medida
    # a un nivel de frecuencia FG_1 nunca debe clasificarse contra el ridge
    # calibrado a REF. run_postprocess() debe pedirle a load_calibration()
    # el archivo del propio freq_level_id de la corrida, no uno fijo.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_samples(run_dir / "samples.csv", [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
    ])
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()

    requested_freq_level_ids = []

    def fake_load_calibration(calibration_dir, freq_level_id=""):
        requested_freq_level_ids.append(freq_level_id)
        return SimpleNamespace(i_ridge_flops_per_byte=1.0 if freq_level_id == "REF" else 0.5)

    def fake_load_node_profile(calibration_dir):
        return SimpleNamespace(cache_line_size_bytes=64)

    monkeypatch.setattr(postprocess.calibration_module, "load_calibration", fake_load_calibration)
    monkeypatch.setattr(postprocess.node_profile_module, "load_node_profile", fake_load_node_profile)

    kernel_entry = SimpleNamespace(phase_label_hint="compute_bound", binary_checksum="sha256:x",
                                    flops_total_stdout_pattern=None)

    postprocess.run_postprocess(
        run_dir, run_id="r", repetition=1, kernel_ref="npb_ep", kernel_entry=kernel_entry,
        node_id="felix-sc3", freq_level_id="FG_1", calibration_dir=cal_dir,
    )

    assert requested_freq_level_ids == ["FG_1"]


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
