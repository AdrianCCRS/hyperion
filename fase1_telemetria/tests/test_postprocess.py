import csv
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import postprocess


SAMPLES_HEADER = [
    "run_id", "repetition", "kernel", "label", "timestamp_ns", "tag",
    "instructions", "cycles", "cache_references", "cache_misses",
    "stalled_cycles_mem_any", "l2_lines_in_all",
    "fp_scalar_double", "fp_128b_packed_double", "fp_256b_packed_double", "fp_512b_packed_double",
    "time_enabled_ns", "time_running_ns",
    "pkg_uj", "dram_uj", "pkg_delta_uj", "dram_delta_uj", "energy_delta_valid",
    "gpu_power_mw", "gpu_util_pct", "gpu_mem_util_pct", "gpu_sm_clock_mhz", "gpu_energy_mj", "gpu_temperature_c",
    "uncore_cas_count_read_interval", "uncore_cas_count_write_interval",
    "scaling_cur_freq_khz", "scaling_cur_freq_khz_all",
]


def _cpu_row(*, repetition, ts, instructions, cycles, cache_references, cache_misses,
             time_enabled, time_running, stalled_cycles_mem_any=0, l2_lines_in_all=0,
             # ARC-100: 0 by default, same convention as stalled_cycles_mem_any/
             # l2_lines_in_all above -- "supported, zero delta" is the sane
             # default now that FLOPs measurement is the only source (no more
             # prorated fallback to silently exercise instead). Tests that
             # want to exercise the "node/PMU never opened this" path pass
             # fp_scalar_double="" (empty) explicitly, same pattern as the
             # dedicated l2_lines_in_all/stalled_cycles_mem_any "no soportado"
             # tests below.
             fp_scalar_double=0, fp_128b_packed_double=0,
             fp_256b_packed_double=0, fp_512b_packed_double=0,
             # ARC-135: "" (not sampled) by default, same convention as the
             # other optional counters above -- tests exercising the real
             # per-window override pass an explicit value.
             scaling_cur_freq_khz="",
             # ARC-142: "" (not sampled) by default, same convention as
             # scaling_cur_freq_khz above -- tests exercising the multi-CPU
             # spread pass an explicit ';'-separated value.
             scaling_cur_freq_khz_all=""):
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "CPU",
        "instructions": instructions, "cycles": cycles,
        "cache_references": cache_references, "cache_misses": cache_misses,
        "stalled_cycles_mem_any": stalled_cycles_mem_any,
        "l2_lines_in_all": l2_lines_in_all,
        "fp_scalar_double": fp_scalar_double,
        "fp_128b_packed_double": fp_128b_packed_double,
        "fp_256b_packed_double": fp_256b_packed_double,
        "fp_512b_packed_double": fp_512b_packed_double,
        "time_enabled_ns": time_enabled, "time_running_ns": time_running,
        "scaling_cur_freq_khz": scaling_cur_freq_khz,
        "scaling_cur_freq_khz_all": scaling_cur_freq_khz_all,
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


def _uncore_row(*, repetition, ts, cas_count_read_interval, cas_count_write_interval):
    # ARC-119: these are ALREADY per-interval deltas (perf stat -I
    # semantics), not cumulative counters -- never differenced against a
    # previous UNCORE row.
    return {
        "run_id": "r", "repetition": repetition, "kernel": "k", "label": "k",
        "timestamp_ns": ts, "tag": "UNCORE",
        "uncore_cas_count_read_interval": cas_count_read_interval,
        "uncore_cas_count_write_interval": cas_count_write_interval,
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
        i_ridge_flops_per_byte=1.0, llc_line_size_bytes=64,
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


def test_arc135_freq_khz_observed_real_por_ventana_sobreescribe_el_de_contexto(tmp_path):
    # ARC-135: scaling_cur_freq_khz llega ahora por ventana desde samples.csv
    # (muestreado por el colector C++ en el mismo tick que los contadores de
    # PMU) -- reemplaza al viejo freq_khz_observed de WindowContext, una
    # unica lectura de Python tomada DESPUES de que el proceso ya termino,
    # que en datos de campana real no se correlacionaba con el nivel
    # solicitado en absoluto.
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 scaling_cur_freq_khz=2200000),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz=2199500),
    ])
    # WindowContext trae un valor de contexto deliberadamente distinto
    # (2261000, el default de _context()) para confirmar que el real gana.
    windows = postprocess.build_windows(samples, _context())

    assert windows[1]["freq_khz_observed"] == 2199500


def test_arc135_freq_khz_observed_sin_columna_real_usa_el_de_contexto(tmp_path):
    # Compatibilidad hacia atras: samples.csv de antes de ARC-135 (columna
    # ausente/vacia) sigue usando el valor de contexto, nunca se inventa uno.
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    windows = postprocess.build_windows(samples, _context(freq_khz_observed=2261000))

    assert windows[1]["freq_khz_observed"] == 2261000


@pytest.mark.parametrize("raw,expected", [
    (None, None),
    ("", None),
    ("2200000", None),  # una sola lectura: no hay con que comparar
    ("2200000;2200000;2200000", 0),
    ("2200000;3600000;2200000", 1400000),
    # ARC-142: 0 en una posicion = esa lectura individual fallo -- se excluye
    # del spread, no se cuenta como "0 kHz real".
    ("2200000;0;2200000", 0),
    ("0;0", None),
])
def test_arc142_observed_freq_spread(raw, expected):
    assert postprocess._observed_freq_spread(raw) == expected


def test_arc142_freq_khz_observed_spread_por_ventana(tmp_path):
    """ARC-142: scaling_cur_freq_khz_all trae una lectura por CPU delegado
    (no solo CPU0) -- pacca tiene dominio cpufreq por-core, asi que los
    demas nucleos pueden divergir del representativo bajo Turbo/HWP sin que
    freq_khz_observed (el escalar) lo revele."""
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 scaling_cur_freq_khz=2200000, scaling_cur_freq_khz_all="2200000;2200000;2200000"),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz=2200000, scaling_cur_freq_khz_all="2200000;3600000;2200000"),
    ])
    windows = postprocess.build_windows(samples, _context())

    assert windows[1]["freq_khz_observed_spread"] == 1400000
    assert windows[1]["freq_khz_observed"] == 2200000  # el escalar no cambia


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
    windows = postprocess.build_windows(samples, _context())

    window = windows[1]
    assert window["delta_t_ns"] == 1_500_000  # no el --interval-ns nominal
    assert window["delta_instructions"] == 2_000_000
    assert window["ipc"] == pytest.approx(2.0)
    assert window["llc_miss_rate"] == pytest.approx(0.01)
    assert window["mpki"] == pytest.approx(0.5)
    assert window["ips"] == pytest.approx(2_000_000 / (1_500_000 / 1e9))
    assert window["running_ratio"] == pytest.approx(1.0)
    # ARC-123: sin cobertura real de uncore, la ventana queda
    # intensity_undefined -- ok solo se alcanza con datos reales de uncore
    # (ver test_arc122_*), nunca con el proxy de cache_misses.
    assert window["quality_status"] == "intensity_undefined"


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


def test_stalled_cycles_mem_any_delta_y_ratio_se_calculan(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 stalled_cycles_mem_any=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 stalled_cycles_mem_any=400_000),
    ])
    windows = postprocess.build_windows(samples, _context())

    window = windows[1]
    assert window["delta_stalled_cycles_mem_any"] == 400_000
    assert window["stall_mem_ratio"] == pytest.approx(0.4)
    # ARC-123: sin uncore, intensity_undefined -- no relacionado con stall_mem_ratio.
    assert window["quality_status"] == "intensity_undefined"


def test_l2_lines_in_all_delta_y_bytes_moved_l2_proxy_se_calculan(tmp_path):
    # ARC-63: mismo patron que stalled_cycles_mem_any -- delta crudo y una
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
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=64))

    window = windows[1]
    assert window["delta_l2_lines_in_all"] == 2_000
    assert window["bytes_moved_l2_proxy"] == 2_000 * 64
    # ARC-123: sin uncore, intensity_undefined -- no relacionado con bytes_moved_l2_proxy.
    assert window["quality_status"] == "intensity_undefined"


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
    windows = postprocess.build_windows(samples, _context())

    window = windows[1]
    assert window["delta_l2_lines_in_all"] is None
    assert window["bytes_moved_l2_proxy"] is None
    # ARC-123: sin uncore, intensity_undefined -- no relacionado con l2_lines_in_all.
    assert window["quality_status"] == "intensity_undefined"
    assert window["bytes_moved_window"] == 1_000 * 64  # sigue basado en cache_misses, sin cambios (solo reportado)


def test_arc122_uncore_se_agrega_por_intervalo_y_reemplaza_al_proxy(tmp_path):
    # ARC-119: perf stat -I ya reporta un delta por intervalo (no un
    # acumulado -- nunca se resta contra otra fila UNCORE). Un intervalo de
    # perf mas ancho que una ventana de CPU se agrega sumando el
    # flops_measured_window de TODAS las ventanas que cubre, y esa
    # intensidad se difunde igual a cada una -- nunca se asigna el byte
    # count completo del intervalo a una sola ventana angosta.
    # ARC-122/123: con uncore ya verificado funcionando, operational_intensity/
    # phase_label_train (las columnas que entrenan) SOLO usan el valor real
    # de uncore -- bytes_moved_window (el proxy) se sigue reportando
    # intacto, pero nunca alimenta la clasificacion (ni siquiera de respaldo).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 fp_scalar_double=0, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
        # window1: [1_000_000_000, 1_001_000_000], 150 flops
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 fp_scalar_double=10, fp_128b_packed_double=10,
                 fp_256b_packed_double=10, fp_512b_packed_double=10),
        # window2: [1_001_000_000, 1_002_000_000], another 150 flops
        _cpu_row(repetition=1, ts=1_002_000_000, instructions=8_000_000, cycles=4_000_000,
                 cache_references=400_000, cache_misses=2_000, time_enabled=2_000_000, time_running=2_000_000,
                 fp_scalar_double=20, fp_128b_packed_double=20,
                 fp_256b_packed_double=20, fp_512b_packed_double=20),
        # One perf interval spanning [run_start=1_000_000_000, 1_010_000_000]
        # (~10ms, perf's own floor) -- covers BOTH CPU windows above.
        _uncore_row(repetition=1, ts=1_010_000_000, cas_count_read_interval=1_000, cas_count_write_interval=600),
    ])
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=128, i_ridge_flops_per_byte=1.0))

    window1, window2 = windows[1], windows[2]
    expected_bytes = 1_600 * 64
    expected_intensity_uncore = (150.0 + 150.0) / expected_bytes

    for window in (window1, window2):
        assert window["uncore_cas_count_read_interval"] == 1_000
        assert window["uncore_cas_count_write_interval"] == 600
        assert window["bytes_moved_uncore_real"] == expected_bytes
        assert window["operational_intensity_uncore_real"] == pytest.approx(expected_intensity_uncore)
        assert window["phase_label_uncore_real"] == "memory_bound"
        # ARC-122: the training-facing columns now equal the real uncore
        # value, not the proxy-derived one computed earlier in the loop.
        assert window["operational_intensity"] == pytest.approx(expected_intensity_uncore)
        assert window["phase_label_train"] == "memory_bound"
        assert window["quality_status"] == "ok"

    # bytes_moved_window (the proxy) is still reported, untouched, for
    # comparison -- it just no longer drives the classification.
    assert window1["bytes_moved_window"] == 1_000 * 128
    assert window2["bytes_moved_window"] == (2_000 - 1_000) * 128
    proxy_intensity_window1 = 150.0 / (1_000 * 128)
    assert window1["operational_intensity"] != pytest.approx(proxy_intensity_window1)


def test_arc123_ventana_sin_cobertura_de_uncore_queda_indefinida(tmp_path):
    # ARC-123: una ventana que ningun intervalo de perf llega a cubrir (la
    # corrida termina antes de que el primer intervalo cierre) queda
    # intensity_undefined -- ya NO cae al proxy de cache_misses (decision
    # revertida de ARC-122: mezclar una medicion real con una sesgada por
    # no ver prefetch, aunque sea solo en algunas filas, ensuciaba la
    # calidad de la columna de clasificacion de forma inconsistente).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 fp_scalar_double=0, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 fp_scalar_double=10, fp_128b_packed_double=10,
                 fp_256b_packed_double=10, fp_512b_packed_double=10),
        # The perf interval closes BEFORE this window's own t_end -- the
        # window's t_end falls outside (interval_start, interval_end], so
        # no interval covers it.
        _uncore_row(repetition=1, ts=1_000_500_000, cas_count_read_interval=1_000, cas_count_write_interval=600),
    ])
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=128))

    window1 = windows[1]
    assert window1["bytes_moved_uncore_real"] is None
    assert math.isnan(window1["operational_intensity"])
    assert window1["phase_label_train"] is None
    assert window1["quality_status"] == "intensity_undefined"
    # bytes_moved_window (el proxy) se sigue reportando, solo no clasifica.
    assert window1["bytes_moved_window"] == 1_000 * 128


def test_arc123_sin_uncore_en_absoluto_toda_ventana_queda_indefinida(tmp_path):
    # ARC-123: sin ninguna fila UNCORE (nodo/corrida sin ese permiso,
    # manifest.uncore.enabled=False, o un dataset generado antes de que
    # esto existiera): TODA ventana queda intensity_undefined -- ya no hay
    # ningun camino en el que el proxy de cache_misses alimente
    # operational_intensity/phase_label_train.
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 fp_scalar_double=0, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 fp_scalar_double=10, fp_128b_packed_double=10,
                 fp_256b_packed_double=10, fp_512b_packed_double=10),
    ])
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=128))

    window1 = windows[1]
    assert window1["uncore_cas_count_read_interval"] is None
    assert window1["bytes_moved_uncore_real"] is None
    assert window1["operational_intensity_uncore_real"] is None
    assert window1["phase_label_uncore_real"] is None
    assert math.isnan(window1["operational_intensity"])
    assert window1["phase_label_train"] is None
    assert window1["quality_status"] == "intensity_undefined"
    # bytes_moved_window (el proxy) se sigue reportando, solo no clasifica.
    assert window1["bytes_moved_window"] == 1_000 * 128


def test_arc119_cas_count_negativo_no_produce_bytes_moved_uncore_real(tmp_path):
    # Una lectura corrupta (no debería pasar -- perf reporta cuentas sin
    # signo -- pero tratada con la misma disciplina que cualquier otro
    # contador crudo de este archivo: nunca alimenta una cifra negativa a
    # una columna de bytes).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0,
                 fp_scalar_double=0, fp_128b_packed_double=0,
                 fp_256b_packed_double=0, fp_512b_packed_double=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 fp_scalar_double=10, fp_128b_packed_double=10,
                 fp_256b_packed_double=10, fp_512b_packed_double=10),
        _uncore_row(repetition=1, ts=1_001_500_000, cas_count_read_interval=-5, cas_count_write_interval=100),
    ])
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=128))

    window1 = windows[1]
    assert window1["uncore_cas_count_read_interval"] == -5
    assert window1["bytes_moved_uncore_real"] is None
    assert window1["operational_intensity_uncore_real"] is None
    assert window1["phase_label_uncore_real"] is None
    # ARC-123: sin una lectura real valida, la ventana queda indefinida --
    # nunca cae al proxy, ni siquiera cuando bytes_moved_window si es valido.
    assert math.isnan(window1["operational_intensity"])
    assert window1["quality_status"] == "intensity_undefined"


def test_stalled_cycles_mem_any_no_soportado_en_el_nodo_no_marca_pmu_degraded(tmp_path):
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
    old_header = [c for c in SAMPLES_HEADER if c != "stalled_cycles_mem_any"]
    with samples.open("w", newline="", encoding="utf-8") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=old_header, restval="")
        writer.writeheader()
        for row in [
            _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                     cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
            _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                     cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        ]:
            row.pop("stalled_cycles_mem_any", None)
            writer.writerow(row)
    windows = postprocess.build_windows(samples, _context())

    window = windows[1]
    assert window["delta_stalled_cycles_mem_any"] is None
    assert window["stall_mem_ratio"] is None
    # ARC-123: sin uncore, intensity_undefined -- no relacionado con STALLS_MEM_ANY.
    assert window["quality_status"] == "intensity_undefined"
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
    windows = postprocess.build_windows(samples, _context(rapl_enabled=True))

    window = windows[1]
    assert window["energy_valid"] is False
    assert window["pkg_delta_uj"] is None  # nunca el 500000 "crudo" invalido
    assert window["power_w"] is None
    # ARC-123: intensity_undefined (sin uncore) tiene prioridad sobre
    # energy_invalid en _QUALITY_PRIORITY -- la invalidez de energia sigue
    # reflejada en energy_valid/pkg_delta_uj/power_w arriba.
    assert window["quality_status"] == "intensity_undefined"


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
    windows = postprocess.build_windows(samples, _context(rapl_enabled=True))

    window = windows[1]
    assert window["energy_valid"] is True
    assert window["pkg_delta_uj"] == 2_000_000
    assert window["power_w"] == pytest.approx(2.0 / 0.001)
    # ARC-123: sin uncore, intensity_undefined -- no relacionado con energia.
    assert window["quality_status"] == "intensity_undefined"


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
    windows = postprocess.build_windows(samples, _context(rapl_enabled=True))

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
    windows = postprocess.build_windows(samples, _context(warmup_seconds=1.0))

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
    windows = postprocess.build_windows(samples, _context())

    window = windows[1]
    assert window["bytes_moved_window"] == 0
    assert math.isnan(window["operational_intensity"])
    assert window["quality_status"] == "intensity_undefined"
    assert window["phase_label_train"] is None


def test_post10_bytes_moved_usa_line_size_del_node_profile(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                 cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=128))

    window1 = windows[1]
    assert window1["bytes_moved_window"] == 1_000 * 128  # linea real del node_profile, no 64 hardcodeado


def test_arc100_flops_medidos_alimentan_operational_intensity(tmp_path):
    # ARC-100: FLOPs por ventana ya no se estiman por prorrateo -- se miden
    # directamente por hardware (ARC-97/98/99). Este test verifica el
    # camino completo: los 4 contadores crudos ponderados 1/2/4/8 ->
    # flops_measured_window -> operational_intensity = flops_measured_window
    # / bytes reales de uncore (ARC-123 -- ya no del proxy de cache_misses,
    # cubierto aqui con un intervalo real para que la ventana no quede
    # indefinida).
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
        _uncore_row(repetition=1, ts=1_001_000_000, cas_count_read_interval=500, cas_count_write_interval=300),
    ])
    windows = postprocess.build_windows(samples, _context(llc_line_size_bytes=128))

    window1 = windows[1]
    assert window1["flops_measured_window"] == 150.0
    assert window1["bytes_moved_window"] == 1_000 * 128  # el proxy se sigue reportando, sin usarse
    assert window1["operational_intensity"] == pytest.approx(150.0 / (800 * 64))
    assert window1["quality_status"] == "ok"


def test_arc100_sin_fp_arith_operational_intensity_queda_indefinida(tmp_path):
    # Sin fallback: un nodo/corrida donde FP_ARITH_INST_RETIRED nunca se abrio
    # (columnas fp_* ausentes por completo, no solo vacias) debe dejar
    # operational_intensity/phase_label_train indefinidos, nunca recurrir a
    # una estimacion silenciosa.
    samples = tmp_path / "samples.csv"
    old_header = [c for c in SAMPLES_HEADER if c not in (
        "fp_scalar_double", "fp_128b_packed_double", "fp_256b_packed_double", "fp_512b_packed_double",
    )]
    with samples.open("w", newline="", encoding="utf-8") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=old_header, restval="")
        writer.writeheader()
        for row in [
            _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                     cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
            _cpu_row(repetition=1, ts=1_001_000_000, instructions=4_000_000, cycles=2_000_000,
                     cache_references=200_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
        ]:
            for column in ("fp_scalar_double", "fp_128b_packed_double", "fp_256b_packed_double", "fp_512b_packed_double"):
                row.pop(column, None)
            writer.writerow(row)
    windows = postprocess.build_windows(samples, _context())

    window1 = windows[1]
    assert window1["flops_measured_window"] is None
    assert math.isnan(window1["operational_intensity"])
    assert window1["phase_label_train"] is None
    assert window1["quality_status"] == "intensity_undefined"


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
    windows = postprocess.build_windows(samples, _context())

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
        # ARC-123: cobertura real de uncore para que la ventana no quede
        # indefinida -- flops_measured_window=0 (fp_* por defecto), asi que
        # cualquier byte count positivo real da I=0 -> memory_bound.
        _uncore_row(repetition=1, ts=1_001_000_000, cas_count_read_interval=1_000, cas_count_write_interval=1_000),
    ])
    # phase_label_hint dice compute_bound, pero I = flops/bytes sera bajo -> memory_bound.
    windows = postprocess.build_windows(
        samples, _context(phase_label_hint="compute_bound", i_ridge_flops_per_byte=1.0)
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


def test_arc174_filas_gpu_no_reciben_clasificacion_de_frecuencia(tmp_path):
    # Alcance limitado a CPU (aprobado explícitamente) -- GPU tiene otra
    # cadencia/señal (gpu_sm_clock_mhz) y no debe tocarse por este cambio.
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz_all="2200000;2200000"),
        _gpu_row(repetition=1, ts=1_000_500_000, gpu_power_mw=36324, gpu_util_pct=0),
    ])
    windows = postprocess.build_windows(
        samples,
        _context(freq_tolerance_fraction=0.03, freq_khz_applied=2_200_000, freq_expected_cpu_count=2),
    )
    gpu_rows = [w for w in windows if w["quality_status"] == "gpu_telemetry"]
    assert len(gpu_rows) == 1
    assert gpu_rows[0]["frequency_quality_status"] is None
    assert gpu_rows[0]["frequency_outlier_cpu_count"] is None


def test_arc174_ventana_cpu_valida_bajo_nivel_fijo(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz_all="2200000;2199000"),
    ])
    windows = postprocess.build_windows(
        samples,
        _context(freq_tolerance_fraction=0.03, freq_khz_applied=2_200_000, freq_is_native_governor=False),
    )
    assert windows[1]["frequency_quality_status"] == "valid"
    assert windows[1]["frequency_outlier_cpu_count"] == 0


def test_arc174_ventana_cpu_no_confiable_bajo_nivel_fijo(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz_all="2200000;1600000"),
    ])
    windows = postprocess.build_windows(
        samples,
        _context(freq_tolerance_fraction=0.03, freq_khz_applied=2_200_000, freq_is_native_governor=False),
    )
    assert windows[1]["frequency_quality_status"] == "observation_unreliable"
    assert windows[1]["frequency_outlier_cpu_count"] == 1


def test_arc174_ventana_dentro_de_grace_seconds_queda_unverified(tmp_path):
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=0, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz_all="1600000;1600000"),
    ])
    windows = postprocess.build_windows(
        samples,
        _context(
            freq_tolerance_fraction=0.03, freq_khz_applied=2_200_000, freq_is_native_governor=False,
            freq_grace_seconds=10.0,
        ),
    )
    assert windows[1]["frequency_quality_status"] == "observation_unverified_grace"


def test_arc174_ref_no_confundido_con_actuacion_desactivada(tmp_path):
    # REF explícito (freq_is_native_governor=True) -- clasifica
    # "not_applicable_native" pese a no tener expected_khz, a diferencia de
    # un nivel fixed sin actuación (mismos None, pero is_native_governor=False
    # -- cubierto por test_arc174_config_incompleta... en test_validation.py).
    samples = tmp_path / "samples.csv"
    _write_samples(samples, [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000,
                 scaling_cur_freq_khz_all="3600000;3550000"),
    ])
    windows = postprocess.build_windows(samples, _context(freq_is_native_governor=True))
    assert windows[1]["frequency_quality_status"] == "not_applicable_native"


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

    from fase1_telemetria import calibration as calibration_module
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


def test_arc174_run_postprocess_output_dir_no_toca_el_run_dir_original(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_samples(run_dir / "samples.csv", [
        _cpu_row(repetition=1, ts=1_000_000_000, instructions=0, cycles=0,
                 cache_references=0, cache_misses=0, time_enabled=0, time_running=0),
        _cpu_row(repetition=1, ts=1_001_000_000, instructions=2_000_000, cycles=1_000_000,
                 cache_references=100_000, cache_misses=1_000, time_enabled=1_000_000, time_running=1_000_000),
    ])
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()

    monkeypatch.setattr(
        postprocess.calibration_module, "load_calibration",
        lambda calibration_dir, freq_level_id="": SimpleNamespace(i_ridge_flops_per_byte=1.0),
    )
    monkeypatch.setattr(
        postprocess.node_profile_module, "load_node_profile",
        lambda calibration_dir: SimpleNamespace(cache_line_size_bytes=64),
    )
    kernel_entry = SimpleNamespace(phase_label_hint="compute_bound", binary_checksum="sha256:x",
                                    flops_total_stdout_pattern=None)

    derived_dir = tmp_path / "derived" / "run"
    windows_path = postprocess.run_postprocess(
        run_dir, run_id="r", repetition=1, kernel_ref="npb_ep", kernel_entry=kernel_entry,
        node_id="felix-sc3", freq_level_id="REF", calibration_dir=cal_dir,
        output_dir=derived_dir,
    )

    assert windows_path == derived_dir / "windows.csv"
    assert windows_path.exists()
    assert (derived_dir / "frequency_quality_summary.json").exists()
    # El run_dir original nunca recibe windows.csv/el resumen -- solo tenía
    # samples.csv de entrada.
    assert not (run_dir / "windows.csv").exists()
    assert not (run_dir / "frequency_quality_summary.json").exists()


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_fixture_fake_samples_cubre_wrap_running_ratio_bajo_bytes_cero_y_warmup():
    """Ejercita tests/orchestrator/fixtures/fake_samples.csv de punta a punta:
    primera muestra, warmup, wrap/reset de contador, running_ratio bajo, y
    bytes_moved_window == 0 -- los casos que F2.5 pide cubrir. El fixture no
    tiene filas UNCORE, asi que (ARC-123) ninguna ventana llega a "ok" --
    toda ventana sin un problema de mayor prioridad queda intensity_undefined."""
    windows = postprocess.build_windows(
        FIXTURES_DIR / "fake_samples.csv",
        _context(
            run_id="fake_run", kernel_ref="npb_ep", warmup_seconds=0.0005,
            running_ratio_min=0.9,
        ),
    )

    statuses = [row["quality_status"] for row in windows]
    assert statuses[0] == "first_sample_no_delta"
    assert "warmup_excluded" in statuses
    assert "pmu_degraded" in statuses  # cubre tanto el wrap como el running_ratio bajo
    assert "intensity_undefined" in statuses  # cache_misses=0 en la ultima ventana, y sin uncore en el resto

    # POST-02: el wrap se conserva crudo, no se enmascara con un delta positivo falso.
    wrap_window = next(w for w in windows if w["delta_instructions"] == -200_000)
    assert wrap_window["quality_status"] == "pmu_degraded"
