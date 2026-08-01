from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import calibration
from orchestrator.catalog import KernelEntry
from orchestrator.runner import RunResult


def _kernel_entry(**overrides) -> KernelEntry:
    defaults = dict(
        id="k", suite="s", role="calibration", exec_path="/bin/true", binary_checksum="sha256:x",
        phase_label_hint=None, size_variant=None, expected_runtime_seconds=None, warmup_seconds=None,
        success_check={"type": "exit_code"},
    )
    defaults.update(overrides)
    return KernelEntry(**defaults)


def _manifest(tmp_path: Path, *, calibration_refs=("stream", "ert"), datasheet=None) -> SimpleNamespace:
    return SimpleNamespace(
        campaign_id="camp01",
        output_dir=tmp_path,
        calibration=calibration_refs,
        cores=SimpleNamespace(delegated_cpus=(2, 3, 4, 5)),
        hardware_datasheet=datasheet,
    )


def _fake_run_result(run_dir: Path, *, success=True, elapsed_seconds=1.0) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunResult(
        run_id=run_dir.name, kernel_ref="k", freq_level_id="F0_calibration", repetition_index=0,
        command=(), exit_code=0 if success else 1, timed_out=False, success=success,
        elapsed_seconds=elapsed_seconds, run_dir=run_dir, stdout_path=run_dir / "stdout.txt",
        stderr_path=run_dir / "stderr.txt", metadata={},
    )


def test_cal01_cal02_cal03_run_calibration_extrae_del_stdout_no_de_pmu(tmp_path):
    stream_entry = _kernel_entry(
        id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"Triad:\s+([0-9.]+)"
    )
    ert_entry = _kernel_entry(
        id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"GFLOPs/sec:\s+([0-9.]+)"
    )
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.0e10, "p_pico_flops_per_s": 5.0e10})
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / kernel_ref
        run_dir.mkdir(exist_ok=True)
        # El patron del catalogo es responsable de capturar el numero ya en
        # las unidades declaradas por bw_pico_bytes_per_s/p_pico_flops_per_s;
        # run_calibration no hace conversion de unidades por su cuenta.
        if entry is stream_entry:
            (run_dir / "stdout.txt").write_text("Best Rate ...\nTriad:    9536743200.0   0.02  0.02  0.02\n")
        else:
            (run_dir / "stdout.txt").write_text("Empirical Roofline\nGFLOPs/sec:    46600000000.0\n")
        return _fake_run_result(run_dir)

    result = calibration.run_calibration(manifest, catalog, run_single=fake_run_single)

    assert result.bw_pico_bytes_per_s == pytest.approx(9536743200.0)
    assert result.p_pico_flops_per_s == pytest.approx(46600000000.0)
    assert result.i_ridge_flops_per_byte == pytest.approx(result.p_pico_flops_per_s / result.bw_pico_bytes_per_s)


def test_arc42_multiplicador_de_unidad_convierte_antes_de_guardar(tmp_path):
    # STREAM imprime MB/s, no B/s; ert_probe imprime GFLOP/s, no FLOP/s.
    # Sin el multiplicador, bw_pico/p_pico quedarian en la unidad nativa de
    # cada suite y el ridge point saldria sesgado por el cociente entre
    # prefijos (GFLOP/s sobre MB/s = 1000x menor que el flops/byte real).
    stream_entry = _kernel_entry(
        id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"Triad:\s+([0-9.]+)",
        bandwidth_stdout_unit_multiplier=1_000_000,
    )
    ert_entry = _kernel_entry(
        id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"GFLOPs/sec:\s+([0-9.]+)",
        flops_stdout_unit_multiplier=1_000_000_000,
    )
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.47e10, "p_pico_flops_per_s": 2.4e10})
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / kernel_ref
        run_dir.mkdir(exist_ok=True)
        if entry is stream_entry:
            (run_dir / "stdout.txt").write_text("Triad:    14718.6   0.10  0.10  0.10\n")
        else:
            (run_dir / "stdout.txt").write_text("GFLOPs/sec:    23.966\n")
        return _fake_run_result(run_dir)

    result = calibration.run_calibration(manifest, catalog, run_single=fake_run_single)

    assert result.bw_pico_bytes_per_s == pytest.approx(14718.6 * 1_000_000)
    assert result.p_pico_flops_per_s == pytest.approx(23.966 * 1_000_000_000)


def test_cal04_d03_falla_si_esta_fuera_de_rango_y_bloquea(tmp_path):
    stream_entry = _kernel_entry(id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert_entry = _kernel_entry(id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    # Datasheet dice 10x mas de lo que "se midio" -> fuera de +-40%.
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.0e12, "p_pico_flops_per_s": 1.0e12})
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / kernel_ref
        run_dir.mkdir(exist_ok=True)
        value = "1.0e10" if entry is stream_entry else "1.0e10"
        (run_dir / "stdout.txt").write_text(f"BW={value}\nFLOPS={value}\n")
        return _fake_run_result(run_dir)

    with pytest.raises(calibration.CalibrationError, match="D03"):
        calibration.run_calibration(manifest, catalog, run_single=fake_run_single)

    # CAL-05: el artefacto se persiste igual, para poder investigar el fallo.
    path = tmp_path / "roofline_calibration.json"
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["plausibility_check_passed"] is False


def test_cal04_sin_datasheet_declarado_nunca_aprueba_por_falta_de_datos(tmp_path):
    stream_entry = _kernel_entry(id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert_entry = _kernel_entry(id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    manifest = _manifest(tmp_path, datasheet=None)
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / kernel_ref
        run_dir.mkdir(exist_ok=True)
        (run_dir / "stdout.txt").write_text("BW=1.0e10\nFLOPS=1.0e10\n")
        return _fake_run_result(run_dir)

    with pytest.raises(calibration.CalibrationError, match="D03"):
        calibration.run_calibration(manifest, catalog, run_single=fake_run_single)


def test_cal06_load_calibration_rechaza_si_plausibility_check_failed(tmp_path):
    calibration_obj = calibration.RooflineCalibration(
        campaign_id="c", timestamp="t", delegated_cpus="0-3", bw_pico_bytes_per_s=1.0,
        p_pico_flops_per_s=1.0, i_ridge_flops_per_byte=1.0, stream_raw_output="", ert_raw_output="",
        plausibility_check_passed=False, plausibility_message="D03: fuera de rango",
    )
    calibration.write_calibration(calibration_obj, tmp_path)

    with pytest.raises(calibration.CalibrationError, match="CAL-06"):
        calibration.load_calibration(tmp_path)


def test_cal06_load_calibration_acepta_si_paso(tmp_path):
    calibration_obj = calibration.RooflineCalibration(
        campaign_id="c", timestamp="t", delegated_cpus="0-3", bw_pico_bytes_per_s=1.0,
        p_pico_flops_per_s=2.0, i_ridge_flops_per_byte=2.0, stream_raw_output="", ert_raw_output="",
        plausibility_check_passed=True,
    )
    calibration.write_calibration(calibration_obj, tmp_path)
    assert calibration.load_calibration(tmp_path) == calibration_obj


def _write_samples_csv(run_dir: Path, *, instructions: int, cycles: int, cache_references: int, cache_misses: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    header = "run_id,repetition,kernel,label,timestamp_ns,tag,instructions,cycles,cache_references,cache_misses,time_enabled_ns,time_running_ns\n"
    row = f"r,1,k,l,1000,CPU,{instructions},{cycles},{cache_references},{cache_misses},1000,1000\n"
    (run_dir / "samples.csv").write_text(header + row)


def test_cal09_build_calibration_references_requiere_al_menos_5(tmp_path):
    runs = [_fake_run_result(tmp_path / f"r{i}") for i in range(3)]
    with pytest.raises(calibration.CalibrationError, match="CAL-09"):
        calibration.build_calibration_references(runs, "node1")


def test_cal09_cal10_p95_y_cv_estables_aceptado(tmp_path):
    runs = []
    for i in range(5):
        run_dir = tmp_path / f"ref{i}"
        _write_samples_csv(run_dir, instructions=2_000_000_000, cycles=1_000_000_000,
                            cache_references=10_000_000, cache_misses=100_000)
        runs.append(_fake_run_result(run_dir, elapsed_seconds=1.0))

    refs = calibration.build_calibration_references(runs, "felix-sc3")

    assert refs.repetitions == 5
    assert refs.ipc_p95 == pytest.approx(2.0)
    assert refs.accepted is True
    assert refs.cv_pct == pytest.approx(0.0)


def test_cal10_inestabilidad_marca_accepted_false(tmp_path):
    runs = []
    instructions_values = [2_000_000_000, 2_000_000_000, 2_000_000_000, 2_000_000_000, 4_000_000_000]
    for i, instructions in enumerate(instructions_values):
        run_dir = tmp_path / f"ref{i}"
        _write_samples_csv(run_dir, instructions=instructions, cycles=1_000_000_000,
                            cache_references=10_000_000, cache_misses=100_000)
        runs.append(_fake_run_result(run_dir, elapsed_seconds=1.0))

    refs = calibration.build_calibration_references(runs, "felix-sc3", cv_threshold_pct=5.0)

    assert refs.accepted is False
    assert refs.cv_pct > 5.0


def test_cal11_run_calibration_references_persiste_json_aunque_no_acepte(tmp_path):
    entry = _kernel_entry(id="npb_ep", role="dataset", phase_label_hint="compute_bound",
                           size_variant="S", expected_runtime_seconds=1, warmup_seconds=0.0,
                           estimated_memory_bytes=1)
    manifest = _manifest(tmp_path)

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"rep{repetition}"
        instructions = 2_000_000_000 if repetition != 5 else 9_000_000_000
        _write_samples_csv(run_dir, instructions=instructions, cycles=1_000_000_000,
                            cache_references=10_000_000, cache_misses=100_000)
        return _fake_run_result(run_dir, elapsed_seconds=1.0)

    refs = calibration.run_calibration_references(
        entry, manifest, "npb_ep", node_id="felix-sc3", run_single=fake_run_single
    )

    assert (tmp_path / "calibration_references.json").exists()
    assert refs.accepted is False
    loaded = calibration.load_calibration_references(tmp_path)
    assert loaded == refs
