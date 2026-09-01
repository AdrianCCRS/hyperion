from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import calibration
from common.hpc.catalog import KernelEntry
from fase1_telemetria.runner import RunResult


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


def _fake_run_result(
    run_dir: Path, *, success=True, elapsed_seconds=1.0, metadata=None,
) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunResult(
        run_id=run_dir.name, kernel_ref="k", freq_level_id="F0_calibration", repetition_index=0,
        command=(), exit_code=0 if success else 1, timed_out=False, success=success,
        elapsed_seconds=elapsed_seconds, run_dir=run_dir, stdout_path=run_dir / "stdout.txt",
        stderr_path=run_dir / "stderr.txt", metadata=metadata or {},
    )


@pytest.mark.parametrize("frequency_trace", [
    None,
    {"accepted": False, "factor_id": "E01", "message": "fuera de objetivo"},
])
def test_arc167_cal07_calibracion_ya_no_bloquea_solo_advierte(tmp_path, frequency_trace, caplog):
    """ARC-167: CAL-07 en calibración degradó de bloqueante a advertencia --
    decisión explícita del usuario tras evidencia real de que `stream_official`
    produce dispersión dispersa (no un transitorio) que no refleja un P_pico/
    BW_pico erróneo (ambos vienen del stdout, nunca de esta traza)."""
    stream_entry = _kernel_entry(
        id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)",
    )
    ert_entry = _kernel_entry(
        id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)",
    )
    manifest = _manifest(
        tmp_path,
        datasheet={"bw_pico_bytes_per_s": 1.0e10, "p_pico_flops_per_s": 1.0e10},
    )
    manifest.frequency_validation = {"require_per_window": True, "tolerance_fraction": 0.03}
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / kernel_ref
        run_dir.mkdir(exist_ok=True)
        (run_dir / "stdout.txt").write_text("BW=10000000000\nFLOPS=10000000000\n")
        metadata = {} if frequency_trace is None else {"frequency_trace_validation": frequency_trace}
        return _fake_run_result(run_dir, metadata=metadata)

    with caplog.at_level("WARNING"):
        result = calibration.run_calibration(manifest, catalog, run_single=fake_run_single)

    assert result.plausibility_check_passed is True
    assert any("CAL-07" in record.message for record in caplog.records)


def test_arc167_cal07_referencias_ya_no_bloquea_solo_advierte(tmp_path, caplog):
    entry = _kernel_entry(
        id="npb_ep", role="dataset", phase_label_hint="compute_bound",
        size_variant="S", expected_runtime_seconds=1, warmup_seconds=0.0,
        estimated_memory_bytes=1,
    )
    manifest = _manifest(tmp_path)
    manifest.frequency_validation = {"require_per_window": True, "tolerance_fraction": 0.03}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"rep{repetition}"
        _write_samples_csv(
            run_dir, instructions=2_000_000_000, cycles=1_000_000_000,
            cache_references=10_000_000, cache_misses=100_000,
        )
        return _fake_run_result(run_dir, metadata={
            "frequency_trace_validation": {
                "accepted": repetition != 3,
                "factor_id": "E01",
                "message": "fuera de objetivo",
            },
        })

    with caplog.at_level("WARNING"):
        calibration.run_calibration_references(
            entry, manifest, "npb_ep", node_id="pacca-a100", run_single=fake_run_single,
        )

    assert (tmp_path / "calibration_references.json").exists()
    assert any("CAL-07" in record.message for record in caplog.records)


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


def test_arc78_run_calibration_calibra_por_cada_nivel_de_frecuencia(tmp_path):
    # P_pico escala con el reloj de nucleo, BW_pico no -- un i_ridge unico
    # para toda la campana clasificaria mal las ventanas de cualquier nivel
    # distinto al que se calibro (ver Consolidacion_Kernels_Dataset_Fase1.md
    # seccion 0). run_calibration() debe calibrar cada nivel por separado.
    stream_entry = _kernel_entry(id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert_entry = _kernel_entry(id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.0e10, "p_pico_flops_per_s": 1.0e10})
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="FG_1", mode="fixed", fraction=0.5),
    )
    catalog = {"stream": stream_entry, "ert": ert_entry}

    apply_calls = []

    def fake_apply_frequency(cpus, level, env):
        apply_calls.append(level.id)

    run_calls = []

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_calls.append((kernel_ref, freq_level_id))
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        # BW_pico no cambia con el reloj; P_pico si -- FG_1 mide la mitad.
        # (regex de prueba es [0-9.]+, sin soporte de notacion cientifica --
        # se escriben los numeros completos, no "1.0e10").
        flops = "10000000000" if freq_level_id == "REF" else "5000000000"
        if entry is stream_entry:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text(f"FLOPS={flops}\n")
        return _fake_run_result(run_dir)

    env_profile = SimpleNamespace(frequency_write_capable=True)

    result = calibration.run_calibration(
        manifest, catalog, environment_profile=env_profile, run_single=fake_run_single,
        apply_frequency=fake_apply_frequency,
    )

    # Se aplico frecuencia para los 2 niveles (referencia primero), y se
    # corrieron stream/ert una vez cada uno POR NIVEL, no una sola vez para
    # toda la campana.
    assert apply_calls == ["REF", "FG_1"]
    assert run_calls == [("stream", "REF"), ("ert", "REF"), ("stream", "FG_1"), ("ert", "FG_1")]

    # El valor de retorno sigue siendo el de la referencia (compatibilidad
    # con validate_campaign_calibration/CampaignResult, que esperan un solo
    # RooflineCalibration).
    assert result.freq_level_id == "REF"
    assert result.p_pico_flops_per_s == pytest.approx(1.0e10)

    # Cada nivel quedo persistido en su propio archivo, con su propio ridge
    # -- postprocess.py debe poder pedir el de FG_1 sin tocar el de REF.
    ref_loaded = calibration.load_calibration(tmp_path, "REF")
    fg1_loaded = calibration.load_calibration(tmp_path, "FG_1")
    assert ref_loaded.i_ridge_flops_per_byte == pytest.approx(1.0)
    assert fg1_loaded.i_ridge_flops_per_byte == pytest.approx(0.5)
    assert fg1_loaded.p_pico_flops_per_s < ref_loaded.p_pico_flops_per_s


def test_arc161_run_calibration_invoca_settle_if_configured_por_nivel(tmp_path, monkeypatch):
    # ARC-161: sin verificar que la frecuencia se asento antes de medir
    # P_pico, ert_probe puede medir bajo el techo del nivel anterior en vez
    # del pedido (confirmado en paccaA100, EPP=performance bajo HWP decae
    # lento hacia frecuencias mas bajas). run_calibration() debe invocar
    # freqctl.settle_if_configured() una vez POR NIVEL, con la
    # AppliedFrequency real de ese nivel.
    stream_entry = _kernel_entry(id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert_entry = _kernel_entry(id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.0e10, "p_pico_flops_per_s": 1.0e10})
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="FG_1", mode="fixed", fraction=0.5),
    )
    manifest.frequency_settle = {"enabled": True, "timeout_seconds": 15.0, "tolerance_fraction": 0.05}
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_apply_frequency(cpus, level, env):
        return SimpleNamespace(
            level_id=level.id, strategy="bounded_range", requested_khz=(None if level.id == "REF" else 1600000),
            applied_khz=None, per_cpu_applied_khz={}, governor_applied=None, write_skipped_reason=None,
        )

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        flops = "10000000000" if freq_level_id == "REF" else "5000000000"
        if entry is stream_entry:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text(f"FLOPS={flops}\n")
        return _fake_run_result(run_dir)

    settle_calls = []
    monkeypatch.setattr(
        calibration.freqctl, "settle_if_configured",
        lambda cpus, applied, env, *, settle_config: settle_calls.append((applied.level_id, settle_config)),
    )

    calibration.run_calibration(
        manifest, catalog, environment_profile=SimpleNamespace(frequency_write_capable=True),
        run_single=fake_run_single, apply_frequency=fake_apply_frequency,
    )

    assert settle_calls == [
        ("REF", manifest.frequency_settle),
        ("FG_1", manifest.frequency_settle),
    ]


def test_arc78_nivel_no_referencia_no_puede_medir_mas_flops_que_la_referencia(tmp_path):
    # Si un nivel de frecuencia mas bajo mide MAS FLOPs/s que la referencia,
    # algo salio mal (la frecuencia pedida no se aplico de verdad, o al
    # reves) -- D03 debe bloquear la campana, no solo el nivel.
    stream_entry = _kernel_entry(id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert_entry = _kernel_entry(id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.0e10, "p_pico_flops_per_s": 1.0e10})
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="FG_1", mode="fixed", fraction=0.5),
    )
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        flops = "10000000000" if freq_level_id == "REF" else "20000000000"
        if entry is stream_entry:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text(f"FLOPS={flops}\n")
        return _fake_run_result(run_dir)

    # ARC-102: FG_1 es "fixed" -- desde el guard RUN-09, run_calibration()
    # exige capacidad real de aplicar frecuencia para medirlo (si no, aborta
    # antes de llegar a D03, que es justo lo que este test quiere ejercitar).
    with pytest.raises(calibration.CalibrationError, match="D03"):
        calibration.run_calibration(
            manifest, catalog, run_single=fake_run_single,
            environment_profile=SimpleNamespace(frequency_write_capable=True),
            apply_frequency=lambda cpus, level, env: None,
        )


def test_arc78_load_calibration_sin_freq_level_id_usa_el_archivo_legado(tmp_path):
    calibration_obj = calibration.RooflineCalibration(
        campaign_id="c", timestamp="t", delegated_cpus="0-3", bw_pico_bytes_per_s=1.0,
        p_pico_flops_per_s=2.0, i_ridge_flops_per_byte=2.0, stream_raw_output="", ert_raw_output="",
        plausibility_check_passed=True, freq_level_id="",
    )
    path = calibration.write_calibration(calibration_obj, tmp_path)

    assert path.name == "roofline_calibration.json"
    assert calibration.load_calibration(tmp_path) == calibration_obj


def _gpu_kernel_entry(**overrides) -> KernelEntry:
    defaults = dict(
        id="k", suite="s", role="calibration", exec_path="/bin/true", binary_checksum="sha256:x",
        phase_label_hint=None, size_variant=None, expected_runtime_seconds=None, warmup_seconds=None,
        success_check={"type": "exit_code"}, device="gpu",
    )
    defaults.update(overrides)
    return KernelEntry(**defaults)


def test_arc80_run_gpu_calibration_sin_manifest_gpu_calibration_no_hace_nada(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.gpu = {}
    assert calibration.run_gpu_calibration(manifest, {}) == {}


def test_arc80_run_gpu_calibration_calibra_fp32_y_fp64_por_nivel(tmp_path):
    stream = _gpu_kernel_entry(id="gpu_stream_bw", reports_bandwidth_stdout=True,
                                bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert32 = _gpu_kernel_entry(id="gpu_ert_probe_fp32", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp32")
    ert64 = _gpu_kernel_entry(id="gpu_ert_probe_fp64", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp64")
    catalog_map = {"gpu_stream_bw": stream, "gpu_ert_probe_fp32": ert32, "gpu_ert_probe_fp64": ert64}

    manifest = _manifest(tmp_path)
    manifest.gpu = {"calibration": ["gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]}
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
    )

    run_calls = []

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_calls.append((kernel_ref, freq_level_id))
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        if entry is stream:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        elif entry is ert32:
            (run_dir / "stdout.txt").write_text("FLOPS=20000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=8000000000\n")
        return _fake_run_result(run_dir)

    result = calibration.run_gpu_calibration(manifest, catalog_map, run_single=fake_run_single)

    assert set(run_calls) == {
        ("gpu_stream_bw", "REF"), ("gpu_ert_probe_fp32", "REF"), ("gpu_ert_probe_fp64", "REF"),
    }
    assert result["fp32"].i_ridge_flops_per_byte == pytest.approx(2.0)
    assert result["fp64"].i_ridge_flops_per_byte == pytest.approx(0.8)
    assert result["fp32"].gpu_precision == "fp32"
    assert result["fp64"].gpu_precision == "fp64"

    # Cada precision/nivel queda en su propio archivo -- load_calibration
    # debe poder pedir cualquiera de los dos sin tocar el otro.
    loaded_fp32 = calibration.load_calibration(tmp_path, "REF", gpu_precision="fp32")
    loaded_fp64 = calibration.load_calibration(tmp_path, "REF", gpu_precision="fp64")
    assert loaded_fp32.i_ridge_flops_per_byte == pytest.approx(2.0)
    assert loaded_fp64.i_ridge_flops_per_byte == pytest.approx(0.8)


def test_arc102_run_calibration_nivel_fixed_sin_permiso_falla_en_vez_de_medir_en_nativo(tmp_path):
    # ARC-102: run_calibration() tiene su propia logica de aplicacion de
    # frecuencia (separada de runner.run_single), que nunca heredaba el
    # guard RUN-09 -- un nivel "fixed" sin capacidad real de escritura debe
    # abortar, no calibrar en silencio a la frecuencia nativa y persistir
    # roofline_calibration_<level.id>.json como si fuera real.
    stream_entry = _kernel_entry(id="stream", reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert_entry = _kernel_entry(id="ert", reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    manifest = _manifest(tmp_path, datasheet={"bw_pico_bytes_per_s": 1.0e10, "p_pico_flops_per_s": 1.0e10})
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="FG_1", mode="fixed", fraction=0.5),
    )
    catalog = {"stream": stream_entry, "ert": ert_entry}

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        # REF (native_governor) no requiere escritura, se mide normalmente
        # -- el guard debe activarse recien al llegar a FG_1 (fixed).
        assert freq_level_id != "FG_1", "no debe medirse FG_1 sin capacidad real de frecuencia"
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        if entry is stream_entry:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=10000000000\n")
        return _fake_run_result(run_dir)

    with pytest.raises(calibration.CalibrationError, match="RUN-09"):
        calibration.run_calibration(
            manifest, catalog, run_single=fake_run_single,
            environment_profile=SimpleNamespace(frequency_write_capable=False),
        )


def test_arc102_run_gpu_calibration_nivel_fixed_sin_permiso_gpu_falla(tmp_path):
    # ARC-102: mismo principio, eje GPU -- sin gpu_frequency_write_capable,
    # un nivel "fixed" debe abortar en vez de calibrar el ridge de GPU al
    # mismo reloj nativo en los 6 niveles (justo el problema que ARC-87
    # documenta que este fijado de reloj existe para evitar).
    stream = _gpu_kernel_entry(id="gpu_stream_bw", reports_bandwidth_stdout=True,
                                bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert32 = _gpu_kernel_entry(id="gpu_ert_probe_fp32", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp32")
    ert64 = _gpu_kernel_entry(id="gpu_ert_probe_fp64", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp64")
    catalog_map = {"gpu_stream_bw": stream, "gpu_ert_probe_fp32": ert32, "gpu_ert_probe_fp64": ert64}

    manifest = _manifest(tmp_path)
    manifest.gpu = {"calibration": ["gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]}
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="FG_1", mode="fixed", fraction=0.5),
    )

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        # REF no requiere escritura de GPU; el guard debe activarse recien
        # al llegar a FG_1 (fixed).
        assert freq_level_id != "FG_1", "no debe medirse FG_1 sin capacidad real de frecuencia de GPU"
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        if entry is stream:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=10000000000\n")
        return _fake_run_result(run_dir)

    with pytest.raises(calibration.CalibrationError, match="RUN-09"):
        calibration.run_gpu_calibration(
            manifest, catalog_map, run_single=fake_run_single,
            environment_profile=SimpleNamespace(gpu_frequency_write_capable=False),
        )


def test_arc80_run_gpu_calibration_no_confunde_gpu_dgemm_calibration_con_el_ridge(tmp_path):
    # gpu_dgemm_calibration tambien reporta FLOPs de GPU (referencia
    # informativa de cuBLAS, ARC-76) pero NO debe usarse como fuente del
    # ridge -- solo lo que manifest.gpu["calibration"] declara explicitamente.
    stream = _gpu_kernel_entry(id="gpu_stream_bw", reports_bandwidth_stdout=True,
                                bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert32 = _gpu_kernel_entry(id="gpu_ert_probe_fp32", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp32")
    ert64 = _gpu_kernel_entry(id="gpu_ert_probe_fp64", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp64")
    dgemm_calibration = _gpu_kernel_entry(id="gpu_dgemm_calibration", role="calibration",
                                           reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)")
    catalog_map = {
        "gpu_stream_bw": stream, "gpu_ert_probe_fp32": ert32, "gpu_ert_probe_fp64": ert64,
        "gpu_dgemm_calibration": dgemm_calibration,
    }

    manifest = _manifest(tmp_path)
    manifest.gpu = {"calibration": ["gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]}
    manifest.frequency_levels = (SimpleNamespace(id="REF", mode="native_governor", fraction=None),)

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / kernel_ref
        run_dir.mkdir(exist_ok=True)
        if entry is stream:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=20000000000\n")
        return _fake_run_result(run_dir)

    # No debe fallar por la colision de gpu_dgemm_calibration -- ni siquiera
    # se corre, porque no esta en manifest.gpu["calibration"].
    calibration.run_gpu_calibration(manifest, catalog_map, run_single=fake_run_single)


def test_arc87_run_gpu_calibration_fija_el_reloj_de_gpu_por_nivel(tmp_path):
    # ARC-87: sin fijar tambien el reloj de GPU en cada nivel, un "ridge
    # point por nivel" no mediria nada distinto entre niveles -- el reloj
    # fisico seguiria siendo el mismo en REF y en FG_1.
    stream = _gpu_kernel_entry(id="gpu_stream_bw", reports_bandwidth_stdout=True,
                                bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert32 = _gpu_kernel_entry(id="gpu_ert_probe_fp32", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp32")
    ert64 = _gpu_kernel_entry(id="gpu_ert_probe_fp64", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp64")
    catalog_map = {"gpu_stream_bw": stream, "gpu_ert_probe_fp32": ert32, "gpu_ert_probe_fp64": ert64}

    manifest = _manifest(tmp_path)
    manifest.gpu = {"calibration": ["gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]}
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="FG_1", mode="fixed", fraction=0.5),
    )

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        if entry is stream:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=20000000000\n")
        return _fake_run_result(run_dir)

    apply_calls = []
    env_profile = SimpleNamespace(gpu_frequency_write_capable=True)

    calibration.run_gpu_calibration(
        manifest, catalog_map, run_single=fake_run_single, environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: apply_calls.append((level.id, env)),
    )

    assert apply_calls == [("REF", env_profile), ("FG_1", env_profile)]


def test_arc129_run_gpu_calibration_usa_gpu_frequency_levels_no_frequency_levels(tmp_path):
    # ARC-129: cuando manifest.gpu_frequency_levels existe, es EL eje que
    # importa para el ridge de GPU -- frequency_levels (CPU) debe ignorarse
    # por completo aquí (ids deliberadamente distintos, para que la prueba
    # no pueda pasar "por coincidencia").
    stream = _gpu_kernel_entry(id="gpu_stream_bw", reports_bandwidth_stdout=True,
                                bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert32 = _gpu_kernel_entry(id="gpu_ert_probe_fp32", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp32")
    ert64 = _gpu_kernel_entry(id="gpu_ert_probe_fp64", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp64")
    catalog_map = {"gpu_stream_bw": stream, "gpu_ert_probe_fp32": ert32, "gpu_ert_probe_fp64": ert64}

    manifest = _manifest(tmp_path)
    manifest.gpu = {"calibration": ["gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]}
    manifest.frequency_levels = (
        SimpleNamespace(id="CPU_REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="CPU_F0", mode="fixed", fraction=0.5),
    )
    manifest.gpu_frequency_levels = (
        SimpleNamespace(id="GREF", mode="native_governor", fraction=None),
        SimpleNamespace(id="GF0", mode="fixed", fraction=0.25),
    )

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        if entry is stream:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=20000000000\n")
        return _fake_run_result(run_dir)

    apply_gpu_calls = []
    apply_cpu_calls = []
    env_profile = SimpleNamespace(gpu_frequency_write_capable=True, frequency_write_capable=True)

    result = calibration.run_gpu_calibration(
        manifest, catalog_map, run_single=fake_run_single, environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: apply_gpu_calls.append(level.id),
        apply_frequency=lambda cpus, level, env: apply_cpu_calls.append(level.id),
    )

    # El bucle recorrió gpu_frequency_levels (GREF/GF0), nunca
    # frequency_levels (CPU_REF/CPU_F0).
    assert apply_gpu_calls == ["GREF", "GF0"]
    # apply_frequency (eje de CPU) nunca se invoca -- incidental cuando el
    # ridge de GPU se calibra por su propio eje independiente.
    assert apply_cpu_calls == []
    # La referencia devuelta (nivel native_governor) queda etiquetada con el
    # id del eje de GPU (GREF), nunca con uno del eje de CPU.
    assert set(result.keys()) == {"fp32", "fp64"}
    assert result["fp32"].freq_level_id == "GREF"


def test_arc87_run_gpu_calibration_no_fija_reloj_si_no_hay_permiso(tmp_path):
    stream = _gpu_kernel_entry(id="gpu_stream_bw", reports_bandwidth_stdout=True,
                                bandwidth_stdout_pattern=r"BW=([0-9.]+)")
    ert32 = _gpu_kernel_entry(id="gpu_ert_probe_fp32", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp32")
    ert64 = _gpu_kernel_entry(id="gpu_ert_probe_fp64", reports_flops_stdout=True,
                               flops_stdout_pattern=r"FLOPS=([0-9.]+)", gpu_precision="fp64")
    catalog_map = {"gpu_stream_bw": stream, "gpu_ert_probe_fp32": ert32, "gpu_ert_probe_fp64": ert64}

    manifest = _manifest(tmp_path)
    manifest.gpu = {"calibration": ["gpu_stream_bw", "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"]}
    manifest.frequency_levels = (SimpleNamespace(id="REF", mode="native_governor", fraction=None),)

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"{kernel_ref}_{freq_level_id}"
        run_dir.mkdir(exist_ok=True)
        if entry is stream:
            (run_dir / "stdout.txt").write_text("BW=10000000000\n")
        else:
            (run_dir / "stdout.txt").write_text("FLOPS=20000000000\n")
        return _fake_run_result(run_dir)

    apply_calls = []
    env_profile = SimpleNamespace(gpu_frequency_write_capable=False)

    calibration.run_gpu_calibration(
        manifest, catalog_map, run_single=fake_run_single, environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: apply_calls.append((level.id, env)),
    )

    assert apply_calls == []


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


def test_cal12_run_calibration_references_re_fija_ref_antes_de_medir(tmp_path):
    """ARC-94: run_calibration()/run_gpu_calibration() dejan el ULTIMO
    nivel fixed (tipicamente F4) todavia aplicado -- sin volver a fijar
    native_governor aqui, las 5 repeticiones de referencia (IPC/MPKI P95)
    quedarian medidas bajo esa frecuencia pinneada, contaminando
    ipc_relative/mpki_relative de toda la campaña."""
    entry = _kernel_entry(id="npb_ep", role="dataset", phase_label_hint="compute_bound",
                           size_variant="S", expected_runtime_seconds=1, warmup_seconds=0.0,
                           estimated_memory_bytes=1)
    manifest = _manifest(tmp_path)
    manifest.frequency_levels = (
        SimpleNamespace(id="REF", mode="native_governor", fraction=None),
        SimpleNamespace(id="F4", mode="fixed", fraction=0.0),
    )

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"rep{repetition}"
        _write_samples_csv(run_dir, instructions=2_000_000_000, cycles=1_000_000_000,
                            cache_references=10_000_000, cache_misses=100_000)
        return _fake_run_result(run_dir, elapsed_seconds=1.0)

    apply_calls = []
    env_profile = SimpleNamespace(frequency_write_capable=True)

    calibration.run_calibration_references(
        entry, manifest, "npb_ep", node_id="pacca-a100", run_single=fake_run_single,
        environment_profile=env_profile,
        apply_frequency=lambda cpus, level, env: apply_calls.append((cpus, level.id, env)),
    )

    assert apply_calls == [((2, 3, 4, 5), "REF", env_profile)]


def test_cal13_run_calibration_references_no_aplica_si_no_hay_permiso(tmp_path):
    entry = _kernel_entry(id="npb_ep", role="dataset", phase_label_hint="compute_bound",
                           size_variant="S", expected_runtime_seconds=1, warmup_seconds=0.0,
                           estimated_memory_bytes=1)
    manifest = _manifest(tmp_path)
    manifest.frequency_levels = (SimpleNamespace(id="REF", mode="native_governor", fraction=None),)

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition, **kwargs):
        run_dir = tmp_path / f"rep{repetition}"
        _write_samples_csv(run_dir, instructions=2_000_000_000, cycles=1_000_000_000,
                            cache_references=10_000_000, cache_misses=100_000)
        return _fake_run_result(run_dir, elapsed_seconds=1.0)

    apply_calls = []
    calibration.run_calibration_references(
        entry, manifest, "npb_ep", node_id="pacca-a100", run_single=fake_run_single,
        environment_profile=SimpleNamespace(frequency_write_capable=False),
        apply_frequency=lambda cpus, level, env: apply_calls.append((cpus, level.id, env)),
    )

    assert apply_calls == []
