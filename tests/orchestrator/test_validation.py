import hashlib
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import validation
from orchestrator.catalog import KernelEntry
from orchestrator.preflight import CheckResult


def _write_frequency_samples(path: Path, values: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "scaling_cur_freq_khz"])
        writer.writeheader()
        for value in values:
            writer.writerow({"tag": "CPU", "scaling_cur_freq_khz": value})


def _write_multi_cpu_frequency_samples(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tag", "scaling_cur_freq_khz", "scaling_cur_freq_khz_all"],
        )
        writer.writeheader()
        for primary, all_cpus in rows:
            writer.writerow({
                "tag": "CPU",
                "scaling_cur_freq_khz": primary,
                "scaling_cur_freq_khz_all": all_cpus,
            })


def test_arc138_traza_de_frecuencia_fixed_se_valida_por_muestra(tmp_path):
    path = tmp_path / "samples.csv"
    _write_frequency_samples(path, ["3200000", "3150000", "3099999"])
    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
    )
    assert verdict.accepted is False
    assert verdict.factor_id == "E01"
    assert summary["mismatched_samples"] == 1


def test_arc138_traza_ref_exige_lecturas_pero_no_objetivo(tmp_path):
    path = tmp_path / "samples.csv"
    _write_frequency_samples(path, ["3200000", ""])
    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=None, tolerance_fraction=0.05,
    )
    assert verdict.accepted is False
    assert summary["missing_samples"] == 1


def test_arc145_traza_fixed_valida_todos_los_cpus_delegados(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples(path, [
        ("3200000", "3200000;3200000;3200000;3200000"),
        # CPU0 coincide; CPU3 es el que diverge. La validación escalar
        # anterior aceptaba esta ventana en silencio.
        ("3200000", "3200000;3200000;3200000;2800000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path,
        require_per_window=True,
        expected_khz=3_200_000,
        tolerance_fraction=0.03,
        expected_cpu_count=4,
    )

    assert verdict == validation.Verdict(False, "E01", verdict.message)
    assert summary["mismatched_samples"] == 1
    assert summary["observed_spread_max_khz"] == 400_000


def test_arc145_traza_multi_cpu_falla_cerrado_si_falta_un_cpu(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples(path, [
        ("3200000", "3200000;3200000;3200000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path,
        require_per_window=True,
        expected_khz=None,
        tolerance_fraction=0.03,
        expected_cpu_count=4,
    )

    assert verdict.accepted is False
    assert verdict.factor_id == "E01"
    assert summary["cpu_count_mismatch_samples"] == 1
    assert summary["missing_samples"] == 1


def test_arc145_traza_multi_cpu_completa_y_en_tolerancia_acepta(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples(path, [
        ("3200000", "3200000;3150000;3210000;3190000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path,
        require_per_window=True,
        expected_khz=3_200_000,
        tolerance_fraction=0.03,
        expected_cpu_count=4,
    )

    assert verdict.accepted is True
    assert summary["observed_samples"] == 4
    assert summary["missing_samples"] == 0


def test_arc138_validate_run_rechaza_e01_y_conserva_el_resultado_en_metadata(tmp_path):
    entry = _kernel_entry(tmp_path)
    result = _run_result()
    result.metadata["frequency_trace_validation"] = {
        "accepted": False,
        "factor_id": "E01",
        "message": "frecuencia efectiva fuera de objetivo",
    }
    verdict = validation.validate_run(result, entry)
    assert verdict == validation.Verdict(False, "E01", "frecuencia efectiva fuera de objetivo")


def _kernel_entry(tmp_path: Path) -> KernelEntry:
    binary = tmp_path / "npb_ep.x"
    binary.write_bytes(b"#!/bin/sh\necho ok\n")
    binary.chmod(0o755)
    checksum = f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}"
    return KernelEntry(
        id="npb_ep", suite="npb", role="dataset", exec_path=str(binary), binary_checksum=checksum,
        phase_label_hint="compute_bound", size_variant="S", expected_runtime_seconds=1,
        warmup_seconds=0.0, success_check={"type": "exit_code"}, estimated_memory_bytes=1024,
    )


def _run_result(*, run_id="camp__npb_ep__REF__rep01", success=True, samples_collected=100,
                 push_retries=0) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id, success=success,
        metadata={"samples_collected": samples_collected, "push_retries": push_retries},
    )


def test_val01_i04_samples_collected_cero_rechaza(tmp_path):
    entry = _kernel_entry(tmp_path)
    verdict = validation.validate_run(_run_result(samples_collected=0), entry)
    assert verdict.accepted is False
    assert verdict.factor_id == "I04"


def test_val01_i04_push_retries_positivo_rechaza(tmp_path):
    entry = _kernel_entry(tmp_path)
    verdict = validation.validate_run(_run_result(push_retries=3), entry)
    assert verdict.accepted is False
    assert verdict.factor_id == "I04"


def test_val03_c02_checksum_discrepante_rechaza_aunque_la_corrida_termine_bien(tmp_path):
    entry = _kernel_entry(tmp_path)
    # El binario cambia despues de calcular el checksum del catalogo.
    Path(entry.exec_path).write_bytes(b"#!/bin/sh\necho cambiado\n")

    verdict = validation.validate_run(_run_result(success=True), entry)

    assert verdict.accepted is False
    assert verdict.factor_id == "C02"


def test_val04_c03_success_check_no_cumplido_rechaza(tmp_path):
    entry = _kernel_entry(tmp_path)
    verdict = validation.validate_run(_run_result(success=False), entry)
    assert verdict.accepted is False
    assert verdict.factor_id == "C03"


def test_val07_orden_determinista_i04_antes_que_c02(tmp_path):
    entry = _kernel_entry(tmp_path)
    # Checksum roto Y samples_collected=0 a la vez: I04 debe ganar, no C02.
    Path(entry.exec_path).write_bytes(b"#!/bin/sh\necho cambiado\n")

    verdict = validation.validate_run(_run_result(samples_collected=0), entry)

    assert verdict.factor_id == "I04"


def test_val07_orden_determinista_c02_antes_que_c03(tmp_path):
    entry = _kernel_entry(tmp_path)
    Path(entry.exec_path).write_bytes(b"#!/bin/sh\necho cambiado\n")

    verdict = validation.validate_run(_run_result(success=False), entry)

    assert verdict.factor_id == "C02"


def test_val07_orden_determinista_e06_e07_e08_antes_del_resto(tmp_path):
    entry = _kernel_entry(tmp_path)
    foreign = CheckResult("E06", "Procesos ajenos", False, True, {}, "proceso ajeno detectado")
    governor = CheckResult("E07", "Governor", False, True, {}, "governor cambio")

    verdict = validation.validate_run(
        _run_result(run_id="dup"), entry,
        foreign_processes=foreign, governor=governor, run_id_seen={"dup"},
    )

    # E06 se evalua antes que E07 y antes que I07 (run_id duplicado).
    assert verdict.factor_id == "E06"


def test_val02_i07_run_id_duplicado_rechaza(tmp_path):
    entry = _kernel_entry(tmp_path)
    verdict = validation.validate_run(
        _run_result(run_id="camp__npb_ep__REF__rep01"), entry,
        run_id_seen={"camp__npb_ep__REF__rep01"},
    )
    assert verdict.accepted is False
    assert verdict.factor_id == "I07"


def test_corrida_limpia_se_acepta(tmp_path):
    entry = _kernel_entry(tmp_path)
    verdict = validation.validate_run(_run_result(), entry, run_id_seen=set())
    assert verdict.accepted is True
    assert verdict.factor_id is None


def test_val05_d03_calibracion_no_plausible_rechaza_toda_la_campana():
    calibration = SimpleNamespace(plausibility_check_passed=False, plausibility_message="D03: fuera de rango")
    verdict = validation.validate_campaign_calibration(calibration)
    assert verdict.accepted is False
    assert verdict.factor_id == "D03"


def test_val05_calibracion_plausible_acepta():
    calibration = SimpleNamespace(plausibility_check_passed=True)
    verdict = validation.validate_campaign_calibration(calibration)
    assert verdict.accepted is True


def test_val08_rechazo_de_ventana_no_invalida_la_corrida(tmp_path):
    """validate_run ni siquiera recibe windows.csv como argumento: no hay
    forma de que un quality_status de ventana (I01/I02/I03/warmup/
    intensity_undefined) llegue a influir el veredicto de la corrida."""
    import inspect
    signature = inspect.signature(validation.validate_run)
    assert "windows" not in signature.parameters
    assert "quality_status" not in signature.parameters


def test_val06_write_verdict_nunca_borra_conserva_rechazo(tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    (run_dir / "samples.csv").write_text("crudo")

    verdict = validation.Verdict(accepted=False, factor_id="C02", message="checksum discrepante")
    path = validation.write_verdict(verdict, run_dir)

    assert path.exists()
    assert (run_dir / "samples.csv").exists()  # el crudo se conserva

    loaded = validation.load_verdict(run_dir)
    assert loaded == verdict


def _write_windows_csv(path: Path, rows: list[dict]) -> Path:
    import csv
    fieldnames = sorted({key for row in rows for key in row} | {"quality_status", "phase_label_train"})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_val09_windows_por_debajo_del_objetivo_rechaza(tmp_path):
    """ARC-94: antes de este cambio, target_windows_per_repetition se
    declaraba en el manifiesto pero ninguna decision de aceptacion lo
    consultaba de verdad."""
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "ok", "phase_label_train": "compute_bound"} for _ in range(3)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=10, device="cpu")
    assert verdict.accepted is False
    assert verdict.factor_id == "I10"


def test_val09_windows_cero_ok_rechaza_aunque_haya_filas(tmp_path):
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "no_freq_reading", "phase_label_train": None} for _ in range(50)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=10, device="cpu")
    assert verdict.accepted is False
    assert verdict.factor_id == "I10"


def test_val09_windows_suficientes_pero_sin_etiqueta_rechaza(tmp_path):
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "ok", "phase_label_train": ""} for _ in range(10)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=10, device="cpu")
    assert verdict.accepted is False
    assert verdict.factor_id == "I11"


def test_val09_windows_suficientes_y_etiquetadas_acepta(tmp_path):
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "ok", "phase_label_train": "memory_bound"} for _ in range(10)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=10, device="cpu")
    assert verdict.accepted is True


def test_val09_windows_gpu_usa_gpu_telemetry_como_estado_usable(tmp_path):
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "gpu_telemetry", "phase_label_train": "compute_bound", "gpu_util_pct": "50"}
        for _ in range(5)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=5, device="gpu")
    assert verdict.accepted is True
    # Filas 'ok' de CPU no cuentan como validas para un kernel GPU.
    windows_path_mixed = _write_windows_csv(tmp_path / "windows2.csv", [
        {"quality_status": "ok", "phase_label_train": "compute_bound"} for _ in range(5)
    ])
    rechazo = validation.validate_windows(windows_path_mixed, target_windows_per_repetition=5, device="gpu")
    assert rechazo.accepted is False


def test_arc129_windows_gpu_bajo_el_piso_de_ruido_no_cuentan_como_usables(tmp_path):
    # ARC-129: quality_status=="gpu_telemetry" solo no distingue actividad
    # real de una GPU esencialmente ociosa -- se exige gpu_util_pct sobre el
    # mismo piso de measure_warmup.py (5.0%).
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "gpu_telemetry", "phase_label_train": "compute_bound", "gpu_util_pct": "2"}
        for _ in range(5)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=5, device="gpu")
    assert verdict.accepted is False
    assert verdict.factor_id == "I10"


def test_arc129_windows_gpu_util_vacio_no_cuenta_como_usable(tmp_path):
    # ARC-129: gpu_util_pct ausente/no numérico se trata como sin señal,
    # nunca se asume "0 pasa" ni se ignora el filtro.
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "gpu_telemetry", "phase_label_train": "compute_bound", "gpu_util_pct": ""}
        for _ in range(5)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=5, device="gpu")
    assert verdict.accepted is False
    assert verdict.factor_id == "I10"


def test_arc129_windows_gpu_mezcla_sobre_y_bajo_el_piso_solo_cuenta_las_reales(tmp_path):
    rows = [
        {"quality_status": "gpu_telemetry", "phase_label_train": "compute_bound", "gpu_util_pct": "50"}
        for _ in range(3)
    ] + [
        {"quality_status": "gpu_telemetry", "phase_label_train": "compute_bound", "gpu_util_pct": "1"}
        for _ in range(3)
    ]
    windows_path = _write_windows_csv(tmp_path / "windows.csv", rows)
    # Solo 3 de las 6 filas superan el piso -- por debajo de un objetivo de 4.
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=4, device="gpu")
    assert verdict.accepted is False
    assert "3 ventanas" in verdict.message
    # Pero sí alcanzan un objetivo de 3.
    verdict_ok = validation.validate_windows(windows_path, target_windows_per_repetition=3, device="gpu")
    assert verdict_ok.accepted is True
