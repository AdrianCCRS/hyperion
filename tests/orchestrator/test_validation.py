import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import validation
from orchestrator.catalog import KernelEntry
from orchestrator.preflight import CheckResult


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
        {"quality_status": "gpu_telemetry", "phase_label_train": "compute_bound"} for _ in range(5)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=5, device="gpu")
    assert verdict.accepted is True
    # Filas 'ok' de CPU no cuentan como validas para un kernel GPU.
    windows_path_mixed = _write_windows_csv(tmp_path / "windows2.csv", [
        {"quality_status": "ok", "phase_label_train": "compute_bound"} for _ in range(5)
    ])
    rechazo = validation.validate_windows(windows_path_mixed, target_windows_per_repetition=5, device="gpu")
    assert rechazo.accepted is False
