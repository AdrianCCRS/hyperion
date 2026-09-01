import hashlib
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

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
    # ARC-174: una desviación de tolerancia ya no rechaza la corrida
    # completa -- el Verdict agregado solo falla por integridad
    # estructural. El diagnóstico de tolerancia se conserva en summary
    # para quien lo consulte (p.ej. la advertencia CAL-07).
    path = tmp_path / "samples.csv"
    _write_frequency_samples(path, ["3200000", "3150000", "3099999"])
    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
    )
    assert verdict.accepted is True
    assert summary["structural_valid"] is True
    assert summary["tolerance_all_within"] is False
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

    # ARC-174: estructura íntegra (4 de 4 CPUs presentes en ambas
    # ventanas) -- la desviación de un CPU es solo diagnóstico agregado
    # ahora, no rechaza la corrida completa.
    assert verdict.accepted is True
    assert summary["structural_valid"] is True
    assert summary["tolerance_all_within"] is False
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


def _write_multi_cpu_frequency_samples_with_ts(path: Path, rows: list[tuple[int, str, str]]) -> None:
    """rows: (timestamp_ns, primary, all_cpus)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tag", "timestamp_ns", "scaling_cur_freq_khz", "scaling_cur_freq_khz_all"],
        )
        writer.writeheader()
        for ts_ns, primary, all_cpus in rows:
            writer.writerow({
                "tag": "CPU", "timestamp_ns": ts_ns,
                "scaling_cur_freq_khz": primary, "scaling_cur_freq_khz_all": all_cpus,
            })


def test_arc166_grace_seconds_excluye_muestras_tempranas_de_la_tolerancia(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        # ARC-166: dos ventanas tempranas (t=0s, t=5s) con un CPU rezagado
        # fuera de tolerancia -- dentro de grace_seconds=10.0, no cuentan.
        (0, "3200000", "3200000;3200000;3200000;2400000"),
        (5_000_000_000, "3200000", "3200000;3200000;3200000;2400000"),
        # A partir de t=11s (fuera de la gracia), ya limpio.
        (11_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
        (12_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4, grace_seconds=10.0,
    )

    assert verdict.accepted is True
    assert summary["excluded_by_grace_samples"] == 8
    assert summary["observed_samples"] == 8


def test_arc166_grace_seconds_no_exime_los_chequeos_estructurales(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        # Ventana temprana (dentro de la gracia) con solo 3 de 4 CPUs --
        # sigue siendo un fallo estructural real, no de dispersión.
        (0, "3200000", "3200000;3200000;3200000"),
        (11_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4, grace_seconds=10.0,
    )

    assert verdict.accepted is False
    assert summary["cpu_count_mismatch_samples"] == 1
    assert summary["missing_samples"] == 1


def test_arc166_toda_la_traza_dentro_de_grace_seconds_falla_en_voz_alta(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        (0, "3200000", "3200000;3200000;3200000;3200000"),
        (5_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4, grace_seconds=10.0,
    )

    assert verdict.accepted is False
    assert verdict.factor_id == "E01"
    assert "grace_seconds" in verdict.message


def test_arc166_grace_seconds_default_cero_preserva_comportamiento_anterior(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        (0, "3200000", "3200000;3200000;3200000;2400000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4,
    )

    # ARC-174: estructuralmente íntegra -- la desviación de un CPU ya no
    # rechaza la corrida completa, solo queda como diagnóstico agregado.
    assert verdict.accepted is True
    assert summary["tolerance_all_within"] is False
    assert summary["excluded_by_grace_samples"] == 0


def test_arc169_tail_grace_seconds_excluye_muestras_tardias_de_la_tolerancia(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        # ARC-169: la corrida asienta bien al principio, pero cerca del
        # FINAL un hilo termina antes que sus pares y queda ocioso --
        # dentro de tail_grace_seconds=10.0 (medido hacia atras desde el
        # ultimo tick), no cuenta.
        (0, "3200000", "3200000;3200000;3200000;3200000"),
        (1_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
        (11_000_000_000, "3200000", "3200000;3200000;3200000;2400000"),
        (12_000_000_000, "3200000", "3200000;3200000;3200000;2400000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4, tail_grace_seconds=10.0,
    )

    assert verdict.accepted is True
    assert summary["excluded_by_grace_samples"] == 8
    assert summary["observed_samples"] == 8


def test_arc169_grace_seconds_y_tail_grace_seconds_combinados(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        # Temprano (dentro de grace_seconds) Y tardio (dentro de
        # tail_grace_seconds) ambos fuera de tolerancia; solo el tramo
        # intermedio limpio cuenta.
        (0, "3200000", "3200000;3200000;3200000;2400000"),
        (12_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
        (13_000_000_000, "3200000", "3200000;3200000;3200000;3200000"),
        (25_000_000_000, "3200000", "3200000;3200000;3200000;2400000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4, grace_seconds=10.0, tail_grace_seconds=10.0,
    )

    assert verdict.accepted is True
    assert summary["excluded_by_grace_samples"] == 8
    assert summary["observed_samples"] == 8


def test_arc169_tail_grace_seconds_default_cero_preserva_comportamiento_anterior(tmp_path):
    path = tmp_path / "samples.csv"
    _write_multi_cpu_frequency_samples_with_ts(path, [
        (0, "3200000", "3200000;3200000;3200000;3200000"),
        (1_000_000_000, "3200000", "3200000;3200000;3200000;2400000"),
    ])

    verdict, summary = validation.validate_cpu_frequency_trace(
        path, require_per_window=True, expected_khz=3_200_000, tolerance_fraction=0.03,
        expected_cpu_count=4,
    )

    # ARC-174: estructuralmente íntegra -- ya no rechaza por tolerancia.
    assert verdict.accepted is True
    assert summary["tolerance_all_within"] is False
    assert summary["tail_grace_seconds"] == 0.0


# ARC-174: classify_frequency_window() -- clasificación de frecuencia POR
# VENTANA, el reemplazo del gate agregado de tolerancia.

def test_arc174_seis_cpu_igualmente_desviados_spread_cero_pero_outliers_seis(tmp_path):
    # El spread (max-min) NO detecta este caso -- los 6 CPUs leen
    # exactamente lo mismo (spread=0) pero los 6 están fuera de tolerancia.
    classification = validation.classify_frequency_window(
        "3000000;3000000;3000000;3000000;3000000;3000000",
        is_native_governor=False, expected_khz=3_200_000, tolerance_fraction=0.03,
        within_grace=False,
    )
    assert classification.status == "observation_unreliable"
    assert classification.outlier_cpu_count == 6
    assert classification.min_khz == classification.max_khz == 3_000_000


def test_arc174_un_cpu_desviado(tmp_path):
    classification = validation.classify_frequency_window(
        "3200000;3200000;3200000;2800000",
        is_native_governor=False, expected_khz=3_200_000, tolerance_fraction=0.03,
        within_grace=False,
    )
    assert classification.status == "observation_unreliable"
    assert classification.outlier_cpu_count == 1


def test_arc174_varios_cpu_desviados(tmp_path):
    classification = validation.classify_frequency_window(
        "3200000;2800000;2800000;3200000",
        is_native_governor=False, expected_khz=3_200_000, tolerance_fraction=0.03,
        within_grace=False,
    )
    assert classification.status == "observation_unreliable"
    assert classification.outlier_cpu_count == 2


def test_arc174_valor_exactamente_en_el_limite_de_tolerancia_se_acepta(tmp_path):
    # abs(value - expected) > tolerance_khz es estricto (>) -- el borde
    # exacto (3_200_000 * 0.03 = 96_000 kHz de margen) cae DENTRO.
    classification = validation.classify_frequency_window(
        "3296000;3296000;3296000;3296000",
        is_native_governor=False, expected_khz=3_200_000, tolerance_fraction=0.03,
        within_grace=False,
    )
    assert classification.status == "valid"
    assert classification.outlier_cpu_count == 0


def test_arc174_precedencia_de_unverified_grace_sobre_unreliable(tmp_path):
    # Fuera de tolerancia Y dentro de la ventana de gracia -- gana grace.
    classification = validation.classify_frequency_window(
        "2800000;2800000;2800000;2800000",
        is_native_governor=False, expected_khz=3_200_000, tolerance_fraction=0.03,
        within_grace=True,
    )
    assert classification.status == "observation_unverified_grace"
    # El diagnóstico de outliers se conserva aunque el estado sea grace.
    assert classification.outlier_cpu_count == 4


def test_arc174_ref_valido_no_confundido_con_falta_de_config(tmp_path):
    classification = validation.classify_frequency_window(
        "3600000;3550000;3610000;3590000",
        is_native_governor=True, expected_khz=None, tolerance_fraction=None,
        within_grace=False,
    )
    assert classification.status == "not_applicable_native"
    assert classification.outlier_cpu_count is None
    assert classification.min_khz == 3_550_000
    assert classification.max_khz == 3_610_000


def test_arc174_config_incompleta_falla_cerrado_no_se_confunde_con_ref(tmp_path):
    # is_native_governor=False (nivel fixed) pero sin tolerance_fraction --
    # nunca debe leerse como REF ni como "válida" por defecto.
    classification = validation.classify_frequency_window(
        "3200000;3200000;3200000;3200000",
        is_native_governor=False, expected_khz=3_200_000, tolerance_fraction=None,
        within_grace=False,
    )
    assert classification.status is None


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
        {"quality_status": "ok", "phase_label_train": "", "frequency_quality_status": "valid"} for _ in range(10)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=10, device="cpu")
    assert verdict.accepted is False
    assert verdict.factor_id == "I11"


def test_val09_windows_suficientes_y_etiquetadas_acepta(tmp_path):
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "ok", "phase_label_train": "memory_bound", "frequency_quality_status": "valid"}
        for _ in range(10)
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=10, device="cpu")
    assert verdict.accepted is True


def test_arc174_validate_windows_cpu_exige_calidad_general_y_frecuencia_y_etiqueta(tmp_path):
    # Mezcla: solo la primera fila cumple las tres condiciones a la vez.
    windows_path = _write_windows_csv(tmp_path / "windows.csv", [
        {"quality_status": "ok", "phase_label_train": "compute_bound", "frequency_quality_status": "valid"},
        # calidad general no-ok -- no cuenta aunque frecuencia sea válida.
        {"quality_status": "pmu_degraded", "phase_label_train": "compute_bound", "frequency_quality_status": "valid"},
        # frecuencia no confiable -- no cuenta aunque calidad general sea ok.
        {"quality_status": "ok", "phase_label_train": "compute_bound", "frequency_quality_status": "observation_unreliable"},
        # dentro de gracia -- tampoco cuenta.
        {"quality_status": "ok", "phase_label_train": "compute_bound", "frequency_quality_status": "observation_unverified_grace"},
        # REF (not_applicable_native) sí cuenta si el resto está bien.
        {"quality_status": "ok", "phase_label_train": "compute_bound", "frequency_quality_status": "not_applicable_native"},
    ])
    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=2, device="cpu")
    assert verdict.accepted is True

    verdict_estricto = validation.validate_windows(windows_path, target_windows_per_repetition=3, device="cpu")
    assert verdict_estricto.accepted is False
    assert verdict_estricto.factor_id == "I10"


def test_arc174_summarize_frequency_quality_reporta_cobertura_y_racha_mas_larga(tmp_path):
    # ARC-174: corrida con muchas ventanas inválidas pero suficientes
    # válidas para incluirse -- summarize_frequency_quality() reporta la
    # baja cobertura por separado, sin convertirla en otro umbral de
    # rechazo. 60 'ok', de las cuales 55 son válidas y 5 forman una racha
    # consecutiva 'observation_unreliable'.
    rows = []
    for _ in range(55):
        rows.append({"quality_status": "ok", "phase_label_train": "compute_bound", "frequency_quality_status": "valid"})
    for _ in range(5):
        rows.append({"quality_status": "ok", "phase_label_train": "compute_bound", "frequency_quality_status": "observation_unreliable"})
    for i, row in enumerate(rows):
        row["window_index"] = i
    windows_path = _write_windows_csv(tmp_path / "windows.csv", rows)

    verdict = validation.validate_windows(windows_path, target_windows_per_repetition=50, device="cpu")
    assert verdict.accepted is True

    summary = validation.summarize_frequency_quality(windows_path)
    assert summary["total_candidate_windows"] == 60
    assert summary["valid_window_count"] == 55
    assert summary["frequency_quality_counts"]["observation_unreliable"] == 5
    assert summary["longest_unreliable_streak"] == 5
    assert summary["fraction_valid"] == pytest.approx(55 / 60)


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
