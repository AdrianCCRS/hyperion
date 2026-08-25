import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import campaign
from orchestrator import runner as runner_module
from orchestrator import validation as validation_module
from orchestrator.catalog import KernelEntry
from orchestrator.manifest import Combination, Cores, FrequencyLevel, Manifest, Timeouts


def _kernel_entry(tmp_path: Path, kernel_id: str, **overrides) -> KernelEntry:
    binary = tmp_path / f"{kernel_id}.bin"
    binary.write_bytes(f"#!/bin/sh\necho {kernel_id}\n".encode())
    binary.chmod(0o755)
    checksum = f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}"
    defaults = dict(
        id=kernel_id, suite="npb", role="dataset", exec_path=str(binary), binary_checksum=checksum,
        phase_label_hint="compute_bound", size_variant="S", expected_runtime_seconds=1,
        warmup_seconds=0.1, success_check={"type": "exit_code"}, estimated_memory_bytes=1024,
    )
    defaults.update(overrides)
    return KernelEntry(**defaults)


def _manifest(tmp_path: Path, **overrides) -> Manifest:
    output_dir = tmp_path / "runs"
    output_dir.mkdir(exist_ok=True)
    defaults = dict(
        campaign_id="camp01", environment_tier="local", seed=42, output_dir=output_dir, overwrite=True,
        catalog_path=tmp_path / "catalog.yaml", calibration=("stream", "ert"), kernels=("npb_ep",),
        frequency_levels=(FrequencyLevel("REF", "native_governor"),), repetitions_per_combination=1,
        target_windows_per_repetition=10, interval_ns=1_000_000, running_ratio_min=0.9,
        cores=Cores((2, 3, 4, 5), 0, 1, None), smt_policy="all_threads", cgroup_path=None,
        perf_enabled=True, rapl={"enabled": False}, gpu={}, timeouts_seconds=Timeouts(5, 5, 5),
        hardware_datasheet=None,
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def _catalog(tmp_path: Path) -> dict[str, KernelEntry]:
    return {
        "npb_ep": _kernel_entry(tmp_path, "npb_ep"),
        "stream": _kernel_entry(tmp_path, "stream", role="calibration", phase_label_hint=None, size_variant=None,
                                 expected_runtime_seconds=None, warmup_seconds=None, estimated_memory_bytes=None,
                                 reports_bandwidth_stdout=True, bandwidth_stdout_pattern=r"BW=([0-9.]+)"),
        "ert": _kernel_entry(tmp_path, "ert", role="calibration", phase_label_hint=None, size_variant=None,
                              expected_runtime_seconds=None, warmup_seconds=None, estimated_memory_bytes=None,
                              reports_flops_stdout=True, flops_stdout_pattern=r"FLOPS=([0-9.]+)"),
    }


# ARC-141: harness sin binario real (binary_path="") -- compute_protocol_
# fingerprint() ya trata eso como "sin checksum de launcher disponible"
# (None, nunca un valor inventado), así que usar la misma instancia aquí y
# en cada llamada a run_campaign() de los tests de reanudación/fingerprint
# mantiene ambos lados consistentes sin depender de dónde esté compilado
# telemetry_kernel_launcher en la máquina que corre la suite.
_FAKE_HARNESS = SimpleNamespace(binary_path="")


def _write_matching_fingerprint(manifest, catalog, harness=_FAKE_HARNESS):
    """ARC-94: simula un output_dir legitimamente rastreado por CAM-09 --
    sin esto, cualquier test que cree un run_dir con verdict.json a mano
    (para simular una reanudacion) dispara el chequeo de "carpeta legacy
    sin fingerprint" en vez de la ruta de reanudacion real que el test
    quiere ejercitar."""
    fingerprint_path = Path(manifest.output_dir) / "protocol_fingerprint.json"
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        json.dumps({"sha256": campaign.compute_protocol_fingerprint(manifest, catalog, harness)})
    )


def _fake_run_single(calls):
    def run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                    environment_profile=None, node_id=None, apply_frequency=None,
                    apply_gpu_frequency=None, calibration_refs=None, run_id=None,
                    gpu_freq_level_id=None):
        # ARC-94: usa el run_id que el llamador pasa (como hace runner.py
        # real ahora) en vez de reconstruirlo aquí -- antes, este fake
        # reimplementaba por su cuenta el sufijo __baseline correcto,
        # enmascarando que run_single() de producción nunca lo recibía ni
        # lo aplicaba (baseline y telemetry colisionaban en el mismo
        # directorio en la ruta real).
        if run_id is None:
            base_run_id = runner_module.build_run_id(manifest.campaign_id, kernel_ref, freq_level_id, repetition_index)
            run_id = base_run_id if manifest.perf_enabled else f"{base_run_id}__baseline"
        run_dir = Path(manifest.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        calls.append((run_id, manifest.perf_enabled))
        return SimpleNamespace(
            run_id=run_id, kernel_ref=kernel_ref, freq_level_id=freq_level_id, repetition_index=repetition_index,
            exit_code=0, timed_out=False, success=True, elapsed_seconds=1.0, run_dir=run_dir,
            stdout_path=run_dir / "stdout.txt", stderr_path=run_dir / "stderr.txt",
            metadata={"samples_collected": 10, "push_retries": 0}, applied_frequency=None,
        )
    return run_single


def _fake_calibration_deps():
    postprocess_calls = []

    def run_calibration(manifest, catalog, *, environment_profile, node_id, run_single, apply_frequency=None):
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    def build_node_profile(env, cpus, *, node_id, hostname=""):
        return SimpleNamespace(node_id=node_id, cache_line_size_bytes=64)

    def write_node_profile(profile, output_dir):
        return Path(output_dir) / "node_profile.json"

    def run_calibration_references(entry, manifest, kernel_ref, *, node_id, environment_profile, run_single, apply_frequency=None):
        return SimpleNamespace(node_id=node_id, ipc_p95=1.0, accepted=True)

    # CAM-11 (ARC-142): fakes for the resume path -- a resumed campaign
    # loads calibration/profile/references from disk instead of measuring,
    # mirroring the run_* fakes above so tests exercising CAM-03/CAM-08/
    # CAM-09 resumes don't hit the real (file-backed) loaders.
    def load_calibration(output_dir, freq_level_id="", gpu_precision=""):
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    def load_node_profile(output_dir):
        return SimpleNamespace(node_id="fake", cache_line_size_bytes=64)

    def load_calibration_references(output_dir):
        return SimpleNamespace(node_id="fake", ipc_p95=1.0, accepted=True)

    def run_postprocess(run_dir, **kwargs):
        postprocess_calls.append((Path(run_dir), kwargs))
        # ARC-94: validate_windows() ahora LEE este archivo de verdad para
        # decidir el veredicto final -- necesita >=target_windows_per_repetition
        # filas usables con una etiqueta, no solo existir.
        windows_path = Path(run_dir) / "windows.csv"
        entry = kwargs.get("kernel_entry")
        device = getattr(entry, "device", "cpu")
        status = "gpu_telemetry" if device == "gpu" else "ok"
        with open(windows_path, "w", newline="") as handle:
            import csv as _csv
            writer = _csv.DictWriter(
                handle,
                fieldnames=["quality_status", "phase_label_train", "gpu_util_pct", "frequency_quality_status"],
            )
            writer.writeheader()
            for _ in range(10):
                # ARC-129: sobre el piso de ruido (_GPU_UTIL_NOISE_FLOOR_PCT
                # = 5.0) para que las filas GPU sigan contando como usables
                # -- vacío para filas de CPU, donde no aplica.
                # ARC-174: frequency_quality_status="valid" para filas de
                # CPU (validate_windows() ahora lo exige además de
                # quality_status=="ok") -- vacío para GPU, donde nunca se
                # calcula (ver build_windows()).
                writer.writerow({
                    "quality_status": status, "phase_label_train": "compute_bound",
                    "gpu_util_pct": "50" if device == "gpu" else "",
                    "frequency_quality_status": "" if device == "gpu" else "valid",
                })
        return windows_path

    return dict(
        run_calibration=run_calibration, build_node_profile=build_node_profile,
        write_node_profile=write_node_profile, run_calibration_references=run_calibration_references,
        load_calibration=load_calibration, load_node_profile=load_node_profile,
        load_calibration_references=load_calibration_references,
        run_postprocess=run_postprocess,
    ), postprocess_calls


def _freqctl_fakes():
    restore_calls = []
    install_calls = []
    return dict(
        apply_frequency=lambda cpus, level_id, env, **kwargs: None,
        read_observed_frequency_khz=lambda env, cpu: None,
        snapshot_original_state=lambda cpus, env: "SNAPSHOT",
        restore_original_state=lambda state, env: restore_calls.append(state) or True,
        install_emergency_handlers=lambda restore: install_calls.append(restore),
        # ARC-87: espejo de CPU para el eje de GPU -- por defecto no-ops,
        # los tests que sí quieren ejercitar el reloj de GPU sobreescriben
        # estas dos claves explícitamente.
        apply_gpu_frequency=lambda level, env: None,
        restore_gpu_state=lambda env: True,
        # E06: por defecto nadie es "ajeno" -- los tests que sí quieren
        # ejercitar el rechazo sobreescriben esta clave explícitamente.
        # Sin este default, la implementación real escanearía el /proc real
        # de la máquina que corre los tests (no hermético, y casi seguro
        # rechazaría todo ya que nada está pinneado a los cores del test).
        detect_foreign_affinity_pids=lambda cpus, **kwargs: [],
        # E08 (ARC-101): mismo motivo que detect_foreign_affinity_pids arriba
        # -- sin este default, la implementación real llamaría a
        # os.getloadavg() de la máquina que corre los tests (no hermético, y
        # el resultado dependería de cuánta carga real tenga esa máquina en
        # ese instante). Carga "cero" por defecto; los tests que sí quieren
        # ejercitar el rechazo por carga externa sobreescriben esta clave.
        load_reader=lambda: (0.0, 0.0, 0.0),
    ), restore_calls, install_calls


def test_arc80_run_campaign_invoca_run_gpu_calibration_y_lo_expone_en_el_resultado(tmp_path):
    # ARC-80: run_gpu_calibration() es infraestructura separada de la de CPU
    # -- debe invocarse siempre (aunque no haya kernels de GPU, en cuyo caso
    # la implementación real devuelve {} sin hacer nada) y su resultado debe
    # llegar a CampaignResult, igual que roofline_calibration/references.
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    gpu_calibration_calls = []

    def fake_run_gpu_calibration(manifest, catalog, *, environment_profile, node_id, run_single,
                                  apply_frequency=None, apply_gpu_frequency=None):
        gpu_calibration_calls.append(node_id)
        return {"fp32": SimpleNamespace(i_ridge_flops_per_byte=7.28)}

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), run_gpu_calibration=fake_run_gpu_calibration,
        **calibration_deps, **freqctl_deps,
    )

    assert gpu_calibration_calls == ["felix-sc3"]
    assert result.gpu_roofline_calibration["fp32"].i_ridge_flops_per_byte == pytest.approx(7.28)


def test_cam01_build_matrix_es_un_shuffle_plano(tmp_path):
    manifest = _manifest(
        tmp_path, kernels=("npb_ep", "npb_mg"),
        frequency_levels=(FrequencyLevel("REF", "native_governor"), FrequencyLevel("F0", "fixed", 1.0)),
        repetitions_per_combination=3,
    )
    combinations = campaign.build_matrix(manifest, seed=1)

    assert len(combinations) == 2 * 2 * 3
    assert all(isinstance(c, Combination) for c in combinations)
    # Mismo seed -> mismo orden (reproducible), no agrupado por kernel/freq.
    combinations_again = campaign.build_matrix(manifest, seed=1)
    assert combinations == combinations_again
    ordered_by_kernel = sorted(range(len(combinations)), key=lambda i: combinations[i].kernel_ref)
    assert list(range(len(combinations))) != ordered_by_kernel or len({c.kernel_ref for c in combinations}) == 1


def test_arc129_build_matrix_sin_catalog_preserva_comportamiento_anterior(tmp_path):
    # ARC-129: catalog=None (default) -- ni siquiera un kernel_ref de GPU en
    # manifest.kernels puede activar el producto cartesiano sin catálogo
    # para distinguir device=="gpu". Toda combinación queda con
    # gpu_frequency_level=None, tamaño = kernels x frequency_levels x reps,
    # exactamente como antes de este cambio.
    manifest = _manifest(
        tmp_path, kernels=("npb_ep",),
        frequency_levels=(FrequencyLevel("REF", "native_governor"), FrequencyLevel("F0", "fixed", 1.0)),
        gpu_frequency_levels=(FrequencyLevel("GREF", "native_governor"), FrequencyLevel("GF0", "fixed", 1.0)),
        repetitions_per_combination=1,
    )
    combinations = campaign.build_matrix(manifest, seed=1)
    assert len(combinations) == 2
    assert all(c.gpu_frequency_level is None for c in combinations)


def test_arc129_build_matrix_producto_cartesiano_para_kernel_gpu(tmp_path):
    catalog = {
        "npb_ep": _kernel_entry(tmp_path, "npb_ep"),
        "gpu_kernel": _kernel_entry(tmp_path, "gpu_kernel", device="gpu", operational_intensity_flops_per_byte=2.0, gpu_precision="fp32"),
    }
    manifest = _manifest(
        tmp_path, kernels=("npb_ep", "gpu_kernel"),
        frequency_levels=(FrequencyLevel("REF", "native_governor"), FrequencyLevel("F0", "fixed", 1.0)),
        gpu_frequency_levels=(FrequencyLevel("GREF", "native_governor"), FrequencyLevel("GF0", "fixed", 1.0)),
        repetitions_per_combination=1,
    )
    combinations = campaign.build_matrix(manifest, catalog, seed=1)

    npb_ep_combos = [c for c in combinations if c.kernel_ref == "npb_ep"]
    gpu_combos = [c for c in combinations if c.kernel_ref == "gpu_kernel"]
    # npb_ep es CPU -- sin cambios, 2 niveles x 1 rep = 2, gpu_frequency_level=None.
    assert len(npb_ep_combos) == 2
    assert all(c.gpu_frequency_level is None for c in npb_ep_combos)
    # gpu_kernel es GPU con gpu_frequency_levels declarado -- producto
    # cartesiano completo: 2 niveles CPU x 2 niveles GPU x 1 rep = 4.
    assert len(gpu_combos) == 4
    pairs = {(c.frequency_level.id, c.gpu_frequency_level.id) for c in gpu_combos}
    assert pairs == {
        ("REF", "GREF"), ("REF", "GF0"), ("F0", "GREF"), ("F0", "GF0"),
    }


def test_arc172_run_id_for_incluye_el_nivel_de_gpu(tmp_path):
    """ARC-172: _run_id_for() nunca pasaba gpu_freq_level_id a build_run_id()
    -- a diferencia de las otras dos llamadas en campaign.py (run_ids_in_order
    y telemetry_run_id, ambas correctas). Sin el sufijo __gpu<id>, dos
    combinaciones del mismo (kernel, nivel de CPU, repetición) que solo
    difieren en el nivel de GPU colapsaban sobre el mismo run_id -- y por
    tanto el mismo run_dir en disco, sobrescribiéndose entre sí."""
    manifest = _manifest(tmp_path, campaign_id="camp01")
    combo_a = Combination("gpu_kernel", FrequencyLevel("REF", "native_governor"), 1, FrequencyLevel("GREF", "native_governor"))
    combo_b = Combination("gpu_kernel", FrequencyLevel("REF", "native_governor"), 1, FrequencyLevel("GF0", "fixed", 1.0))

    run_id_a = campaign._run_id_for(manifest, campaign.ScheduledRun(combo_a, "telemetry"))
    run_id_b = campaign._run_id_for(manifest, campaign.ScheduledRun(combo_b, "telemetry"))

    assert run_id_a != run_id_b
    assert run_id_a == "camp01__gpu_kernel__REF__gpuGREF__rep01"
    assert run_id_b == "camp01__gpu_kernel__REF__gpuGF0__rep01"


def test_arc172_run_id_for_sin_nivel_de_gpu_no_cambia(tmp_path):
    manifest = _manifest(tmp_path, campaign_id="camp01")
    combo = Combination("npb_ep", FrequencyLevel("REF", "native_governor"), 1, None)

    run_id = campaign._run_id_for(manifest, campaign.ScheduledRun(combo, "telemetry"))

    assert run_id == "camp01__npb_ep__REF__rep01"


def test_arc172_run_id_for_sufijo_baseline_se_conserva_con_nivel_de_gpu(tmp_path):
    manifest = _manifest(tmp_path, campaign_id="camp01")
    combo = Combination("gpu_kernel", FrequencyLevel("REF", "native_governor"), 1, FrequencyLevel("GF0", "fixed", 1.0))

    run_id = campaign._run_id_for(manifest, campaign.ScheduledRun(combo, "baseline"))

    assert run_id == "camp01__gpu_kernel__REF__gpuGF0__rep01__baseline"


def test_arc129_build_matrix_kernel_gpu_sin_gpu_frequency_levels_no_cambia(tmp_path):
    # ARC-129: catálogo presente, kernel es device=="gpu", pero el
    # manifiesto no declaró gpu_frequency_levels -- debe seguir acoplado al
    # eje de CPU (comportamiento anterior a este cambio), no fallar ni
    # inventar un producto cartesiano de la nada.
    catalog = {"gpu_kernel": _kernel_entry(tmp_path, "gpu_kernel", device="gpu", operational_intensity_flops_per_byte=2.0, gpu_precision="fp32")}
    manifest = _manifest(
        tmp_path, kernels=("gpu_kernel",),
        frequency_levels=(FrequencyLevel("REF", "native_governor"), FrequencyLevel("F0", "fixed", 1.0)),
        repetitions_per_combination=1,
    )
    combinations = campaign.build_matrix(manifest, catalog, seed=1)
    assert len(combinations) == 2
    assert all(c.gpu_frequency_level is None for c in combinations)


def test_cam04_schedule_runs_empareja_baseline_y_telemetry(tmp_path):
    manifest = _manifest(tmp_path)
    combinations = campaign.build_matrix(manifest, seed=1)
    scheduled = campaign.schedule_runs(combinations)

    assert len(scheduled) == 2 * len(combinations)
    for i in range(0, len(scheduled), 2):
        assert scheduled[i].mode == "baseline"
        assert scheduled[i + 1].mode == "telemetry"
        assert scheduled[i].combination == scheduled[i + 1].combination


def test_schedule_runs_sin_baseline_repetition_indices_mide_siempre(tmp_path):
    manifest = _manifest(tmp_path, repetitions_per_combination=3)
    combinations = campaign.build_matrix(manifest, seed=1)

    scheduled = campaign.schedule_runs(combinations, None)

    baseline_reps = sorted(
        item.combination.repetition_index for item in scheduled if item.mode == "baseline"
    )
    assert baseline_reps == [1, 2, 3]


def test_schedule_runs_con_baseline_repetition_indices_solo_las_listadas(tmp_path):
    manifest = _manifest(tmp_path, repetitions_per_combination=3)
    combinations = campaign.build_matrix(manifest, seed=1)

    scheduled = campaign.schedule_runs(combinations, (1,))

    baseline_reps = sorted(
        item.combination.repetition_index for item in scheduled if item.mode == "baseline"
    )
    telemetry_reps = sorted(
        item.combination.repetition_index for item in scheduled if item.mode == "telemetry"
    )
    assert baseline_reps == [1]
    # telemetry SIEMPRE corre para las 3 repeticiones -- solo el par de
    # medicion de overhead se restringe, nunca la corrida real de dataset.
    assert telemetry_reps == [1, 2, 3]


def test_campana_completa_corre_baseline_telemetry_y_postprocesa(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calls: list[tuple[str, bool]] = []
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, restore_calls, install_calls = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # CAM-04: exactamente un par baseline+telemetry para la unica combinacion.
    assert calls == [("camp01__npb_ep__REF__rep01__baseline", False), ("camp01__npb_ep__REF__rep01", True)]
    assert result.progress.accepted_run_ids == ["camp01__npb_ep__REF__rep01"]
    assert result.progress.rejected_run_ids == []
    assert len(postprocess_calls) == 1

    # ARC-94: baseline y telemetry deben escribir en directorios DISTINTOS
    # -- antes de este cambio, run_single() ignoraba el run_id con sufijo
    # __baseline que campaign.py calculaba, y ambos colisionaban en el
    # mismo directorio (el segundo pisaba los artefactos del primero).
    baseline_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01__baseline"
    telemetry_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    assert baseline_dir != telemetry_dir
    assert baseline_dir.exists()
    assert telemetry_dir.exists()

    # CAM-07: se restaura exactamente una vez al cerrar, con el snapshot tomado al inicio.
    assert restore_calls == ["SNAPSHOT"]
    assert len(install_calls) == 1

    verdict = validation_module.load_verdict(manifest.output_dir / "camp01__npb_ep__REF__rep01")
    assert verdict.accepted is True

    metadata = json.loads((manifest.output_dir / "campaign_metadata.json").read_text())
    assert metadata["seed"] == 42  # CAM-02
    assert metadata["run_ids_in_order"] == ["camp01__npb_ep__REF__rep01"]
    assert metadata["accepted_run_ids"] == ["camp01__npb_ep__REF__rep01"]
    assert metadata["skipped_run_ids"] == []  # MET-06
    assert metadata["frequency_restored_verified"] is True  # MET-02


def _fake_run_single_con_elapsed_distinto(calls, *, baseline_elapsed, telemetry_elapsed):
    def run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                    environment_profile=None, node_id=None, apply_frequency=None,
                    apply_gpu_frequency=None, calibration_refs=None, run_id=None,
                    gpu_freq_level_id=None):
        if run_id is None:
            base_run_id = runner_module.build_run_id(manifest.campaign_id, kernel_ref, freq_level_id, repetition_index)
            run_id = base_run_id if manifest.perf_enabled else f"{base_run_id}__baseline"
        run_dir = Path(manifest.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        calls.append((run_id, manifest.perf_enabled))
        elapsed = telemetry_elapsed if manifest.perf_enabled else baseline_elapsed
        return SimpleNamespace(
            run_id=run_id, kernel_ref=kernel_ref, freq_level_id=freq_level_id, repetition_index=repetition_index,
            exit_code=0, timed_out=False, success=True, elapsed_seconds=elapsed, run_dir=run_dir,
            stdout_path=run_dir / "stdout.txt", stderr_path=run_dir / "stderr.txt",
            metadata={"samples_collected": 10, "push_retries": 0}, applied_frequency=None,
        )
    return run_single


def test_cam08_overhead_de_instrumentacion_se_calcula_por_par(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calls: list[tuple[str, bool]] = []
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single_con_elapsed_distinto(calls, baseline_elapsed=2.0, telemetry_elapsed=3.0),
        **calibration_deps, **freqctl_deps,
    )

    # (3.0 - 2.0) / 2.0 * 100 = 50%
    assert result.progress.overhead_pct_values == [50.0]

    metadata = json.loads((manifest.output_dir / "campaign_metadata.json").read_text())
    assert metadata["overhead_pct_values"] == [50.0]


def _fake_run_single_baseline_falla(calls, *, baseline_elapsed, telemetry_elapsed):
    def run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                    environment_profile=None, node_id=None, apply_frequency=None,
                    apply_gpu_frequency=None, calibration_refs=None, run_id=None,
                    gpu_freq_level_id=None):
        if run_id is None:
            base_run_id = runner_module.build_run_id(manifest.campaign_id, kernel_ref, freq_level_id, repetition_index)
            run_id = base_run_id if manifest.perf_enabled else f"{base_run_id}__baseline"
        run_dir = Path(manifest.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        calls.append((run_id, manifest.perf_enabled))
        is_baseline = not manifest.perf_enabled
        elapsed = baseline_elapsed if is_baseline else telemetry_elapsed
        return SimpleNamespace(
            run_id=run_id, kernel_ref=kernel_ref, freq_level_id=freq_level_id, repetition_index=repetition_index,
            # El baseline "corre" (elapsed_seconds > 0, un crash temprano
            # sigue tomando algo de tiempo) pero termina en error -- exit_code
            # distinto de 0, success=False.
            exit_code=1 if is_baseline else 0, timed_out=False, success=not is_baseline,
            elapsed_seconds=elapsed, run_dir=run_dir,
            stdout_path=run_dir / "stdout.txt", stderr_path=run_dir / "stderr.txt",
            metadata={"samples_collected": 10, "push_retries": 0}, applied_frequency=None,
        )
    return run_single


def test_arc142_baseline_fallido_no_se_usa_para_calcular_overhead_pct(tmp_path):
    """ARC-142: un baseline con success=False (crash, kernel que salió con
    error) puede tener elapsed_seconds > 0 sin haber corrido el workload
    completo -- no debe usarse como referencia de overhead_pct, que
    quedaría inflado/sin sentido comparado contra un tiempo que no
    representa una corrida real."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calls: list[tuple[str, bool]] = []
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single_baseline_falla(calls, baseline_elapsed=0.1, telemetry_elapsed=3.0),
        **calibration_deps, **freqctl_deps,
    )

    assert result.progress.overhead_pct_values == []


def test_cam08_reanudacion_no_agrega_overhead_para_combinacion_saltada(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_id = "camp01__npb_ep__REF__rep01"
    run_dir = manifest.output_dir / run_id
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(
        validation_module.Verdict(accepted=True, factor_id=None, message=""), run_dir
    )
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()
    calls: list[tuple[str, bool]] = []

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    assert calls == []  # CAM-03: combinacion ya aceptada, no vuelve a correr el par
    assert result.progress.overhead_pct_values == []


def test_arc142_reanudacion_no_resetea_horas_nucleo_ni_overhead_previos(tmp_path):
    """ARC-142: CampaignProgress() arranca en cero en cada invocación de
    run_campaign() -- sin cargar el campaign_metadata.json previo, una
    reanudación sobrescribía total_core_hours/overhead_pct_values con solo
    lo medido en la sesión actual, descartando en silencio lo acumulado
    antes."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_id = "camp01__npb_ep__REF__rep01"
    run_dir = manifest.output_dir / run_id
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(
        validation_module.Verdict(accepted=True, factor_id=None, message=""), run_dir
    )
    # Simula el campaign_metadata.json que una sesión anterior ya dejó en
    # disco, con horas-núcleo/overhead acumulados de corridas que ya no se
    # vuelven a medir en esta reanudación (la combinación de arriba ya está
    # aceptada, CAM-03).
    (manifest.output_dir / "campaign_metadata.json").write_text(json.dumps({
        "campaign_id": "camp01", "seed": 42, "run_ids_in_order": [run_id],
        "accepted_run_ids": [run_id], "rejected_run_ids": [], "skipped_run_ids": [],
        "total_core_hours": 2.5, "overhead_pct_values": [7.0],
    }))

    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    assert result.progress.total_core_hours == 2.5
    assert result.progress.overhead_pct_values == [7.0]


def test_arc94_pocas_ventanas_utiles_rechaza_pese_a_success_check_ok(tmp_path):
    """ARC-94: antes de este cambio, una corrida con success_check y
    samples_collected>0 quedaba accepted=true sin importar cuantas
    ventanas 'ok' produjera postprocess -- target_windows_per_repetition
    se declaraba en el manifiesto pero nunca se aplicaba de verdad."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, postprocess_calls = _fake_calibration_deps()

    def run_postprocess_pocas_ventanas(run_dir, **kwargs):
        postprocess_calls.append((Path(run_dir), kwargs))
        windows_path = Path(run_dir) / "windows.csv"
        import csv as _csv
        with open(windows_path, "w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=["quality_status", "phase_label_train"])
            writer.writeheader()
            writer.writerow({"quality_status": "ok", "phase_label_train": "compute_bound"})  # 1 < target=10
        return windows_path

    calibration_deps["run_postprocess"] = run_postprocess_pocas_ventanas
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    run_id = "camp01__npb_ep__REF__rep01"
    assert len(postprocess_calls) == 1  # postprocess SI corrio (a diferencia de E06)
    assert result.progress.rejected_run_ids == [run_id]
    assert result.progress.accepted_run_ids == []

    verdict = validation_module.load_verdict(manifest.output_dir / run_id)
    assert verdict.accepted is False
    assert verdict.factor_id == "I10"


def test_e06_procesos_ajenos_saltan_la_combinacion_sin_medir(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()
    freqctl_deps["detect_foreign_affinity_pids"] = lambda cpus, **kwargs: [9999]
    calls: list[tuple[str, bool]] = []

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # No se ejecuta ni el baseline ni el telemetry -- se salta ANTES de medir.
    assert calls == []
    assert len(postprocess_calls) == 0
    run_id = "camp01__npb_ep__REF__rep01"
    assert result.progress.rejected_run_ids == [run_id]
    assert result.progress.accepted_run_ids == []

    verdict = validation_module.load_verdict(manifest.output_dir / run_id)
    assert verdict.accepted is False
    assert verdict.factor_id == "E06"
    assert "9999" in verdict.message


def test_arc129_g01_procesos_cuda_ajenos_saltan_la_combinacion_sin_medir(tmp_path):
    # ARC-129: mismo patron que test_e06_..., pero para el eje de GPU --
    # G01 (procesos CUDA ajenos) ahora se corre por combinacion, no solo
    # una vez al inicio de la campana.
    catalog = dict(_catalog(tmp_path))
    catalog["gpu_kernel"] = _kernel_entry(
        tmp_path, "gpu_kernel", device="gpu",
        operational_intensity_flops_per_byte=2.0, gpu_precision="fp32",
    )
    manifest = _manifest(tmp_path, kernels=("gpu_kernel",))
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()
    calls: list[tuple[str, bool]] = []

    gpu_inspector = SimpleNamespace(
        active_processes=lambda: [1234],
        persistence_mode=lambda: True,
        mig_configuration=lambda: "disabled",
    )

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="gpu_kernel",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
        gpu_inspector=gpu_inspector,
    )

    # No se ejecuta ni el baseline ni el telemetry -- se salta ANTES de medir.
    assert calls == []
    assert len(postprocess_calls) == 0
    run_id = "camp01__gpu_kernel__REF__rep01"
    assert result.progress.rejected_run_ids == [run_id]
    assert result.progress.accepted_run_ids == []

    verdict = validation_module.load_verdict(manifest.output_dir / run_id)
    assert verdict.accepted is False
    assert verdict.factor_id == "G01"
    assert "1234" in verdict.message


def test_arc129_g01_sin_gpu_inspector_no_bloquea_kernel_gpu(tmp_path):
    # ARC-129: gpu_inspector=None (el default) desactiva el check por
    # completo -- una campana con kernels de GPU pero sin inspector NVML
    # disponible no debe rechazar TODO por G01 (eso ya es lo que hacía
    # check_gpu_foreign_activity(None) para la campaña completa, un
    # comportamiento intencional distinto al de por-combinación).
    catalog = dict(_catalog(tmp_path))
    catalog["gpu_kernel"] = _kernel_entry(
        tmp_path, "gpu_kernel", device="gpu",
        operational_intensity_flops_per_byte=2.0, gpu_precision="fp32",
    )
    manifest = _manifest(tmp_path, kernels=("gpu_kernel",))
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()
    calls: list[tuple[str, bool]] = []

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="gpu_kernel",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    run_id = "camp01__gpu_kernel__REF__rep01"
    assert result.progress.rejected_run_ids == []
    assert run_id in result.progress.accepted_run_ids


def test_arc102_e08_carga_externa_salta_la_combinacion_sin_medir(tmp_path):
    # ARC-102: mismo patron que test_e06_procesos_ajenos_saltan_la_
    # combinacion_sin_medir, pero para carga externa -- el revisor senalo
    # correctamente que todos los tests previos solo inyectaban carga cero
    # (load_reader=lambda: (0.0,0.0,0.0) en _freqctl_fakes()), sin ejercitar
    # nunca el rechazo real desde run_campaign().
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()
    # Carga baja para el chequeo previo a calibrar (una sola lectura), alta
    # de ahi en adelante -- para aislar el rechazo POR COMBINACION (dentro
    # del bucle de la matriz) del aborto previo a calibrar que ya cubre
    # test_arc102_e08_antes_de_calibrar_aborta_toda_la_campana.
    load_reads = [0.0]

    def load_reader():
        current = load_reads[0]
        load_reads[0] = 999.0
        return (current, 0.0, 0.0)

    freqctl_deps["load_reader"] = load_reader
    calls: list[tuple[str, bool]] = []

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # No se ejecuta ni el baseline ni el telemetry -- se salta ANTES de medir.
    assert calls == []
    assert len(postprocess_calls) == 0
    run_id = "camp01__npb_ep__REF__rep01"
    assert result.progress.rejected_run_ids == [run_id]
    assert result.progress.accepted_run_ids == []

    verdict = validation_module.load_verdict(manifest.output_dir / run_id)
    assert verdict.accepted is False
    assert verdict.factor_id == "E08"


def test_arc102_e08_antes_de_calibrar_aborta_toda_la_campana(tmp_path):
    # ARC-102: a diferencia de una combinacion individual (que se salta y
    # la campana sigue), carga externa alta ANTES de calibrar debe abortar
    # la campana completa -- no hay run_id todavia que rechazar, y una
    # calibracion contaminada desplazaria el ridge point de todas las
    # ventanas clasificadas contra ella.
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    freqctl_deps, _, _ = _freqctl_fakes()
    freqctl_deps["load_reader"] = lambda: (999.0, 0.0, 0.0)

    calibration_calls = []

    def fake_run_calibration(manifest, catalog, *, environment_profile, node_id, run_single, apply_frequency=None):
        calibration_calls.append(True)
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    with pytest.raises(campaign.CampaignPreflightError, match="E08"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), run_calibration=fake_run_calibration,
            **freqctl_deps,
        )

    # run_calibration() nunca debe llegar a llamarse -- el chequeo E08 va
    # ANTES de la primera medicion real, no despues.
    assert calibration_calls == []


def test_arc138_e02_sensor_requerido_ausente_aborta_antes_de_calibrar(tmp_path):
    manifest = replace(
        _manifest(tmp_path),
        temperature={"require_package_sensor": True, "minimum_c": 0, "maximum_c": 90},
    )
    catalog = _catalog(tmp_path)
    freqctl_deps, _, _ = _freqctl_fakes()
    calibration_calls = []

    def fake_run_calibration(manifest, catalog, *, environment_profile, node_id, run_single, apply_frequency=None):
        calibration_calls.append(True)
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    with pytest.raises(campaign.CampaignPreflightError, match="E02"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), run_calibration=fake_run_calibration,
            package_temperature_reader=lambda: None,
            **freqctl_deps,
        )

    assert calibration_calls == []


def test_e06_sin_procesos_ajenos_corre_normalmente(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()
    calls: list[tuple[str, bool]] = []

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    assert calls == [("camp01__npb_ep__REF__rep01__baseline", False), ("camp01__npb_ep__REF__rep01", True)]
    assert result.progress.accepted_run_ids == ["camp01__npb_ep__REF__rep01"]


def test_frq03_frq10_frecuencia_solicitada_aplicada_y_observada_llegan_a_postprocess(tmp_path):
    manifest = _manifest(
        tmp_path, frequency_levels=(FrequencyLevel("F0", "fixed", 1.0),),
    )
    catalog = _catalog(tmp_path)
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    applied = SimpleNamespace(requested_khz=2261000, applied_khz=2261000, governor_applied="userspace")

    def run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                    environment_profile=None, node_id=None, apply_frequency=None,
                    apply_gpu_frequency=None, calibration_refs=None, run_id=None,
                    gpu_freq_level_id=None):
        if run_id is None:
            base_run_id = runner_module.build_run_id(manifest.campaign_id, kernel_ref, freq_level_id, repetition_index)
            run_id = base_run_id if manifest.perf_enabled else f"{base_run_id}__baseline"
        run_dir = Path(manifest.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            run_id=run_id, kernel_ref=kernel_ref, freq_level_id=freq_level_id, repetition_index=repetition_index,
            exit_code=0, timed_out=False, success=True, elapsed_seconds=1.0, run_dir=run_dir,
            stdout_path=run_dir / "stdout.txt", stderr_path=run_dir / "stderr.txt",
            metadata={"samples_collected": 10, "push_retries": 0},
            applied_frequency=applied if manifest.perf_enabled else None,
        )

    freqctl_deps["read_observed_frequency_khz"] = lambda env, cpu: 2261000

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=True),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=run_single, **calibration_deps, **freqctl_deps,
    )

    assert len(postprocess_calls) == 1
    _, kwargs = postprocess_calls[0]
    assert kwargs["freq_khz_requested"] == 2261000
    assert kwargs["freq_khz_applied"] == 2261000
    assert kwargs["freq_khz_observed"] == 2261000


def test_arc174_run_postprocess_recibe_insumos_de_clasificacion_de_frecuencia(tmp_path):
    # ARC-174: campaign.py debe pasar tolerancia/gracias/cantidad de CPU/
    # modo del nivel a run_postprocess() -- sin este plumbing,
    # build_windows() nunca podría clasificar ninguna ventana por
    # frecuencia en la ruta en vivo.
    manifest = _manifest(
        tmp_path,
        frequency_levels=(FrequencyLevel("F0", "fixed", 1.0),),
        frequency_validation={"tolerance_fraction": 0.03, "grace_seconds": 1.5, "tail_grace_seconds": 2.5},
    )
    catalog = _catalog(tmp_path)
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    assert len(postprocess_calls) == 1
    _, kwargs = postprocess_calls[0]
    assert kwargs["freq_tolerance_fraction"] == 0.03
    assert kwargs["freq_grace_seconds"] == 1.5
    assert kwargs["freq_tail_grace_seconds"] == 2.5
    assert kwargs["freq_expected_cpu_count"] == len(manifest.cores.delegated_cpus)
    assert kwargs["freq_is_native_governor"] is False


def test_cam09_primera_corrida_escribe_el_fingerprint_de_protocolo(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    fingerprint_path = manifest.output_dir / "protocol_fingerprint.json"
    assert fingerprint_path.exists()
    stored = json.loads(fingerprint_path.read_text())
    assert stored["sha256"] == campaign.compute_protocol_fingerprint(manifest, catalog, _FAKE_HARNESS)


def test_cam09_reanudacion_con_mismo_protocolo_no_falla(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )
    # Segunda invocacion (reanudacion real): mismo manifest/catalog objects,
    # la combinacion ya aceptada se salta (CAM-03), no debe fallar por CAM-09.
    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )
    assert result.progress.skipped_run_ids == ["camp01__npb_ep__REF__rep01"]


def test_cam09_reanudacion_con_protocolo_distinto_falla_cerrado(tmp_path):
    """ARC-94: reproduce el bug real encontrado en pacca_gpu_ref_20260807
    (mezcla de gpu_interval_ns=100ms y 5ms bajo el mismo campaign_id) --
    antes de este cambio, nada comparaba el manifiesto entre reanudaciones,
    así que corridas viejas de un protocolo distinto seguían marcadas
    'accepted' cuando el manifiesto cambiaba."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    manifest_protocolo_nuevo = _manifest(tmp_path, interval_ns=5_000_000)  # antes 1_000_000
    with pytest.raises(campaign.CampaignProtocolMismatchError, match="CAM-09"):
        campaign.run_campaign(
            manifest_protocolo_nuevo, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
        )


def test_cam09_carpeta_legacy_sin_fingerprint_no_se_adopta_en_silencio(tmp_path):
    """ARC-94 (segunda ronda): las dos carpetas reales de campaña
    (pacca_gpu_ref_20260807, etc.) ya tienen verdict.json pero nunca
    tuvieron protocol_fingerprint.json -- sin este chequeo, la primera
    corrida con CAM-09 ya activo las habría 'adoptado' escribiendo el
    fingerprint actual, exactamente el escenario que CAM-09 existe para
    prevenir."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)
    # A proposito: NO se escribe protocol_fingerprint.json aqui.

    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    with pytest.raises(campaign.CampaignProtocolMismatchError, match="nunca tuvo protocol_fingerprint"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
        )
    # Tampoco debe haber escrito un fingerprint nuevo como efecto secundario del intento fallido.
    assert not (manifest.output_dir / "protocol_fingerprint.json").exists()


def test_cam09_carpeta_vacia_sin_corridas_previas_si_escribe_fingerprint(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )
    assert (manifest.output_dir / "protocol_fingerprint.json").exists()


def test_cam09_fingerprint_cambia_con_gpu_enabled_rapl_y_perf_enabled(tmp_path):
    """ARC-94 (segunda ronda): confirmado que el fingerprint original NO
    cambiaba con gpu.enabled/gpu.calibration/rapl.enabled/perf_enabled,
    pese a que los cuatro afectan directamente qué mide una corrida."""
    manifest_base = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    base = campaign.compute_protocol_fingerprint(manifest_base, catalog)

    variantes = [
        _manifest(tmp_path, gpu={"enabled": True}),
        _manifest(tmp_path, gpu={"enabled": False, "calibration": ["stream"]}),
        _manifest(tmp_path, rapl={"enabled": True, "domains": ["package"]}),
        _manifest(tmp_path, perf_enabled=False),
    ]
    for variante in variantes:
        assert campaign.compute_protocol_fingerprint(variante, catalog) != base


def test_arc141_fingerprint_cubre_campos_de_protocolo_omitidos(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    base = campaign.compute_protocol_fingerprint(manifest, catalog, _FAKE_HARNESS)

    variants = [
        replace(manifest, seed=manifest.seed + 1),
        replace(manifest, uncore={"enabled": True}),
        replace(manifest, timeouts_seconds=Timeouts(7, 11, 13)),
        replace(manifest, load_threshold=0.25),
        replace(manifest, hardware_datasheet={"bw_pico_bytes_per_s": 1.0}),
        replace(manifest, gpu_frequency_levels=(FrequencyLevel("GREF", "native_governor"),)),
    ]

    for variant in variants:
        assert campaign.compute_protocol_fingerprint(variant, catalog, _FAKE_HARNESS) != base


def test_arc141_fingerprint_cambia_si_cambia_el_launcher(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    launcher = tmp_path / "telemetry_kernel_launcher"
    launcher.write_bytes(b"version-a")
    harness = SimpleNamespace(binary_path=str(launcher))
    before = campaign.compute_protocol_fingerprint(manifest, catalog, harness)

    launcher.write_bytes(b"version-b")

    assert campaign.compute_protocol_fingerprint(manifest, catalog, harness) != before


def test_cam09_fingerprint_cambia_con_argumentos_de_kernel(tmp_path):
    """El fingerprint no solo depende del manifiesto -- exec_args/checksum/
    warmup_seconds del catálogo también definen el protocolo real medido."""
    manifest = _manifest(tmp_path)
    catalog_a = _catalog(tmp_path)
    catalog_b = _catalog(tmp_path)
    catalog_b["npb_ep"] = _kernel_entry(tmp_path, "npb_ep", warmup_seconds=99.0)

    assert campaign.compute_protocol_fingerprint(manifest, catalog_a) != campaign.compute_protocol_fingerprint(manifest, catalog_b)


def test_arc95_fingerprint_cambia_con_device_del_kernel(tmp_path):
    """ARC-95: cambiar un kernel de device=cpu a device=gpu en el catálogo
    cambia si se le aplica frecuencia de GPU y como se valida su
    telemetría -- el fingerprint no lo notaba."""
    manifest = _manifest(tmp_path)
    catalog_a = _catalog(tmp_path)
    catalog_b = _catalog(tmp_path)
    catalog_b["npb_ep"] = _kernel_entry(
        tmp_path, "npb_ep", device="gpu", operational_intensity_flops_per_byte=5.0, gpu_precision="fp32",
    )

    assert campaign.compute_protocol_fingerprint(manifest, catalog_a) != campaign.compute_protocol_fingerprint(manifest, catalog_b)


def test_arc95_fingerprint_cambia_con_campaign_id(tmp_path):
    manifest_a = _manifest(tmp_path, campaign_id="campA")
    manifest_b = _manifest(tmp_path, campaign_id="campB")
    catalog = _catalog(tmp_path)

    assert campaign.compute_protocol_fingerprint(manifest_a, catalog) != campaign.compute_protocol_fingerprint(manifest_b, catalog)


def test_cam03_reanudacion_salta_combinacion_ya_aceptada(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    calls: list[tuple[str, bool]] = []
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # Ni el baseline ni el telemetry de la combinacion ya aceptada se repiten.
    assert calls == []
    assert result.progress.accepted_run_ids == []
    assert result.progress.skipped_run_ids == ["camp01__npb_ep__REF__rep01"]  # MET-06
    assert postprocess_calls == []


def test_cam11_reanudacion_carga_calibracion_en_vez_de_remedirla(tmp_path):
    """CAM-11 (ARC-142): una reanudacion (ya hay verdict.json en output_dir)
    no debe llamar run_calibration/run_gpu_calibration/run_calibration_
    references/build_node_profile -- eso remediria y sobrescribiria los
    archivos que las corridas ya aceptadas referencian por ruta. Debe cargar
    con load_calibration/load_node_profile/load_calibration_references."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    calibration_deps, _ = _fake_calibration_deps()
    measure_calls: list[str] = []

    def run_calibration_debe_no_llamarse(*args, **kwargs):
        measure_calls.append("run_calibration")
        raise AssertionError("CAM-11: no debe remedirse calibracion en una reanudacion")

    def load_calibration_marcada(output_dir, freq_level_id="", gpu_precision=""):
        measure_calls.append("load_calibration")
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    calibration_deps["run_calibration"] = run_calibration_debe_no_llamarse
    calibration_deps["load_calibration"] = load_calibration_marcada
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    assert measure_calls == ["load_calibration"]


def test_cam11_reanudacion_sin_calibracion_en_disco_falla_cerrado(tmp_path):
    """CAM-11 (ARC-142): si output_dir ya tiene corridas aceptadas pero al
    cargador de calibracion le falta el archivo esperado, la reanudacion
    debe fallar cerrado (CampaignCalibrationMissingError), nunca remedir en
    silencio."""
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    calibration_deps, _ = _fake_calibration_deps()

    def load_calibration_faltante(output_dir, freq_level_id="", gpu_precision=""):
        raise FileNotFoundError("roofline_calibration_REF.json no existe")

    calibration_deps["load_calibration"] = load_calibration_faltante
    freqctl_deps, _, _ = _freqctl_fakes()

    with pytest.raises(campaign.CampaignCalibrationMissingError, match="CAM-11"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
            run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
        )


def test_cam11_reanudacion_comprueba_todos_los_ridges_cpu_antes_de_medir(tmp_path):
    manifest = _manifest(
        tmp_path,
        frequency_levels=(
            FrequencyLevel("REF", "native_governor"),
            FrequencyLevel("F4", "fixed", 0.0),
        ),
    )
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    calibration_deps, _ = _fake_calibration_deps()
    loaded: list[str] = []

    def load_calibration_con_f4_faltante(output_dir, freq_level_id="", gpu_precision=""):
        loaded.append(freq_level_id)
        if freq_level_id == "F4":
            raise FileNotFoundError("roofline_calibration_F4.json no existe")
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    calibration_deps["load_calibration"] = load_calibration_con_f4_faltante
    freqctl_deps, _, _ = _freqctl_fakes()

    with pytest.raises(campaign.CampaignCalibrationMissingError, match="CAM-11"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
            run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
        )

    assert loaded == ["REF", "F4"]


def test_cam11_calibracion_sin_primer_veredicto_tambien_activa_reanudacion(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    # Simula interrupción después de escribir el ridge pero antes de la
    # primera combinación/verdict.json.
    (manifest.output_dir / "roofline_calibration_REF.json").write_text("{}")

    calibration_deps, _ = _fake_calibration_deps()
    calls: list[str] = []

    def run_calibration_debe_no_llamarse(*args, **kwargs):
        calls.append("run")
        raise AssertionError("no debe sobrescribir la calibración existente")

    def load_calibration_marcada(*args, **kwargs):
        calls.append("load")
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    calibration_deps["run_calibration"] = run_calibration_debe_no_llamarse
    calibration_deps["load_calibration"] = load_calibration_marcada
    freqctl_deps, _, _ = _freqctl_fakes()

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    assert calls == ["load"]


def test_cam03_reanudacion_reintenta_combinacion_rechazada(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(False, "C02", "checksum discrepante"), run_dir)

    calls: list[tuple[str, bool]] = []
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # Un rechazo no es "hecho": se reintenta el par completo.
    assert calls == [("camp01__npb_ep__REF__rep01__baseline", False), ("camp01__npb_ep__REF__rep01", True)]
    assert result.progress.accepted_run_ids == ["camp01__npb_ep__REF__rep01"]

    # CAM-10 (ARC-94): el verdict.json del rechazo original NO se pisa --
    # VAL-06 promete que una corrida rechazada nunca se borra.
    archived_verdict = validation_module.load_verdict(manifest.output_dir / "camp01__npb_ep__REF__rep01__rejected1")
    assert archived_verdict.accepted is False
    assert archived_verdict.factor_id == "C02"
    assert archived_verdict.message == "checksum discrepante"
    # El directorio activo ahora tiene el veredicto NUEVO (aceptado).
    nuevo_verdict = validation_module.load_verdict(manifest.output_dir / "camp01__npb_ep__REF__rep01")
    assert nuevo_verdict.accepted is True


def test_cam10_reanudacion_archiva_corrida_incompleta_antes_de_reintentar(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    _write_matching_fingerprint(manifest, catalog)
    telemetry_id = "camp01__npb_ep__REF__rep01"
    baseline_id = f"{telemetry_id}__baseline"
    for run_id in (telemetry_id, baseline_id):
        run_dir = manifest.output_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "samples.csv").write_text("crudo interrumpido")

    calls: list[tuple[str, bool]] = []
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep", harness=_FAKE_HARNESS,
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    assert result.progress.accepted_run_ids == [telemetry_id]
    assert (manifest.output_dir / f"{telemetry_id}__incomplete1" / "samples.csv").read_text() == "crudo interrumpido"
    assert (manifest.output_dir / f"{baseline_id}__incomplete1" / "samples.csv").read_text() == "crudo interrumpido"
    assert validation_module.load_verdict(manifest.output_dir / telemetry_id).accepted is True


def test_cam10_reintentos_multiples_archivan_cada_rechazo_por_separado(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    output_dir = Path(manifest.output_dir)
    (output_dir / "camp01__npb_ep__REF__rep01").mkdir(parents=True)
    (output_dir / "camp01__npb_ep__REF__rep01__rejected1").mkdir(parents=True)

    campaign._archive_rejected_run(output_dir, "camp01__npb_ep__REF__rep01")

    assert (output_dir / "camp01__npb_ep__REF__rep01__rejected1").exists()
    assert (output_dir / "camp01__npb_ep__REF__rep01__rejected2").exists()
    assert not (output_dir / "camp01__npb_ep__REF__rep01").exists()


def test_cam06_timeout_de_campana_no_se_cuelga(tmp_path, monkeypatch):
    manifest = _manifest(
        tmp_path, kernels=("npb_ep",),
        frequency_levels=(FrequencyLevel("REF", "native_governor"),), repetitions_per_combination=2,
    )
    catalog = _catalog(tmp_path)
    calls: list[tuple[str, bool]] = []
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, restore_calls, _ = _freqctl_fakes()

    # El reloj "avanza" mucho apenas arranca la matriz, forzando el timeout
    # antes de completar la segunda combinacion.
    clock = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(clock, 100.0))

    with pytest.raises(campaign.CampaignTimeoutError):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep", campaign_timeout_seconds=1.0,
            run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
        )

    # CAM-07: incluso al abortar por timeout, se restaura la frecuencia original.
    assert restore_calls == ["SNAPSHOT"]


def test_cam07_restaura_incluso_si_la_calibracion_falla(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    freqctl_deps, restore_calls, _ = _freqctl_fakes()

    def failing_run_calibration(manifest, catalog, *, environment_profile, node_id, run_single, apply_frequency=None):
        raise RuntimeError("D03: fuera de rango")

    with pytest.raises(RuntimeError, match="D03"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), run_calibration=failing_run_calibration, **freqctl_deps,
        )

    assert restore_calls == ["SNAPSHOT"]


def test_arc87_run_gpu_calibration_recibe_apply_gpu_frequency(tmp_path):
    # ARC-87: run_gpu_calibration necesita apply_gpu_frequency para poder
    # fijar el reloj de GPU por nivel (F0-F4) -- sin esto, "calibrar el
    # ridge point por nivel" mediría el mismo reloj físico en todos.
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    received = {}

    def fake_run_gpu_calibration(manifest, catalog, *, environment_profile, node_id, run_single,
                                  apply_frequency=None, apply_gpu_frequency=None):
        received["apply_gpu_frequency"] = apply_gpu_frequency
        return {}

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), run_gpu_calibration=fake_run_gpu_calibration,
        **calibration_deps, **freqctl_deps,
    )

    assert received["apply_gpu_frequency"] is freqctl_deps["apply_gpu_frequency"]


def test_arc87_run_single_recibe_apply_gpu_frequency_en_la_matriz(tmp_path):
    # ARC-87: cada corrida real de la matriz (no solo la calibración) debe
    # recibir apply_gpu_frequency, para que un kernel device="gpu" pueda
    # fijar su reloj en cada combinación, igual que ya ocurre con CPU.
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    received_kwargs = []

    def fake_run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                         environment_profile=None, node_id=None, apply_frequency=None,
                         apply_gpu_frequency=None, calibration_refs=None, run_id=None,
                         gpu_freq_level_id=None):
        received_kwargs.append(apply_gpu_frequency)
        if run_id is None:
            base_run_id = runner_module.build_run_id(manifest.campaign_id, kernel_ref, freq_level_id, repetition_index)
            run_id = base_run_id if manifest.perf_enabled else f"{base_run_id}__baseline"
        run_dir = Path(manifest.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            run_id=run_id, kernel_ref=kernel_ref, freq_level_id=freq_level_id, repetition_index=repetition_index,
            exit_code=0, timed_out=False, success=True, elapsed_seconds=1.0, run_dir=run_dir,
            stdout_path=run_dir / "stdout.txt", stderr_path=run_dir / "stderr.txt",
            metadata={"samples_collected": 10, "push_retries": 0}, applied_frequency=None,
        )

    campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=fake_run_single, **calibration_deps, **freqctl_deps,
    )

    assert len(received_kwargs) == 2  # baseline + telemetry, una repeticion
    assert all(fn is freqctl_deps["apply_gpu_frequency"] for fn in received_kwargs)


def test_arc87_restaura_gpu_incluso_si_la_calibracion_falla(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    freqctl_deps, restore_calls, _ = _freqctl_fakes()
    gpu_restore_calls = []
    freqctl_deps["restore_gpu_state"] = lambda env: gpu_restore_calls.append(env) or True

    def failing_run_calibration(manifest, catalog, *, environment_profile, node_id, run_single, apply_frequency=None):
        raise RuntimeError("D03: fuera de rango")

    env_profile = SimpleNamespace(frequency_write_capable=False)
    with pytest.raises(RuntimeError, match="D03"):
        campaign.run_campaign(
            manifest, catalog, env_profile,
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), run_calibration=failing_run_calibration, **freqctl_deps,
        )

    # ARC-87: la restauracion de GPU debe intentarse siempre, igual que la
    # de CPU (CAM-07), incluso cuando la campana nunca llega a la matriz.
    assert restore_calls == ["SNAPSHOT"]
    assert gpu_restore_calls == [env_profile]


def test_arc87_frequency_restored_verified_es_falso_si_gpu_no_restaura(tmp_path):
    # ARC-87: una campana no esta "restaurada limpiamente" si CPU volvio a
    # su estado pero el reloj de GPU se quedo fijado -- el flag combinado
    # debe reflejar ambos, no solo CPU.
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, restore_calls, _ = _freqctl_fakes()
    freqctl_deps["restore_gpu_state"] = lambda env: False  # simula una falla real de nvidia-smi -rgc

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single([]), **calibration_deps, **freqctl_deps,
    )

    assert result.progress.frequency_restored_verified is False
