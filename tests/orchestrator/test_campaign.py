import hashlib
import json
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


def _fake_run_single(calls):
    def run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                    environment_profile=None, node_id=None, apply_frequency=None, calibration_refs=None):
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

    def run_calibration(manifest, catalog, *, environment_profile, node_id, run_single):
        return SimpleNamespace(plausibility_check_passed=True, plausibility_message="")

    def build_node_profile(env, cpus, *, node_id, hostname=""):
        return SimpleNamespace(node_id=node_id, cache_line_size_bytes=64)

    def write_node_profile(profile, output_dir):
        return Path(output_dir) / "node_profile.json"

    def run_calibration_references(entry, manifest, kernel_ref, *, node_id, environment_profile, run_single):
        return SimpleNamespace(node_id=node_id, ipc_p95=1.0, accepted=True)

    def run_postprocess(run_dir, **kwargs):
        postprocess_calls.append((Path(run_dir), kwargs))
        return Path(run_dir) / "windows.csv"

    return dict(
        run_calibration=run_calibration, build_node_profile=build_node_profile,
        write_node_profile=write_node_profile, run_calibration_references=run_calibration_references,
        run_postprocess=run_postprocess,
    ), postprocess_calls


def _freqctl_fakes():
    restore_calls = []
    install_calls = []
    return dict(
        apply_frequency=lambda cpus, level_id, env: None,
        read_observed_frequency_khz=lambda env, cpu: None,
        snapshot_original_state=lambda cpus, env: "SNAPSHOT",
        restore_original_state=lambda state, env: restore_calls.append(state) or True,
        install_emergency_handlers=lambda restore: install_calls.append(restore),
    ), restore_calls, install_calls


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


def test_cam04_schedule_runs_empareja_baseline_y_telemetry(tmp_path):
    manifest = _manifest(tmp_path)
    combinations = campaign.build_matrix(manifest, seed=1)
    scheduled = campaign.schedule_runs(combinations)

    assert len(scheduled) == 2 * len(combinations)
    for i in range(0, len(scheduled), 2):
        assert scheduled[i].mode == "baseline"
        assert scheduled[i + 1].mode == "telemetry"
        assert scheduled[i].combination == scheduled[i + 1].combination


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


def test_frq03_frq10_frecuencia_solicitada_aplicada_y_observada_llegan_a_postprocess(tmp_path):
    manifest = _manifest(
        tmp_path, frequency_levels=(FrequencyLevel("F0", "fixed", 1.0),),
    )
    catalog = _catalog(tmp_path)
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    applied = SimpleNamespace(requested_khz=2261000, applied_khz=2261000, governor_applied="userspace")

    def run_single(entry, manifest, kernel_ref, freq_level_id, repetition_index, *,
                    environment_profile=None, node_id=None, apply_frequency=None, calibration_refs=None):
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


def test_cam03_reanudacion_salta_combinacion_ya_aceptada(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(True, None, "ok"), run_dir)

    calls: list[tuple[str, bool]] = []
    calibration_deps, postprocess_calls = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # Ni el baseline ni el telemetry de la combinacion ya aceptada se repiten.
    assert calls == []
    assert result.progress.accepted_run_ids == []
    assert result.progress.skipped_run_ids == ["camp01__npb_ep__REF__rep01"]  # MET-06
    assert postprocess_calls == []


def test_cam03_reanudacion_reintenta_combinacion_rechazada(tmp_path):
    manifest = _manifest(tmp_path)
    catalog = _catalog(tmp_path)
    run_dir = manifest.output_dir / "camp01__npb_ep__REF__rep01"
    run_dir.mkdir(parents=True)
    validation_module.write_verdict(validation_module.Verdict(False, "C02", "checksum discrepante"), run_dir)

    calls: list[tuple[str, bool]] = []
    calibration_deps, _ = _fake_calibration_deps()
    freqctl_deps, _, _ = _freqctl_fakes()

    result = campaign.run_campaign(
        manifest, catalog, SimpleNamespace(frequency_write_capable=False),
        node_id="felix-sc3", reference_kernel_ref="npb_ep",
        run_single=_fake_run_single(calls), **calibration_deps, **freqctl_deps,
    )

    # Un rechazo no es "hecho": se reintenta el par completo.
    assert calls == [("camp01__npb_ep__REF__rep01__baseline", False), ("camp01__npb_ep__REF__rep01", True)]
    assert result.progress.accepted_run_ids == ["camp01__npb_ep__REF__rep01"]


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

    def failing_run_calibration(manifest, catalog, *, environment_profile, node_id, run_single):
        raise RuntimeError("D03: fuera de rango")

    with pytest.raises(RuntimeError, match="D03"):
        campaign.run_campaign(
            manifest, catalog, SimpleNamespace(frequency_write_capable=False),
            node_id="felix-sc3", reference_kernel_ref="npb_ep",
            run_single=_fake_run_single([]), run_calibration=failing_run_calibration, **freqctl_deps,
        )

    assert restore_calls == ["SNAPSHOT"]
