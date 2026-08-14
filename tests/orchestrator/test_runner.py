from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import runner
from orchestrator.catalog import KernelEntry
from orchestrator.config import HarnessConfig

FAKE_LAUNCHER = Path(__file__).resolve().parent / "fixtures" / "fake_launcher.py"


def _make_entry(tmp_path: Path, *, success_check: dict | None = None, device: str = "cpu") -> KernelEntry:
    binary = tmp_path / "npb_ep.x"
    binary.write_bytes(b"#!/bin/sh\necho fake npb binary\n")
    binary.chmod(0o755)
    checksum = f"sha256:{hashlib.sha256(binary.read_bytes()).hexdigest()}"
    return KernelEntry(
        id="npb_ep",
        suite="npb",
        role="dataset",
        exec_path=str(binary),
        binary_checksum=checksum,
        phase_label_hint="compute_bound",
        size_variant="S",
        expected_runtime_seconds=1,
        warmup_seconds=0.0,
        success_check=success_check or {"type": "stdout_regex", "pattern": "VERIFICATION SUCCESSFUL"},
        estimated_memory_bytes=1024,
        device=device,
        operational_intensity_flops_per_byte=5.0 if device == "gpu" else None,
        gpu_precision="fp32" if device == "gpu" else None,
    )


def _make_manifest(tmp_path: Path, *, cgroup_path: str | None = None, perf_enabled: bool = True) -> SimpleNamespace:
    output_dir = tmp_path / "runs"
    output_dir.mkdir()
    return SimpleNamespace(
        campaign_id="camp01",
        output_dir=output_dir,
        interval_ns=1_000_000,
        cgroup_path=cgroup_path,
        perf_enabled=perf_enabled,
        cores=SimpleNamespace(delegated_cpus=(2, 3, 4, 5), collector_cpu=0, consumer_cpu=1),
        timeouts_seconds=SimpleNamespace(ready=5, run=5, shutdown=5),
        # ARC-78: apply_frequency necesita resolver el objeto completo a
        # partir de freq_level_id -- ambos ids que usan los tests de este
        # archivo ("REF"/"F0") deben existir aquí.
        frequency_levels=(
            SimpleNamespace(id="REF", mode="native_governor", fraction=None),
            SimpleNamespace(id="F0", mode="fixed", fraction=0.5),
        ),
    )


def _harness() -> HarnessConfig:
    return HarnessConfig(exec_flag="--exec", exec_args_flag="--exec-args", binary_path=str(FAKE_LAUNCHER))


def test_run01_comando_se_construye_desde_catalogo_y_manifest(tmp_path):
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path, cgroup_path="/delegated/cgroup")
    command = runner.build_command(entry, manifest, "run_x", _harness())

    assert command[0] == str(FAKE_LAUNCHER)
    assert "--exec" in command and entry.exec_path in command
    assert "--perf-cpus" in command
    assert command[command.index("--perf-cpus") + 1] == "2,3,4,5"
    assert command[command.index("--collector-cpu") + 1] == "0"
    assert command[command.index("--consumer-cpu") + 1] == "1"
    assert command[command.index("--output-dir") + 1] == str(manifest.output_dir)
    assert command[command.index("--run-id") + 1] == "run_x"
    assert command[command.index("--cgroup-path") + 1] == "/delegated/cgroup"


def test_run01_sin_cgroup_path_no_agrega_la_bandera(tmp_path):
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path, cgroup_path=None)
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--cgroup-path" not in command


def test_run02_run_id_determinista():
    run_id = runner.build_run_id("camp01", "npb_ep", "REF", 3)
    assert run_id == "camp01__npb_ep__REF__rep03"


def test_arc70_build_command_agrega_enable_gpu_si_device_gpu(tmp_path):
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--enable-gpu" in command


def test_arc70_build_command_sin_enable_gpu_si_device_cpu(tmp_path):
    entry = _make_entry(tmp_path, device="cpu")
    manifest = _make_manifest(tmp_path)
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--enable-gpu" not in command


def test_arc70_build_command_agrega_gpu_interval_ns_si_esta_en_el_manifiesto(tmp_path):
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    manifest.gpu_interval_ns = 50_000_000
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert command[command.index("--gpu-interval-ns") + 1] == "50000000"


def test_arc116_build_command_agrega_enable_uncore_si_esta_habilitado(tmp_path):
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    manifest.uncore = {"enabled": True}
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--enable-uncore" in command


def test_arc116_build_command_sin_uncore_por_defecto(tmp_path):
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--enable-uncore" not in command


def test_arc131_build_command_agrega_uncore_pin_cpu_fuera_de_los_reservados(tmp_path):
    # ARC-131: delegated_cpus=(2,3,4,5), collector_cpu=0, consumer_cpu=1 en
    # _make_manifest() -- max(reservados)=5, el pin debe caer en 6, el
    # primer CPU logico libre despues de todos los reservados.
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    manifest.uncore = {"enabled": True}
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert command[command.index("--uncore-pin-cpu") + 1] == "6"


def test_arc131_build_command_sin_uncore_no_agrega_pin_cpu(tmp_path):
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--uncore-pin-cpu" not in command


def test_arc135_build_command_agrega_cpu_freq_sysfs_path(tmp_path):
    # ARC-135: reemplaza el viejo read_observed_frequency_khz() post-hoc
    # (una sola lectura de Python DESPUES de que el proceso ya termino,
    # encontrado sin correlacion real con el nivel solicitado en datos de
    # campana real) por muestreo real del colector C++, en el mismo tick
    # que los contadores de PMU. delegated_cpus=(2,3,4,5) en _make_manifest()
    # -- debe resolver la ruta para el CPU 2 (el primero de la lista).
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    cpufreq = tmp_path / "cpu2" / "cpufreq"
    cpufreq.mkdir(parents=True)
    governor_path = cpufreq / "scaling_governor"
    governor_path.write_text("performance")
    env_profile = SimpleNamespace(
        frequency_control_paths={2: {"scaling_governor": str(governor_path)}},
    )
    command = runner.build_command(entry, manifest, "run_x", _harness(), environment_profile=env_profile)
    assert command[command.index("--cpu-freq-sysfs-path") + 1] == str(cpufreq / "scaling_cur_freq")


def test_arc135_build_command_sin_environment_profile_no_agrega_freq_path(tmp_path):
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    command = runner.build_command(entry, manifest, "run_x", _harness())
    assert "--cpu-freq-sysfs-path" not in command


def test_arc70_run_single_setea_ld_preload_y_library_path_para_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    fake_shim = tmp_path / "fake_shim.so"
    fake_shim.write_bytes(b"")
    fake_cuda_lib = tmp_path / "cuda_lib"
    fake_cuda_lib.mkdir()
    monkeypatch.setattr(runner, "compiled_blocking_sync_shim", lambda: fake_shim)
    monkeypatch.setattr(runner, "cuda_lib_dirs", lambda: [fake_cuda_lib])

    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    assert result.success is True
    assert result.metadata["observed_enable_gpu"] is True
    assert result.metadata["observed_ld_preload"].startswith(str(fake_shim))
    assert result.metadata["observed_ld_library_path"].startswith(str(fake_cuda_lib))


def test_arc70_run_single_gpu_sin_shim_disponible_no_falla(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    monkeypatch.setattr(runner, "compiled_blocking_sync_shim", lambda: None)
    monkeypatch.setattr(runner, "cuda_lib_dirs", lambda: [])

    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)

    with caplog.at_level("WARNING"):
        result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    # ARC-70: sin CUDA en este nodo, la corrida sigue -- degradación conocida
    # (spin en vez de bloqueo real), nunca un fallo duro.
    assert result.success is True
    assert result.metadata["observed_ld_preload"] == ""
    assert "ARC-70" in caplog.text


def test_run05_run06_run07_corrida_exitosa(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(
        entry, manifest, "npb_ep", "REF", 1, harness=_harness(), node_id="felix-sc3"
    )

    assert result.success is True
    assert result.timed_out is False
    assert result.exit_code == 0
    # RUN-07: stdout/stderr completos en output_dir/<run_id>/
    assert result.stdout_path.read_text().strip() == "VERIFICATION SUCCESSFUL"
    assert result.stdout_path.parent == result.run_dir
    assert result.run_dir == manifest.output_dir / result.run_id
    # RUN-06: metadata fusionada (launcher + orquestador), sin pisar campos.
    assert result.metadata["samples_collected"] == 0
    assert result.metadata["perf_attach_mode"] == "pid_inherit"
    assert result.metadata["campaign_id"] == "camp01"
    assert result.metadata["node_id"] == "felix-sc3"
    assert result.metadata["binary_checksum"] == entry.binary_checksum


def test_arc94_metadata_fusionada_se_persiste_en_disco(tmp_path, monkeypatch):
    """ARC-94 (segunda ronda): RunResult.metadata (RUN-06) solo vivia en
    memoria -- metadata.json en disco se quedaba con lo que el launcher
    escribio (samples_collected, push_retries), nunca con campaign_id/
    kernel_ref/checksum que el orquestador agrega despues. Confirmado que
    el 100% de los metadata.json aceptados en produccion carecian de esos
    campos."""
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(
        entry, manifest, "npb_ep", "REF", 1, harness=_harness(), node_id="felix-sc3"
    )

    on_disk = json.loads((result.run_dir / "metadata.json").read_text())
    assert on_disk["campaign_id"] == "camp01"
    assert on_disk["kernel_ref"] == "npb_ep"
    assert on_disk["node_id"] == "felix-sc3"
    assert on_disk["binary_checksum"] == entry.binary_checksum
    assert on_disk["samples_collected"] == 0  # del launcher, se conserva
    assert on_disk["command"] == list(result.command)


def test_arc94_run_id_explicito_anula_el_derivado(tmp_path, monkeypatch):
    """ARC-94: campaign.py necesita poder pasar el run_id con sufijo
    __baseline del par baseline/telemetry -- antes, run_single() siempre
    reconstruía el id 'plano' internamente vía build_run_id(), sin importar
    qué id el llamador tuviera en mente, y el par colisionaba en el mismo
    directorio."""
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    explicit_run_id = "camp01__npb_ep__REF__rep01__baseline"
    result = runner.run_single(
        entry, manifest, "npb_ep", "REF", 1, harness=_harness(), run_id=explicit_run_id,
    )

    assert result.run_id == explicit_run_id
    assert result.run_dir == manifest.output_dir / explicit_run_id
    assert result.run_dir.exists()


def test_arc94_run_id_por_defecto_sigue_siendo_build_run_id(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    assert result.run_id == runner.build_run_id("camp01", "npb_ep", "REF", 1)


def test_run05_success_check_falla_si_falta_el_patron(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LAUNCHER_BEHAVIOR", "fail")
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    assert result.exit_code == 1
    assert result.success is False
    assert "simulated failure" in result.stderr_path.read_text()


def test_run06_colision_de_metadata_lanza_error(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    with pytest.raises(ValueError, match="RUN-06"):
        runner.run_single(
            entry,
            manifest,
            "npb_ep",
            "REF",
            1,
            harness=_harness(),
            # "campaign_id" ya lo agrega runner.py: cualquier referencia de
            # calibración que repita una clave debe fallar en vez de pisarla.
            calibration_refs={"campaign_id": "otra_campana"},
        )


def test_run08_no_invoca_apply_frequency_si_no_hay_permiso(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False)
    calls = []

    runner.run_single(
        entry,
        manifest,
        "npb_ep",
        "REF",
        1,
        harness=_harness(),
        environment_profile=env_profile,
        apply_frequency=lambda cpus, level, env: calls.append((cpus, level, env)),
    )

    assert calls == []


def test_run08_invoca_apply_frequency_si_hay_permiso(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=True)
    calls = []

    runner.run_single(
        entry,
        manifest,
        "npb_ep",
        "F0",
        1,
        harness=_harness(),
        environment_profile=env_profile,
        apply_frequency=lambda cpus, level, env: calls.append((cpus, level, env)),
    )

    # ARC-78: debe recibir el objeto FrequencyLevel completo (con .mode/
    # .fraction), no solo el string "F0" -- freqctl.apply_frequency() los
    # necesita para decidir la estrategia correcta.
    assert len(calls) == 1
    cpus, level, env = calls[0]
    assert cpus == (2, 3, 4, 5)
    assert level.id == "F0"
    assert level.mode == "fixed"
    assert env is env_profile


def test_run09_nivel_fixed_sin_permiso_falla_en_vez_de_correr_en_nativo(tmp_path, monkeypatch):
    # ARC-101: antes de este fix, un nivel fixed (F0-F4) sin
    # frequency_write_capable simplemente se omitia (logger.debug) y la
    # corrida seguia a la frecuencia nativa, pero quedaba etiquetada con el
    # freq_level_id solicitado -- confirmado que esto paso de verdad en la
    # campana pacca_ref_full_arc97_20260809 (corridas F0-F4 aceptadas con
    # freq_khz_observed~800MHz y freq_khz_requested/applied vacios). Un
    # nivel fixed sin capacidad real debe abortar la corrida, nunca
    # degradarse en silencio a REF con una etiqueta enganosa.
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False)

    with pytest.raises(RuntimeError, match="RUN-09"):
        runner.run_single(
            entry,
            manifest,
            "npb_ep",
            "F0",
            1,
            harness=_harness(),
            environment_profile=env_profile,
            apply_frequency=lambda cpus, level, env: (_ for _ in ()).throw(AssertionError("no debe llamarse")),
        )


def test_frq03_frecuencia_solicitada_y_aplicada_llegan_a_la_metadata_de_la_corrida(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=True)
    applied = SimpleNamespace(
        level_id="F0", strategy="discrete_bounds", requested_khz=2261000, applied_khz=2261000,
        per_cpu_applied_khz={2: 2261000}, governor_applied="userspace", write_skipped_reason=None,
    )

    result = runner.run_single(
        entry, manifest, "npb_ep", "F0", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_frequency=lambda cpus, level, env: applied,
    )

    # FRQ-03: ni "solo lo solicitado" ni "solo lo aplicado" -- ambos, y
    # nunca se pierden entre freqctl y la metadata.json persistida.
    assert result.applied_frequency is applied
    assert result.metadata["freq_khz_requested"] == 2261000
    assert result.metadata["freq_khz_applied"] == 2261000
    assert result.metadata["freq_governor_applied"] == "userspace"


def test_frq03_sin_apply_frequency_no_agrega_campos_de_frecuencia(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path)
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    assert result.applied_frequency is None
    assert "freq_khz_requested" not in result.metadata


def _leftover_hang_sleep_pids() -> list[int]:
    # The fake launcher's "hang" behavior spawns exactly `sleep 300`; matching
    # on the full cmdline keeps this from tripping on unrelated sleeps.
    return [
        proc.pid
        for proc in psutil.process_iter(["cmdline"])
        if (proc.info["cmdline"] or []) == ["sleep", "300"]
    ]


def test_run03_run04_timeout_mata_grupo_completo_sin_dejar_procesos(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LAUNCHER_BEHAVIOR", "hang")
    entry = _make_entry(tmp_path)
    # expected_runtime_seconds x SAFETY_MARGIN debe expirar mucho antes de
    # que el fake launcher despierte de su sleep(300).
    entry.expected_runtime_seconds = 1
    manifest = _make_manifest(tmp_path)

    assert _leftover_hang_sleep_pids() == []

    result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    assert result.timed_out is True
    assert result.success is False
    # RUN-04: ni el fake launcher ni el "sleep 300" que lanzó (grandchild,
    # cubierto por el process group) deben seguir vivos.
    assert _leftover_hang_sleep_pids() == []


def test_arc87_no_invoca_apply_gpu_frequency_para_kernel_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="cpu")
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=True)
    calls = []

    result = runner.run_single(
        entry, manifest, "npb_ep", "F0", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: calls.append((level, env)),
    )

    # ARC-87: un kernel de CPU nunca toca el reloj de GPU, sin importar si
    # el permiso está disponible -- los dos ejes se gatean por
    # entry.device, no solo por la capacidad de escritura.
    assert calls == []
    assert result.applied_gpu_frequency is None


def test_arc87_gpu_ref_sin_permiso_no_invoca_apply_gpu_frequency(tmp_path, monkeypatch):
    # Nivel REF (native_governor): no requiere escritura real, sigue
    # omitiendose en silencio sin importar el permiso.
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=False)
    calls = []

    runner.run_single(
        entry, manifest, "npb_ep", "REF", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: calls.append((level, env)),
    )

    assert calls == []


def test_run09_nivel_gpu_fixed_sin_permiso_falla_en_vez_de_correr_en_nativo(tmp_path, monkeypatch):
    # ARC-101: mismo criterio que el caso de CPU -- un nivel fixed (F0-F4)
    # en el eje GPU sin gpu_frequency_write_capable debe abortar la
    # corrida, nunca degradarse en silencio al reloj nativo con una
    # etiqueta enganosa.
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=False)

    with pytest.raises(RuntimeError, match="RUN-09"):
        runner.run_single(
            entry, manifest, "npb_ep", "F0", 1,
            harness=_harness(), environment_profile=env_profile,
            apply_gpu_frequency=lambda level, env: (_ for _ in ()).throw(AssertionError("no debe llamarse")),
        )


def test_arc87_invoca_apply_gpu_frequency_para_kernel_gpu_con_permiso(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=True)
    calls = []

    runner.run_single(
        entry, manifest, "npb_ep", "F0", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: calls.append((level, env)),
    )

    # ARC-87: recibe el objeto FrequencyLevel completo (mismo criterio que
    # FRQ-03/ARC-78 para CPU), no solo el string "F0".
    assert len(calls) == 1
    level, env = calls[0]
    assert level.id == "F0"
    assert level.mode == "fixed"
    assert env is env_profile


def test_arc129_gpu_freq_level_id_resuelve_contra_gpu_frequency_levels_no_frequency_levels(tmp_path, monkeypatch):
    # ARC-129: con gpu_freq_level_id explícito, el eje de GPU debe resolver
    # contra manifest.gpu_frequency_levels (una lista de ids totalmente
    # distinta de frequency_levels, para que la prueba no pueda pasar "por
    # coincidencia" si ambos ejes compartieran el mismo id de string).
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    manifest.gpu_frequency_levels = (
        SimpleNamespace(id="GREF", mode="native_governor", fraction=None),
        SimpleNamespace(id="GF0", mode="fixed", fraction=0.25),
    )
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=True)
    calls = []

    runner.run_single(
        entry, manifest, "npb_ep", "F0", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: calls.append((level, env)),
        gpu_freq_level_id="GF0",
    )

    assert len(calls) == 1
    level, _env = calls[0]
    # El nivel de GPU aplicado es "GF0" (0.25), no "F0" (0.5, del eje de
    # CPU) -- confirma que los dos ejes están genuinamente desacoplados.
    assert level.id == "GF0"
    assert level.fraction == 0.25


def test_arc129_gpu_freq_level_id_ausente_reusa_frequency_levels_como_siempre(tmp_path, monkeypatch):
    # ARC-129: sin gpu_freq_level_id (None, el default -- toda llamada
    # anterior a este cambio), el eje de GPU sigue acoplado a freq_level_id
    # contra manifest.frequency_levels, exactamente como antes.
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=True)
    calls = []

    runner.run_single(
        entry, manifest, "npb_ep", "F0", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: calls.append((level, env)),
    )

    assert len(calls) == 1
    level, _env = calls[0]
    assert level.id == "F0"
    assert level.fraction == 0.5


def test_arc87_frecuencia_de_gpu_llega_a_la_metadata_de_la_corrida(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)
    env_profile = SimpleNamespace(frequency_write_capable=False, gpu_frequency_write_capable=True)
    applied = SimpleNamespace(
        level_id="F0", strategy="locked_clocks", requested_mhz=1050, applied_mhz=1050,
        write_skipped_reason=None,
    )

    result = runner.run_single(
        entry, manifest, "npb_ep", "F0", 1,
        harness=_harness(), environment_profile=env_profile,
        apply_gpu_frequency=lambda level, env: applied,
    )

    assert result.applied_gpu_frequency is applied
    assert result.metadata["gpu_freq_mhz_requested"] == 1050
    assert result.metadata["gpu_freq_mhz_applied"] == 1050


def test_arc87_sin_apply_gpu_frequency_no_agrega_campos_de_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_LAUNCHER_BEHAVIOR", raising=False)
    entry = _make_entry(tmp_path, device="gpu")
    manifest = _make_manifest(tmp_path)

    result = runner.run_single(entry, manifest, "npb_ep", "REF", 1, harness=_harness())

    assert result.applied_gpu_frequency is None
    assert "gpu_freq_mhz_requested" not in result.metadata
