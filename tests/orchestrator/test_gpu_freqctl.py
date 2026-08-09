from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import gpu_freqctl


AVAILABLE_MHZ = [765, 900, 1050, 1200, 1350, 1410]


def _env(*, write_capable: bool, available_mhz: list[int] | None = AVAILABLE_MHZ) -> SimpleNamespace:
    return SimpleNamespace(
        gpu_frequency_write_capable=write_capable,
        gpu_available_clocks_mhz=available_mhz,
    )


def _fake_run_nvidia_smi(calls, *, returncode: int = 0, stderr: str = ""):
    def run_nvidia_smi(args, *, gpu_index):
        calls.append((tuple(args), gpu_index))
        return SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")
    return run_nvidia_smi


def test_arc87_apply_unavailable_no_toca_nvidia_smi_si_no_hay_permiso():
    calls = []
    env = _env(write_capable=False)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)

    applied = gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=_fake_run_nvidia_smi(calls))

    assert calls == []
    assert applied.strategy == gpu_freqctl.STRATEGY_UNAVAILABLE
    assert applied.write_skipped_reason == "unavailable"
    assert applied.requested_mhz is None
    assert applied.applied_mhz is None


def test_arc87_apply_native_governor_llama_reset():
    calls = []
    env = _env(write_capable=True)
    level = SimpleNamespace(id="REF", mode="native_governor", fraction=None)

    applied = gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=_fake_run_nvidia_smi(calls))

    assert calls == [(("-rgc",), 0)]
    assert applied.strategy == gpu_freqctl.STRATEGY_LOCKED_CLOCKS
    assert applied.requested_mhz is None
    assert applied.applied_mhz is None
    assert applied.write_skipped_reason is None


def test_arc87_apply_fixed_fija_el_reloj_mas_cercano_a_la_fraccion():
    calls = []
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)  # (765+1410)/2 = 1087.5 -> 1050 más cercano

    applied = gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=_fake_run_nvidia_smi(calls))

    assert calls == [(("-lgc", "1050,1050"), 0)]
    assert applied.strategy == gpu_freqctl.STRATEGY_LOCKED_CLOCKS
    assert applied.requested_mhz == 1050
    assert applied.applied_mhz == 1050


def test_arc87_apply_fixed_extremos_f0_f4():
    env = _env(write_capable=True)
    f0 = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)
    f4 = SimpleNamespace(id="F4", mode="fixed", fraction=0.0)

    applied_f0 = gpu_freqctl.apply_gpu_frequency(f0, env, run_nvidia_smi=_fake_run_nvidia_smi([]))
    applied_f4 = gpu_freqctl.apply_gpu_frequency(f4, env, run_nvidia_smi=_fake_run_nvidia_smi([]))

    assert applied_f0.applied_mhz == max(AVAILABLE_MHZ)
    assert applied_f4.applied_mhz == min(AVAILABLE_MHZ)


def test_arc87_apply_fixed_falla_ruidosamente_si_nvidia_smi_falla():
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)
    run_nvidia_smi = _fake_run_nvidia_smi([], returncode=1, stderr="Insufficient Permissions")

    with pytest.raises(gpu_freqctl.GpuFrequencyControlError, match="Insufficient Permissions"):
        gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=run_nvidia_smi)


def test_arc94_apply_fixed_relee_el_reloj_aplicado():
    """ARC-94: antes de este cambio, el exito se asumia solo por el
    returncode de -lgc, sin ninguna relectura independiente."""
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)  # target=1050

    applied = gpu_freqctl.apply_gpu_frequency(
        level, env, run_nvidia_smi=_fake_run_nvidia_smi([]),
        query_sm_clock_mhz=lambda gpu_index: 1050,
    )

    assert applied.observed_sm_mhz == 1050


def test_arc94_apply_fixed_reloj_ocioso_bajo_el_target_no_es_falla():
    """El techo fijado por -lgc no obliga a la GPU a correr a ese reloj si
    esta ociosa -- solo un reloj POR ENCIMA del techo es evidencia de que
    el candado no se aplico."""
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)  # target=1050

    applied = gpu_freqctl.apply_gpu_frequency(
        level, env, run_nvidia_smi=_fake_run_nvidia_smi([]),
        query_sm_clock_mhz=lambda gpu_index: 210,  # reloj ocioso tipico
    )

    assert applied.observed_sm_mhz == 210
    assert applied.applied_mhz == 1050


def test_arc94_apply_fixed_falla_si_la_relectura_supera_el_techo():
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)  # target=1050

    with pytest.raises(gpu_freqctl.GpuFrequencyControlError, match="supera el techo fijado"):
        gpu_freqctl.apply_gpu_frequency(
            level, env, run_nvidia_smi=_fake_run_nvidia_smi([]),
            query_sm_clock_mhz=lambda gpu_index: 1410,  # nunca deberia superar 1050
        )


def test_arc94_apply_fixed_relectura_no_disponible_no_bloquea():
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)

    applied = gpu_freqctl.apply_gpu_frequency(
        level, env, run_nvidia_smi=_fake_run_nvidia_smi([]),
        query_sm_clock_mhz=lambda gpu_index: None,
    )

    assert applied.observed_sm_mhz is None
    assert applied.applied_mhz == 1050


def test_arc94_apply_native_governor_tambien_relee():
    env = _env(write_capable=True)
    level = SimpleNamespace(id="REF", mode="native_governor", fraction=None)

    applied = gpu_freqctl.apply_gpu_frequency(
        level, env, run_nvidia_smi=_fake_run_nvidia_smi([]),
        query_sm_clock_mhz=lambda gpu_index: 1410,
    )

    assert applied.observed_sm_mhz == 1410


def test_arc87_apply_fixed_sin_relojes_disponibles_falla_ruidosamente():
    env = _env(write_capable=True, available_mhz=[])
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)

    with pytest.raises(gpu_freqctl.GpuFrequencyControlError, match="gpu_available_clocks_mhz"):
        gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=_fake_run_nvidia_smi([]))


def test_arc95_apply_native_governor_convierte_excepcion_de_subproceso():
    """ARC-95: run_nvidia_smi (un subprocess.run real) puede levantar
    OSError/FileNotFoundError (binario ausente) o TimeoutExpired sin que
    nada lo capturara -- ahora se normaliza a GpuFrequencyControlError,
    el tipo de error que este modulo ya documenta."""
    env = _env(write_capable=True)
    level = SimpleNamespace(id="REF", mode="native_governor", fraction=None)

    def run_nvidia_smi_que_falla(args, *, gpu_index):
        raise FileNotFoundError("nvidia-smi no encontrado")

    with pytest.raises(gpu_freqctl.GpuFrequencyControlError, match="no se pudo ejecutar"):
        gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=run_nvidia_smi_que_falla)


def test_arc95_apply_fixed_convierte_excepcion_de_subproceso():
    env = _env(write_capable=True)
    level = SimpleNamespace(id="F2", mode="fixed", fraction=0.5)

    def run_nvidia_smi_que_falla(args, *, gpu_index):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=30)

    with pytest.raises(gpu_freqctl.GpuFrequencyControlError, match="no se pudo ejecutar"):
        gpu_freqctl.apply_gpu_frequency(level, env, run_nvidia_smi=run_nvidia_smi_que_falla)


def test_arc95_restore_gpu_state_nunca_lanza_ante_excepcion_real():
    """ARC-95: el docstring prometia 'nunca lanza' pero nada capturaba una
    excepcion real del subproceso -- critico porque puede correr desde un
    manejador de SIGINT/SIGTERM sin segunda oportunidad."""
    env = _env(write_capable=True)

    def run_nvidia_smi_que_falla(args, *, gpu_index):
        raise FileNotFoundError("nvidia-smi no encontrado")

    assert gpu_freqctl.restore_gpu_state(env, run_nvidia_smi=run_nvidia_smi_que_falla) is False


def test_arc87_restore_no_op_si_no_hay_permiso():
    calls = []
    env = _env(write_capable=False)

    assert gpu_freqctl.restore_gpu_state(env, run_nvidia_smi=_fake_run_nvidia_smi(calls)) is True
    assert calls == []


def test_arc87_restore_llama_reset_y_devuelve_true_si_confirma():
    calls = []
    env = _env(write_capable=True)

    assert gpu_freqctl.restore_gpu_state(env, run_nvidia_smi=_fake_run_nvidia_smi(calls)) is True
    assert calls == [(("-rgc",), 0)]


def test_arc87_restore_nunca_lanza_si_nvidia_smi_falla():
    env = _env(write_capable=True)
    run_nvidia_smi = _fake_run_nvidia_smi([], returncode=1, stderr="GPU is lost")

    # Espejo de freqctl.restore_original_state: best-effort, nunca lanza --
    # puede llamarse desde un manejador de señal sin segunda oportunidad.
    assert gpu_freqctl.restore_gpu_state(env, run_nvidia_smi=run_nvidia_smi) is False
