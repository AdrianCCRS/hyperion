from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import freqctl


AVAILABLE_KHZ = [1064000, 1330000, 1596000, 1862000, 2128000, 2261000]


def _write_cpu(tmp_path: Path, cpu: int, *, governor: str = "performance",
               min_khz: int = 1064000, max_khz: int = 2261000,
               setspeed_khz: int | None = None) -> dict[str, str]:
    cpufreq = tmp_path / f"cpu{cpu}" / "cpufreq"
    cpufreq.mkdir(parents=True)
    (cpufreq / "scaling_governor").write_text(governor)
    (cpufreq / "scaling_min_freq").write_text(str(min_khz))
    (cpufreq / "scaling_max_freq").write_text(str(max_khz))
    (cpufreq / "scaling_cur_freq").write_text(str(max_khz))
    if setspeed_khz is not None:
        (cpufreq / "scaling_setspeed").write_text(str(setspeed_khz))
    return {
        "scaling_governor": str(cpufreq / "scaling_governor"),
        "scaling_min_freq": str(cpufreq / "scaling_min_freq"),
        "scaling_max_freq": str(cpufreq / "scaling_max_freq"),
    }


def _env(control_paths: dict, *, write_capable: bool, strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        frequency_write_capable=write_capable,
        frequency_control_strategy=strategy,
        frequency_control_paths=control_paths,
        available_frequencies_khz=AVAILABLE_KHZ,
    )


def test_frq01_snapshot_es_de_solo_lectura(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance")}
    env = _env(paths, write_capable=True, strategy="discrete_bounds")

    snapshot = freqctl.snapshot_original_state([0], env)

    assert snapshot.per_cpu[0].governor == "performance"
    assert snapshot.per_cpu[0].min_freq_khz == 1064000
    assert snapshot.per_cpu[0].max_freq_khz == 2261000
    # No write happened: the file must still read exactly what we wrote.
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "performance"


def test_frq06_write_capable_false_no_escribe_nada(tmp_path):
    original_governor_text = "performance"
    paths = {0: _write_cpu(tmp_path, 0, governor=original_governor_text)}
    env = _env(paths, write_capable=False, strategy="discrete_bounds")
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    result = freqctl.apply_frequency([0], level, env)

    assert result.write_skipped_reason == "unavailable"
    assert result.requested_khz is None
    assert result.applied_khz is None
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == original_governor_text
    assert not (tmp_path / "cpu0/cpufreq/scaling_setspeed").exists()


def test_frq02_frq03_apply_discrete_bounds_escribe_y_verifica(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance", setspeed_khz=0)}
    env = _env(paths, write_capable=True, strategy="discrete_bounds")
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    result = freqctl.apply_frequency([0], level, env)

    assert result.requested_khz == 2261000  # fraction=1.0 -> el mayor disponible
    assert result.applied_khz == 2261000
    assert result.per_cpu_applied_khz[0] == 2261000
    assert result.governor_applied == "userspace"
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "userspace"
    assert (tmp_path / "cpu0/cpufreq/scaling_setspeed").read_text() == "2261000"


def test_frq02_apply_discrete_bounds_elige_el_mas_cercano(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, setspeed_khz=0)}
    env = _env(paths, write_capable=True, strategy="discrete_bounds")
    # fraction=0.5 -> punto medio (1064000+2261000)/2=1662500, mas cercano es 1596000.
    level = SimpleNamespace(id="MID", mode="fixed", fraction=0.5)

    result = freqctl.apply_frequency([0], level, env)

    assert result.requested_khz == 1596000
    assert result.applied_khz == 1596000


def test_frq02_apply_bounded_range_fija_min_igual_a_max(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="powersave")}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    result = freqctl.apply_frequency([0], level, env)

    assert result.applied_khz == 2261000
    assert (tmp_path / "cpu0/cpufreq/scaling_min_freq").read_text() == "2261000"
    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"


def test_frq02_apply_falla_ruidosamente_si_la_relectura_no_coincide(tmp_path, monkeypatch):
    paths = {0: _write_cpu(tmp_path, 0, setspeed_khz=0)}
    env = _env(paths, write_capable=True, strategy="discrete_bounds")
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    # Simula un sysfs que ignora la escritura silenciosamente (kernel real
    # que redondea a otro valor, o hardware defectuoso): freqctl nunca debe
    # asumir éxito.
    real_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self.name == "scaling_setspeed":
            return "0"
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    with pytest.raises(freqctl.FrequencyControlError):
        freqctl.apply_frequency([0], level, env)


def test_frq09_solo_toca_los_cpus_delegados(tmp_path):
    paths = {
        0: _write_cpu(tmp_path, 0, setspeed_khz=0),
        1: _write_cpu(tmp_path, 1, governor="performance"),
    }
    env = _env(paths, write_capable=True, strategy="discrete_bounds")
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    freqctl.apply_frequency([0], level, env)

    # cpu1 nunca fue pasado como delegado: debe seguir intacto.
    assert (tmp_path / "cpu1/cpufreq/scaling_governor").read_text() == "performance"
    assert not (tmp_path / "cpu1/cpufreq/scaling_setspeed").exists()


def test_frq04_restore_es_idempotente_y_verificado(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance", setspeed_khz=None)}
    env = _env(paths, write_capable=True, strategy="discrete_bounds")
    original = freqctl.snapshot_original_state([0], env)
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    freqctl.apply_frequency([0], level, env)
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "userspace"

    assert freqctl.restore_original_state(original, env) is True
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "performance"
    assert (tmp_path / "cpu0/cpufreq/scaling_min_freq").read_text() == "1064000"
    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"

    # Idempotent: calling it again with everything already restored is still
    # a verified success, not a no-op that skips verification.
    assert freqctl.restore_original_state(original, env) is True


def test_frq04_restore_no_escribe_si_write_capable_es_false(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance")}
    env = _env(paths, write_capable=False, strategy="discrete_bounds")
    original = freqctl.OriginalState(
        cpus=(0,),
        strategy="discrete_bounds",
        per_cpu={0: freqctl.CpuOriginalState("performance", 1064000, 2261000, None)},
    )

    assert freqctl.restore_original_state(original, env) is True
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "performance"


def test_frq05_install_emergency_handlers_registra_las_tres_rutas(monkeypatch):
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        calls = []
        killed = []
        monkeypatch.setattr(freqctl.os, "kill", lambda pid, sig: killed.append((pid, sig)))
        monkeypatch.setattr(freqctl.atexit, "register", lambda fn: calls.append("atexit_registered"))

        freqctl.install_emergency_handlers(lambda: calls.append("restored") or True)

        assert calls == ["atexit_registered"]
        sigint_handler = signal.getsignal(signal.SIGINT)
        sigterm_handler = signal.getsignal(signal.SIGTERM)
        assert sigint_handler not in (signal.SIG_DFL, signal.SIG_IGN, original_sigint)
        assert sigterm_handler not in (signal.SIG_DFL, signal.SIG_IGN, original_sigterm)

        sigterm_handler(signal.SIGTERM, None)
        assert calls == ["atexit_registered", "restored"]
        assert killed  # el proceso real se re-arma para terminar, no se traga la señal
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def test_frq10_read_observed_frequency_khz(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, max_khz=1862000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    (tmp_path / "cpu0/cpufreq/scaling_cur_freq").write_text("1862000")

    assert freqctl.read_observed_frequency_khz(env, 0) == 1862000
    assert freqctl.read_observed_frequency_khz(env, 99) is None
