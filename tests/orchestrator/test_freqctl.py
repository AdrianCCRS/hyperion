from pathlib import Path
import os
import signal
import subprocess
import sys
import textwrap
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


def test_frq02_apply_bounded_range_transicion_descendente(tmp_path, monkeypatch):
    """ARC-94: el kernel real exige min<=max en CADA escritura individual,
    no solo al final. Partiendo de un nivel ya fijado arriba (min=max=F0),
    bajar a un nivel menor con el orden viejo (max,min,max) intenta
    max=target por debajo del min vigente -> EINVAL. Simula esa restricción
    del kernel para confirmar que el nuevo orden (min primero cuando el
    target queda debajo del min actual) nunca la viola."""
    paths = {0: _write_cpu(tmp_path, 0, governor="powersave", min_khz=2261000, max_khz=2261000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    # fraction=0.5 sobre AVAILABLE_KHZ -> target=1662500, redondeado al paso
    # de 100MHz mas cercano (2026-08-28, ver _target_khz) a 1700000, por
    # debajo del min vigente (2261000): la transicion descendente que rompia
    # el orden viejo.
    level = SimpleNamespace(id="F_MID", mode="fixed", fraction=0.5)

    min_path = tmp_path / "cpu0/cpufreq/scaling_min_freq"
    max_path = tmp_path / "cpu0/cpufreq/scaling_max_freq"
    original_write_text = freqctl._write_text

    def kernel_como_de_verdad(path: Path, value: str) -> None:
        # Simula min<=max como invariante de kernel, verificado en cada
        # escritura individual (no solo al final de la secuencia).
        nuevo = int(value)
        if path == max_path:
            min_vigente = int(min_path.read_text())
            if nuevo < min_vigente:
                raise OSError(22, "Invalid argument")  # EINVAL real del kernel
        elif path == min_path:
            max_vigente = int(max_path.read_text())
            if nuevo > max_vigente:
                raise OSError(22, "Invalid argument")
        original_write_text(path, value)

    monkeypatch.setattr(freqctl, "_write_text", kernel_como_de_verdad)

    result = freqctl.apply_frequency([0], level, env)

    assert result.applied_khz == 1700000
    assert min_path.read_text() == "1700000"
    assert max_path.read_text() == "1700000"


def test_frq_native_governor_restaura_min_max_no_solo_el_governor(tmp_path):
    """ARC-94: si REF (native_governor) se aplica DESPUES de un nivel fixed
    (bounded_range) en la misma corrida, restaurar solo el string del
    governor deja min/max pinneados en el ultimo nivel medido -- REF ya no
    seria "frecuencia nativa/libre" de verdad."""
    paths = {0: _write_cpu(tmp_path, 0, governor="powersave", min_khz=1064000, max_khz=2261000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")

    original = freqctl.snapshot_original_state([0], env)
    assert original.per_cpu[0].min_freq_khz == 1064000
    assert original.per_cpu[0].max_freq_khz == 2261000

    # Simula un nivel fixed ya aplicado antes de REF (pinneado en el tope).
    fixed_level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)
    freqctl.apply_frequency([0], fixed_level, env, original=original)
    assert (tmp_path / "cpu0/cpufreq/scaling_min_freq").read_text() == "2261000"
    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"

    ref_level = SimpleNamespace(id="REF", mode="native_governor")
    freqctl.apply_frequency([0], ref_level, env, original=original)

    assert (tmp_path / "cpu0/cpufreq/scaling_min_freq").read_text() == "1064000"
    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"
    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "powersave"


def test_arc94_native_governor_bounded_range_no_escribe_governor(tmp_path):
    """ARC-94 (segunda ronda): restaurar REF bajo bounded_range no debe
    necesitar permiso de escritura sobre scaling_governor -- P1 (el
    permiso real solicitado) solo cubre scaling_min_freq/max_freq.
    Simulado con el archivo de governor de solo lectura: si el código
    intentara escribirlo, esto lanzaría PermissionError."""
    import os as os_module
    paths = {0: _write_cpu(tmp_path, 0, governor="powersave", min_khz=1064000, max_khz=2261000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    original = freqctl.snapshot_original_state([0], env)

    governor_path = tmp_path / "cpu0/cpufreq/scaling_governor"
    os_module.chmod(governor_path, 0o444)

    fixed_level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)
    freqctl.apply_frequency([0], fixed_level, env, original=original)

    ref_level = SimpleNamespace(id="REF", mode="native_governor")
    result = freqctl.apply_frequency([0], ref_level, env, original=original)  # no debe lanzar PermissionError

    assert (tmp_path / "cpu0/cpufreq/scaling_min_freq").read_text() == "1064000"
    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"
    assert result.governor_applied == "powersave"


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


def test_arc163_apply_frequency_extiende_a_los_hermanos_smt(tmp_path):
    # ARC-162/163: cpu16 es hermano SMT de cpu0 (comparten núcleo físico) --
    # apply_frequency() debe restringirlo también, aunque solo se pase [0]
    # como delegado, para que el candado del delegado no quede sin efecto
    # bajo carga real por culpa del hermano libre.
    paths = {
        0: _write_cpu(tmp_path, 0, governor="powersave"),
        16: _write_cpu(tmp_path, 16, governor="powersave"),
    }
    env = _env(paths, write_capable=True, strategy="bounded_range")
    env.smt_siblings = {0: [0, 16]}
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    result = freqctl.apply_frequency([0], level, env)

    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"
    assert (tmp_path / "cpu16/cpufreq/scaling_max_freq").read_text() == "2261000"
    assert 16 in result.per_cpu_applied_khz


def test_arc163_apply_frequency_sin_smt_siblings_no_toca_otros_cpus(tmp_path):
    # Comportamiento anterior a ARC-163 preservado: sin env.smt_siblings
    # declarado (mismo default que todo fixture/entorno de prueba
    # existente), apply_frequency() sigue tocando solo los CPUs pasados.
    paths = {
        0: _write_cpu(tmp_path, 0, governor="powersave"),
        16: _write_cpu(tmp_path, 16, governor="powersave"),
    }
    env = _env(paths, write_capable=True, strategy="bounded_range")
    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)

    freqctl.apply_frequency([0], level, env)

    assert (tmp_path / "cpu16/cpufreq/scaling_max_freq").read_text() == "2261000"  # valor original de _write_cpu
    assert (tmp_path / "cpu16/cpufreq/scaling_governor").read_text() == "powersave"


def test_arc163_snapshot_y_restore_incluyen_los_hermanos_smt(tmp_path):
    # ARC-163: si apply_frequency() va a escribir en los hermanos, el
    # snapshot tomado al inicio de la campaña debe conocer su estado
    # previo -- de lo contrario restore_original_state() los dejaría
    # modificados permanentemente al terminar (el mismo tipo de fuga de
    # estado residual que motivó esta corrección, ver ARC-162).
    paths = {
        0: _write_cpu(tmp_path, 0, min_khz=1064000, max_khz=2261000),
        16: _write_cpu(tmp_path, 16, min_khz=1064000, max_khz=2261000),
    }
    env = _env(paths, write_capable=True, strategy="bounded_range")
    env.smt_siblings = {0: [0, 16]}

    original = freqctl.snapshot_original_state([0], env)
    assert 16 in original.per_cpu

    level = SimpleNamespace(id="F4", mode="fixed", fraction=0.0)
    freqctl.apply_frequency([0], level, env)
    assert (tmp_path / "cpu16/cpufreq/scaling_max_freq").read_text() == "1064000"

    assert freqctl.restore_original_state(original, env) is True
    assert (tmp_path / "cpu16/cpufreq/scaling_min_freq").read_text() == "1064000"
    assert (tmp_path / "cpu16/cpufreq/scaling_max_freq").read_text() == "2261000"


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


def test_arc94_restore_continua_con_los_demas_cpus_si_uno_falla(tmp_path):
    """ARC-94 (segunda ronda): el propio docstring promete "always attempts
    every CPU, even if an earlier one failed" -- antes de este cambio, una
    excepción sin capturar (p.ej. PermissionError real) en un CPU
    interrumpía el bucle antes de restaurar los siguientes. Critico porque
    puede correr desde un manejador de SIGINT/SIGTERM sin segunda
    oportunidad."""
    paths = {
        0: _write_cpu(tmp_path, 0, governor="performance", min_khz=1064000, max_khz=2261000),
        1: _write_cpu(tmp_path, 1, governor="performance", min_khz=1064000, max_khz=2261000),
    }
    env = _env(paths, write_capable=True, strategy="bounded_range")
    original = freqctl.snapshot_original_state([0, 1], env)

    # cpu0 queda sin permiso de escritura sobre min_freq -- simula un
    # permiso real mas estrecho de lo esperado en un solo core.
    os.chmod(tmp_path / "cpu0/cpufreq/scaling_min_freq", 0o444)

    result = freqctl.restore_original_state(original, env)

    assert result is False  # cpu0 genuinamente fallo
    # cpu1 SI se restauro pese al fallo de cpu0.
    assert (tmp_path / "cpu1/cpufreq/scaling_min_freq").read_text() == "1064000"
    assert (tmp_path / "cpu1/cpufreq/scaling_max_freq").read_text() == "2261000"


def test_arc95_restore_bounded_range_no_exige_governor_escribible(tmp_path):
    """ARC-95: restore_original_state() tenia el mismo bug que
    _apply_native_governor ya corrigio en ARC-94 -- reescribia el governor
    incondicionalmente, exigiendo un permiso que P1 (solo min/max) no
    concede para bounded_range."""
    paths = {0: _write_cpu(tmp_path, 0, governor="powersave", min_khz=1064000, max_khz=2261000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    original = freqctl.snapshot_original_state([0], env)

    governor_path = tmp_path / "cpu0/cpufreq/scaling_governor"
    os.chmod(governor_path, 0o444)

    level = SimpleNamespace(id="F0", mode="fixed", fraction=1.0)
    freqctl.apply_frequency([0], level, env, original=original)

    result = freqctl.restore_original_state(original, env)

    assert result is True  # no debio fallar por no poder escribir el governor
    assert (tmp_path / "cpu0/cpufreq/scaling_min_freq").read_text() == "1064000"
    assert (tmp_path / "cpu0/cpufreq/scaling_max_freq").read_text() == "2261000"


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


def test_frq05_sigint_heredada_como_ignorada_restaura_y_termina(tmp_path):
    """Regresión de la prueba de caos real en paccaA100.

    Una shell no interactiva inicia procesos en background con SIGINT
    ignorada. El manejador anterior restauraba esa disposición antes de
    reenviar la señal: la restauración ocurría, pero SIGINT se tragaba y la
    campaña seguía hasta terminar con rc=0. El proceso debe restaurar y morir
    por SIGINT incluso bajo esa herencia exacta.
    """
    marker = tmp_path / "restored"
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        import signal
        import time
        from orchestrator import freqctl

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        marker = Path({str(marker)!r})
        freqctl.install_emergency_handlers(lambda: marker.write_text("restored") or True)
        print("READY", flush=True)
        while True:
            time.sleep(0.1)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == -signal.SIGINT
        assert marker.read_text() == "restored"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_frq10_read_observed_frequency_khz(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, max_khz=1862000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    (tmp_path / "cpu0/cpufreq/scaling_cur_freq").write_text("1862000")

    assert freqctl.read_observed_frequency_khz(env, 0) == 1862000
    assert freqctl.read_observed_frequency_khz(env, 99) is None


def test_arc108_write_and_verify_reintenta_ante_falla_transitoria(tmp_path, monkeypatch):
    # ARC-108: bajo carga intensa justo antes de la escritura, intel_pstate
    # puede rechazar transitoriamente una escritura -- reproducido en pacca
    # de forma no determinista (el mismo comando, fuera del harness, nunca
    # falla). Simula: la relectura falla las dos primeras veces y luego
    # coincide, sin que el archivo real cambie de contenido entre lecturas
    # (monkeypatch de _read_text, no del archivo).
    path = tmp_path / "scaling_max_freq"
    path.write_text("800000")
    sleeps = []
    monkeypatch.setattr(freqctl.time, "sleep", lambda s: sleeps.append(s))

    reads = iter(["800000", "800000", "2261000"])
    monkeypatch.setattr(freqctl, "_read_text", lambda p: next(reads))

    assert freqctl._write_and_verify(path, "2261000", attr="scaling_max_freq", cpu=0) is True
    assert len(sleeps) == 2


def test_arc108_write_and_verify_falla_tras_agotar_reintentos(tmp_path, monkeypatch):
    path = tmp_path / "scaling_max_freq"
    path.write_text("800000")
    monkeypatch.setattr(freqctl.time, "sleep", lambda s: None)
    monkeypatch.setattr(freqctl, "_read_text", lambda p: "800000")

    # Un permiso realmente ausente falla siempre, no de forma intermitente
    # -- el reintento no debe convertir esto en un falso positivo.
    assert freqctl._write_and_verify(path, "2261000", attr="scaling_max_freq", cpu=0) is False


def test_arc161_wait_for_frequency_settled_retorna_de_inmediato_si_ya_esta_dentro_de_tolerancia(tmp_path):
    paths = {
        0: _write_cpu(tmp_path, 0, max_khz=805000),
        1: _write_cpu(tmp_path, 1, max_khz=798000),
    }
    env = _env(paths, write_capable=True, strategy="bounded_range")

    observed = freqctl.wait_for_frequency_settled(
        [0, 1], 800000, env, tolerance_fraction=0.05, timeout_seconds=5.0,
    )

    assert observed == {0: 805000, 1: 798000}


def test_arc161_wait_for_frequency_settled_target_none_no_espera(tmp_path, monkeypatch):
    # ARC-161: un nivel REF/native_governor no tiene objetivo numerico --
    # nada que asentar, retorna de inmediato sin dormir ni fallar.
    paths = {0: _write_cpu(tmp_path, 0, max_khz=3200000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    monkeypatch.setattr(freqctl.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("no deberia dormir")))

    observed = freqctl.wait_for_frequency_settled(
        [0], None, env, tolerance_fraction=0.05, timeout_seconds=5.0,
    )

    assert observed == {0: 3200000}


def test_arc161_wait_for_frequency_settled_reintenta_hasta_asentar(tmp_path, monkeypatch):
    # ARC-161: simula el decaimiento lento real de paccaA100 (EPP=performance
    # bajo HWP) -- las primeras lecturas siguen cerca del nivel anterior
    # (3200000), y solo tras un par de reintentos cae dentro de tolerancia
    # del objetivo (800000).
    cur_path = tmp_path / "cpu0/cpufreq/scaling_cur_freq"
    paths = {0: _write_cpu(tmp_path, 0, max_khz=3200000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")

    readings = iter(["3200000", "2100000", "820000"])

    def fake_sleep(_seconds: float) -> None:
        cur_path.write_text(next(readings))

    monkeypatch.setattr(freqctl.time, "sleep", fake_sleep)

    observed = freqctl.wait_for_frequency_settled(
        [0], 800000, env, tolerance_fraction=0.05, timeout_seconds=5.0, poll_interval_seconds=0.01,
    )

    assert observed == {0: 820000}


def test_arc161_wait_for_frequency_settled_falla_tras_timeout(tmp_path, monkeypatch):
    # ARC-161: si la frecuencia nunca converge dentro del plazo, falla en
    # voz alta (FrequencyControlError) en vez de proceder a medir sin
    # confirmacion -- exactamente el problema que este mecanismo evita.
    paths = {0: _write_cpu(tmp_path, 0, max_khz=3200000)}
    env = _env(paths, write_capable=True, strategy="bounded_range")
    monkeypatch.setattr(freqctl.time, "sleep", lambda s: None)

    clock = iter([0.0, 0.01, 0.02, 10.0])
    monkeypatch.setattr(freqctl.time, "monotonic", lambda: next(clock))

    with pytest.raises(freqctl.FrequencyControlError):
        freqctl.wait_for_frequency_settled(
            [0], 800000, env, tolerance_fraction=0.05, timeout_seconds=5.0, poll_interval_seconds=0.01,
        )


def test_arc161_settle_if_configured_no_hace_nada_si_deshabilitado(tmp_path, monkeypatch):
    applied = freqctl.AppliedFrequency(
        level_id="F4", strategy="bounded_range", requested_khz=800000, applied_khz=800000,
        per_cpu_applied_khz={0: 800000}, governor_applied=None, write_skipped_reason=None,
    )
    env = _env({0: _write_cpu(tmp_path, 0)}, write_capable=True, strategy="bounded_range")
    monkeypatch.setattr(
        freqctl, "wait_for_frequency_settled",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no deberia llamarse")),
    )

    assert freqctl.settle_if_configured([0], applied, env, settle_config=None) is None
    assert freqctl.settle_if_configured([0], applied, env, settle_config={}) is None
    assert freqctl.settle_if_configured([0], applied, env, settle_config={"enabled": False}) is None


def test_arc161_settle_if_configured_delega_con_los_parametros_del_manifiesto(tmp_path, monkeypatch):
    applied = freqctl.AppliedFrequency(
        level_id="F4", strategy="bounded_range", requested_khz=800000, applied_khz=800000,
        per_cpu_applied_khz={0: 800000}, governor_applied=None, write_skipped_reason=None,
    )
    env = _env({0: _write_cpu(tmp_path, 0)}, write_capable=True, strategy="bounded_range")

    captured = {}

    def fake_wait(cpus, target_khz, env_arg, *, tolerance_fraction, timeout_seconds, poll_interval_seconds):
        captured.update(
            cpus=cpus, target_khz=target_khz, tolerance_fraction=tolerance_fraction,
            timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds,
        )
        return {0: 800000}

    monkeypatch.setattr(freqctl, "wait_for_frequency_settled", fake_wait)
    monkeypatch.setattr(freqctl, "_start_warmup_load", lambda cpus: ["fake_proc"])
    monkeypatch.setattr(freqctl, "_stop_warmup_load", lambda procs: None)

    result = freqctl.settle_if_configured(
        [0], applied, env,
        settle_config={"enabled": True, "tolerance_fraction": 0.05, "timeout_seconds": 12.0, "poll_interval_seconds": 0.25},
    )

    assert result == {0: 800000}
    assert captured == {
        "cpus": [0], "target_khz": 800000, "tolerance_fraction": 0.05,
        "timeout_seconds": 12.0, "poll_interval_seconds": 0.25,
    }


def test_arc165_settle_if_configured_arranca_y_detiene_el_warmup_alrededor_del_sondeo(tmp_path, monkeypatch):
    applied = freqctl.AppliedFrequency(
        level_id="F0", strategy="bounded_range", requested_khz=3200000, applied_khz=3200000,
        per_cpu_applied_khz={0: 3200000}, governor_applied=None, write_skipped_reason=None,
    )
    env = _env({0: _write_cpu(tmp_path, 0)}, write_capable=True, strategy="bounded_range")

    calls = []
    fake_procs = ["fake_proc_0"]

    def fake_start(cpus):
        calls.append(("start", list(cpus)))
        return fake_procs

    def fake_stop(procs):
        calls.append(("stop", procs))

    monkeypatch.setattr(freqctl, "_start_warmup_load", fake_start)
    monkeypatch.setattr(freqctl, "_stop_warmup_load", fake_stop)
    monkeypatch.setattr(freqctl, "wait_for_frequency_settled", lambda *a, **k: calls.append(("wait",)) or {0: 3200000})

    freqctl.settle_if_configured(
        [0], applied, env,
        settle_config={"enabled": True, "tolerance_fraction": 0.05, "timeout_seconds": 12.0},
    )

    assert calls == [("start", [0]), ("wait",), ("stop", fake_procs)]


def test_arc165_settle_if_configured_detiene_el_warmup_aunque_el_sondeo_falle(tmp_path, monkeypatch):
    applied = freqctl.AppliedFrequency(
        level_id="F0", strategy="bounded_range", requested_khz=3200000, applied_khz=3200000,
        per_cpu_applied_khz={0: 3200000}, governor_applied=None, write_skipped_reason=None,
    )
    env = _env({0: _write_cpu(tmp_path, 0)}, write_capable=True, strategy="bounded_range")

    stopped = []
    monkeypatch.setattr(freqctl, "_start_warmup_load", lambda cpus: ["fake_proc"])
    monkeypatch.setattr(freqctl, "_stop_warmup_load", lambda procs: stopped.append(procs))

    def fake_wait_falla(*a, **k):
        raise freqctl.FrequencyControlError("timeout simulado")

    monkeypatch.setattr(freqctl, "wait_for_frequency_settled", fake_wait_falla)

    with pytest.raises(freqctl.FrequencyControlError):
        freqctl.settle_if_configured(
            [0], applied, env,
            settle_config={"enabled": True, "tolerance_fraction": 0.05, "timeout_seconds": 12.0},
        )

    assert stopped == [["fake_proc"]]


def test_arc165_settle_if_configured_sin_objetivo_numerico_no_arranca_warmup(tmp_path, monkeypatch):
    applied = freqctl.AppliedFrequency(
        level_id="REF", strategy="bounded_range", requested_khz=None, applied_khz=None,
        per_cpu_applied_khz={0: None}, governor_applied=None, write_skipped_reason=None,
    )
    env = _env({0: _write_cpu(tmp_path, 0)}, write_capable=True, strategy="bounded_range")

    monkeypatch.setattr(
        freqctl, "_start_warmup_load",
        lambda cpus: (_ for _ in ()).throw(AssertionError("no deberia arrancar warmup sin objetivo")),
    )
    monkeypatch.setattr(freqctl, "wait_for_frequency_settled", lambda *a, **k: {0: 800000})

    result = freqctl.settle_if_configured(
        [0], applied, env,
        settle_config={"enabled": True, "tolerance_fraction": 0.05, "timeout_seconds": 12.0},
    )
    assert result == {0: 800000}


def test_arc165_start_warmup_load_lanza_taskset_yes_por_cpu(monkeypatch):
    launched = []

    class _FakeProc:
        def __init__(self, cmd):
            self.cmd = cmd

    def fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return _FakeProc(cmd)

    monkeypatch.setattr(freqctl.subprocess, "Popen", fake_popen)

    procs = freqctl._start_warmup_load([0, 1])

    assert len(procs) == 2
    assert launched == [["taskset", "-c", "0", "yes"], ["taskset", "-c", "1", "yes"]]


def test_arc165_stop_warmup_load_termina_y_espera_cada_proceso(monkeypatch):
    class _FakeProc:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True

    procs = [_FakeProc(), _FakeProc()]
    freqctl._stop_warmup_load(procs)

    assert all(p.terminated and p.waited for p in procs)
