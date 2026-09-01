from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

STRATEGY_DISCRETE = "discrete_bounds"
STRATEGY_BOUNDED = "bounded_range"
STRATEGY_UNAVAILABLE = "unavailable"

USERSPACE_GOVERNOR = "userspace"

_GOVERNOR_ATTR = "scaling_governor"
_MIN_ATTR = "scaling_min_freq"
_MAX_ATTR = "scaling_max_freq"
_SETSPEED_ATTR = "scaling_setspeed"
_CUR_FREQ_ATTR = "scaling_cur_freq"


class FrequencyControlError(RuntimeError):
    """A sysfs write did not take effect on reread (FRQ-02/FRQ-04)."""


@dataclass(frozen=True)
class CpuOriginalState:
    governor: str | None
    min_freq_khz: int | None
    max_freq_khz: int | None
    setspeed_khz: int | None


@dataclass(frozen=True)
class OriginalState:
    """FRQ-01: the campaign takes exactly one of these, at startup."""

    cpus: tuple[int, ...]
    strategy: str
    per_cpu: Mapping[int, CpuOriginalState] = field(default_factory=dict)


@dataclass(frozen=True)
class AppliedFrequency:
    """FRQ-03: requested and applied are always both present, never only one."""

    level_id: str
    strategy: str
    requested_khz: int | None
    applied_khz: int | None
    per_cpu_applied_khz: Mapping[int, int | None]
    governor_applied: str | None
    write_skipped_reason: str | None  # "unavailable" (FRQ-06), or None


def _control_paths(env: Any) -> Mapping[int, Mapping[str, str]]:
    return getattr(env, "frequency_control_paths", {}) or {}


def _attr_path(env: Any, cpu: int, attr: str) -> Path | None:
    controls = _control_paths(env).get(cpu, {})
    value = controls.get(attr)
    return Path(value) if value else None


def _setspeed_path(env: Any, cpu: int) -> Path | None:
    # scaling_setspeed is only exposed by the kernel while governor=userspace
    # is active, so it is never in the statically-discovered control_paths
    # (environment.py detects the node before freqctl ever runs). Derive it
    # as a sibling of scaling_governor instead of assuming it is present.
    governor_path = _attr_path(env, cpu, _GOVERNOR_ATTR)
    return governor_path.parent / _SETSPEED_ATTR if governor_path else None


def cur_freq_path(env: Any, cpu: int) -> Path | None:
    """Path to one CPU's scaling_cur_freq. Public (ARC-135): runner.py also
    uses this to build --cpu-freq-sysfs-path for the C++ collector's own
    per-window sampling, not just this module's own post-hoc reads."""
    governor_path = _attr_path(env, cpu, _GOVERNOR_ATTR)
    return governor_path.parent / _CUR_FREQ_ATTR if governor_path else None


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_int(path: Path | None) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


_WRITE_VERIFY_RETRIES = 3
_WRITE_VERIFY_RETRY_DELAY_S = 0.05


def _write_and_verify(path: Path, value: str, *, attr: str, cpu: int) -> bool:
    """FRQ-02/FRQ-04: never assume a write succeeded; reread and compare.

    ARC-108: bajo carga intensa en el núcleo justo antes de esta escritura
    (confirmado en pacca: el HWP de intel_pstate puede rechazar
    transitoriamente una escritura de scaling_min_freq inmediatamente
    después de una calibración de FLOPs pico que satura los núcleos
    delegados), la primera relectura puede no coincidir aun cuando el
    permiso es real y la escritura eventualmente se aplica -- reproducido
    de forma determinista fuera de este instrumento (mismo comando de
    shell, mismo orden) sin ninguna falla, así que no es un problema de
    permiso ni de orden de escritura. Se reintenta unas pocas veces con
    una espera corta antes de declarar la falla -- un permiso realmente
    ausente sigue fallando siempre (nunca sería intermitente), así que
    este reintento no puede enmascarar el caso que FRQ-02 existe para
    detectar.
    """
    for attempt in range(_WRITE_VERIFY_RETRIES):
        _write_text(path, value)
        observed = _read_text(path)
        if observed == value:
            return True
        if attempt < _WRITE_VERIFY_RETRIES - 1:
            time.sleep(_WRITE_VERIFY_RETRY_DELAY_S)
    logger.error(
        "freqctl: %s en cpu%d no coincide tras escribir (esperado=%r, leido=%r, intentos=%d)",
        attr, cpu, value, observed, _WRITE_VERIFY_RETRIES,
    )
    return False


def _nearest_available(target_khz: float, available_khz: Iterable[int]) -> int:
    return min(available_khz, key=lambda value: abs(value - target_khz))


def _target_khz(level: Any, available_khz: Iterable[int]) -> int:
    values = list(available_khz)
    low, high = min(values), max(values)
    fraction = getattr(level, "fraction", None)
    if fraction is None:
        raise ValueError(f"freqctl: nivel {getattr(level, 'id', '?')!r} es fixed pero no declara fraction")
    return round(low + float(fraction) * (high - low))


def _expand_with_smt_siblings(cpus: Iterable[int], env: Any) -> tuple[int, ...]:
    """ARC-163: los CPUs delegados pueden compartir núcleo físico con
    hermanos SMT que el manifiesto nunca declara -- confirmado en paccaA100
    (ARC-162, Prueba B) que un hermano sin restringir permite que el reloj
    físico del núcleo compartido supere el candado del CPU delegado bajo
    carga real, pese a que cada CPU lógico tiene su propia política cpufreq
    independiente en software. `env.smt_siblings` (poblado por
    environment.detect_environment(), incluye typically al propio CPU +
    su(s) hermano(s)) es la fuente de verdad -- ausente o vacío (SMT
    deshabilitado, o entorno de prueba sin ese campo) no cambia el
    comportamiento anterior, nunca se infiere una topología no confirmada.
    """
    expanded = set(cpus)
    for siblings in (getattr(env, "smt_siblings", None) or {}).values():
        expanded.update(siblings)
    return tuple(sorted(expanded))


def snapshot_original_state(cpus: Iterable[int], env: Any) -> OriginalState:
    """FRQ-01: read-only snapshot, taken exactly once at campaign start.

    Safe to call regardless of frequency_write_capable: it never writes.

    ARC-163: el conjunto snapshoteado se expande a los hermanos SMT de
    `cpus` (ver `_expand_with_smt_siblings`) -- `restore_original_state()`
    solo puede devolver a su estado original lo que este snapshot capturó,
    así que si `apply_frequency()` va a escribir en los hermanos, este
    snapshot debe conocer su estado previo o la restauración los dejaría
    modificados permanentemente.
    """
    cpus = _expand_with_smt_siblings(cpus, env)
    strategy = getattr(env, "frequency_control_strategy", STRATEGY_UNAVAILABLE)
    if not _control_paths(env):
        return OriginalState(cpus=cpus, strategy=STRATEGY_UNAVAILABLE, per_cpu={})

    per_cpu: dict[int, CpuOriginalState] = {}
    for cpu in cpus:
        per_cpu[cpu] = CpuOriginalState(
            governor=_read_text(_attr_path(env, cpu, _GOVERNOR_ATTR)),
            min_freq_khz=_read_int(_attr_path(env, cpu, _MIN_ATTR)),
            max_freq_khz=_read_int(_attr_path(env, cpu, _MAX_ATTR)),
            setspeed_khz=_read_int(_setspeed_path(env, cpu)),
        )
    return OriginalState(cpus=cpus, strategy=strategy, per_cpu=per_cpu)


def _apply_unavailable(level_id: str) -> AppliedFrequency:
    # FRQ-06: no sysfs write of any kind happens on this path.
    return AppliedFrequency(
        level_id=level_id,
        strategy=STRATEGY_UNAVAILABLE,
        requested_khz=None,
        applied_khz=None,
        per_cpu_applied_khz={},
        governor_applied=None,
        write_skipped_reason="unavailable",
    )


def _apply_native_governor(cpus: tuple[int, ...], env: Any, level: Any, original: OriginalState) -> AppliedFrequency:
    """ARC-94: además de restaurar el string del governor, restaura
    scaling_min_freq/scaling_max_freq a su rango original -- si un nivel
    fixed (bounded_range) se aplicó antes que REF en la misma corrida (una
    matriz de campaña real intercala niveles), el rango se queda pinneado
    en ese nivel para siempre si solo se restaura el governor: REF dejaría
    de ser "frecuencia nativa/libre" y en realidad seguiría fijo al último
    nivel medido. Esto también contamina calibration_references() si esas
    corridas de referencia se miden después de un nivel fixed sin volver a
    aplicar REF explícitamente (ver calibration.run_calibration_references).

    Segunda corrección, misma sesión: el string del governor solo se
    reescribe para la estrategia ``discrete_bounds`` (acpi-cpufreq), la
    única que alguna vez lo cambia (_apply_discrete lo fija a
    ``userspace``). ``bounded_range`` (intel_pstate/amd-pstate) nunca toca
    ``scaling_governor`` -- pinea min=max=target bajo el governor que ya
    esté activo -- así que exigir permiso de escritura sobre ese archivo
    para restaurar REF era innecesario y, peor, hacía que
    ``frequency_write_capable`` dependiera de un permiso (escritura de
    governor) que el permiso P1 solicitado (solo scaling_min_freq/max_freq)
    nunca iba a conceder.
    """
    strategy = getattr(env, "frequency_control_strategy", STRATEGY_UNAVAILABLE)
    per_cpu_applied: dict[int, int | None] = {}
    governor_ok = True
    for cpu in cpus:
        state = original.per_cpu.get(cpu)
        governor_path = _attr_path(env, cpu, _GOVERNOR_ATTR)
        if state is None or state.governor is None:
            per_cpu_applied[cpu] = None
            continue
        if state.min_freq_khz is not None and state.max_freq_khz is not None:
            min_path = _attr_path(env, cpu, _MIN_ATTR)
            max_path = _attr_path(env, cpu, _MAX_ATTR)
            if min_path is not None and max_path is not None:
                governor_ok = _write_range_safe(
                    min_path, max_path, state.min_freq_khz, state.max_freq_khz, cpu=cpu,
                ) and governor_ok
        if strategy == STRATEGY_DISCRETE and governor_path is not None:
            ok = _write_and_verify(governor_path, state.governor, attr=_GOVERNOR_ATTR, cpu=cpu)
            governor_ok = governor_ok and ok
        per_cpu_applied[cpu] = _read_int(cur_freq_path(env, cpu))
    if not governor_ok:
        raise FrequencyControlError(f"freqctl: no se pudo restaurar el governor nativo para el nivel {level.id!r}")
    return AppliedFrequency(
        level_id=level.id,
        strategy=getattr(env, "frequency_control_strategy", STRATEGY_UNAVAILABLE),
        requested_khz=None,
        applied_khz=None,
        per_cpu_applied_khz=per_cpu_applied,
        governor_applied=next((original.per_cpu[c].governor for c in cpus if c in original.per_cpu), None),
        write_skipped_reason=None,
    )


def _apply_discrete(cpus: tuple[int, ...], level: Any, env: Any) -> AppliedFrequency:
    target = _nearest_available(_target_khz(level, env.available_frequencies_khz), env.available_frequencies_khz)
    per_cpu_applied: dict[int, int | None] = {}
    all_ok = True
    for cpu in cpus:
        governor_path = _attr_path(env, cpu, _GOVERNOR_ATTR)
        if governor_path is None:
            raise FrequencyControlError(f"freqctl: cpu{cpu} no tiene scaling_governor en frequency_control_paths")
        ok = _write_and_verify(governor_path, USERSPACE_GOVERNOR, attr=_GOVERNOR_ATTR, cpu=cpu)
        setspeed_path = _setspeed_path(env, cpu)
        if setspeed_path is None:
            raise FrequencyControlError(f"freqctl: cpu{cpu} no expone scaling_setspeed")
        ok = _write_and_verify(setspeed_path, str(target), attr=_SETSPEED_ATTR, cpu=cpu) and ok
        all_ok = all_ok and ok
        per_cpu_applied[cpu] = _read_int(setspeed_path)
    if not all_ok:
        raise FrequencyControlError(f"freqctl: apply_frequency({level.id!r}) discrete_bounds falló la relectura")
    return AppliedFrequency(
        level_id=level.id,
        strategy=STRATEGY_DISCRETE,
        requested_khz=target,
        applied_khz=target,
        per_cpu_applied_khz=per_cpu_applied,
        governor_applied=USERSPACE_GOVERNOR,
        write_skipped_reason=None,
    )


def _write_range_safe(min_path: Path, max_path: Path, target_min: int, target_max: int, *, cpu: int) -> bool:
    """Escribe scaling_min_freq/scaling_max_freq en el orden que nunca
    viola min<=max en NINGÚN paso intermedio -- el kernel lo exige en cada
    escritura individual, no solo al final de la secuencia (ARC-94).
    Sirve tanto para pinear a un punto (target_min==target_max, ver
    _apply_bounded) como para restaurar un rango original arbitrario (ver
    _apply_native_governor): si el nuevo techo quedaría por debajo del
    piso vigente, el piso se escribe primero; en cualquier otro caso,
    escribir el techo primero sigue siendo seguro porque el piso vigente
    ya es <= target_max.
    """
    current_min = _read_int(min_path)
    if current_min is not None and target_max < current_min:
        ok = _write_and_verify(min_path, str(target_min), attr=_MIN_ATTR, cpu=cpu)
        ok = _write_and_verify(max_path, str(target_max), attr=_MAX_ATTR, cpu=cpu) and ok
    else:
        ok = _write_and_verify(max_path, str(target_max), attr=_MAX_ATTR, cpu=cpu)
        ok = _write_and_verify(min_path, str(target_min), attr=_MIN_ATTR, cpu=cpu) and ok
    return ok


def _apply_bounded(cpus: tuple[int, ...], level: Any, env: Any) -> AppliedFrequency:
    target = _target_khz(level, env.available_frequencies_khz)
    per_cpu_applied: dict[int, int | None] = {}
    all_ok = True
    for cpu in cpus:
        min_path = _attr_path(env, cpu, _MIN_ATTR)
        max_path = _attr_path(env, cpu, _MAX_ATTR)
        if min_path is None or max_path is None:
            raise FrequencyControlError(f"freqctl: cpu{cpu} no tiene scaling_min_freq/scaling_max_freq")
        # Pin the range to a single point: min == max == target.
        ok = _write_range_safe(min_path, max_path, target, target, cpu=cpu)
        all_ok = all_ok and ok
        per_cpu_applied[cpu] = _read_int(min_path)
    if not all_ok:
        raise FrequencyControlError(f"freqctl: apply_frequency({level.id!r}) bounded_range falló la relectura")
    return AppliedFrequency(
        level_id=level.id,
        strategy=STRATEGY_BOUNDED,
        requested_khz=target,
        applied_khz=target,
        per_cpu_applied_khz=per_cpu_applied,
        governor_applied=None,
        write_skipped_reason=None,
    )


def apply_frequency(
    cpus: Iterable[int], level: Any, env: Any, *, original: OriginalState | None = None
) -> AppliedFrequency:
    """Apply one FrequencyLevel to `cpus` y, cuando `env.smt_siblings` los
    declara, a sus hermanos SMT también (ARC-163 -- ver
    `_expand_with_smt_siblings`; FRQ-09 sigue cumpliéndose en espíritu:
    nunca node-wide, solo los CPUs relacionados con `cpus`).

    FRQ-06: if frequency_write_capable is False, this never touches sysfs and
    reports strategy="unavailable" regardless of env.frequency_control_strategy.
    """
    cpus = tuple(cpus)
    if not cpus:
        raise ValueError("freqctl: apply_frequency requiere al menos un CPU")

    if not getattr(env, "frequency_write_capable", False):
        return _apply_unavailable(level.id)

    cpus = _expand_with_smt_siblings(cpus, env)

    if getattr(level, "mode", None) == "native_governor":
        if original is None:
            raise ValueError("freqctl: apply_frequency(native_governor) requiere el snapshot original")
        return _apply_native_governor(cpus, env, level, original)

    strategy = getattr(env, "frequency_control_strategy", STRATEGY_UNAVAILABLE)
    if strategy == STRATEGY_DISCRETE:
        return _apply_discrete(cpus, level, env)
    if strategy == STRATEGY_BOUNDED:
        return _apply_bounded(cpus, level, env)
    # frequency_write_capable=True without a known strategy should not happen
    # (environment.py only sets write_capable alongside a real strategy), but
    # fail into the safe "no write" branch instead of guessing a mechanism.
    return _apply_unavailable(level.id)


def wait_for_frequency_settled(
    cpus: Iterable[int], target_khz: int | None, env: Any, *,
    tolerance_fraction: float, timeout_seconds: float, poll_interval_seconds: float = 0.2,
) -> Mapping[int, int | None]:
    """ARC-161: espera activa hasta que scaling_cur_freq de cada CPU delegado
    caiga dentro de `tolerance_fraction` de `target_khz`, releyendo cada
    `poll_interval_seconds`, en vez de una pausa fija.

    Motivo: en paccaA100, energy_performance_preference=performance bajo HWP
    hace que el hardware decaiga lentamente hacia un techo de frecuencia más
    bajo tras venir de un nivel más alto -- un barrido real (0.5s a 12s)
    mostró que el asentamiento NO es monótono con el tiempo esperado (8s
    asentó limpio, 12s inmediatamente después falló peor que 8s), así que
    una pausa fija no es confiable sin importar cuánto se alargue: solo
    verificar el estado real, con reintentos, puede confirmarlo. Mismo
    principio que `_write_and_verify` ya aplica a la escritura misma,
    extendido al asentamiento posterior.

    `target_khz=None` (nivel REF/native_governor, sin objetivo numérico) no
    tiene nada que asentar -- retorna de inmediato la lectura actual, sin
    esperar ni fallar.

    Lanza FrequencyControlError si el timeout se agota sin que todos los
    CPUs converjan -- fallar en voz alta es la única opción correcta aquí:
    proceder a medir sin la frecuencia confirmada reintroduciría en silencio
    exactamente el problema que este mecanismo existe para evitar.
    """
    cpus = tuple(cpus)
    if target_khz is None:
        return {cpu: _read_int(cur_freq_path(env, cpu)) for cpu in cpus}

    tolerance_khz = tolerance_fraction * target_khz
    deadline = time.monotonic() + timeout_seconds
    observed: Mapping[int, int | None] = {}
    while True:
        observed = {cpu: _read_int(cur_freq_path(env, cpu)) for cpu in cpus}
        if all(
            value is not None and abs(value - target_khz) <= tolerance_khz
            for value in observed.values()
        ):
            return observed
        if time.monotonic() >= deadline:
            raise FrequencyControlError(
                f"freqctl: la frecuencia no se asentó dentro de {timeout_seconds}s "
                f"(objetivo={target_khz}kHz ± {tolerance_khz:.0f}kHz, observado={observed})"
            )
        time.sleep(poll_interval_seconds)


def _start_warmup_load(cpus: Iterable[int]) -> list[subprocess.Popen]:
    """ARC-165: `scaling_cur_freq` bajo `intel_pstate` se calcula del ratio
    APERF/MPERF -- solo refleja un candado alto pineado cuando el CPU está
    ejecutando instrucciones de verdad; en reposo reporta cerca del piso sin
    importar `scaling_min/max_freq` (confirmado en `paccaA100`, ARC-164).
    Genera esa actividad con un `yes` por CPU, confinado con `taskset`, para
    que `wait_for_frequency_settled` sondee un valor que puede converger.
    Falla en silencio por CPU (best-effort): si `taskset`/`yes` no están
    disponibles, el sondeo posterior simplemente seguirá viendo el CPU en
    reposo y fallará su propio timeout de forma visible -- no hay un modo
    silencioso nuevo que ocultar.
    """
    procs: list[subprocess.Popen] = []
    for cpu in cpus:
        try:
            procs.append(subprocess.Popen(
                ["taskset", "-c", str(cpu), "yes"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        except OSError:
            logger.warning("freqctl: no se pudo lanzar el warm-up de asentamiento en cpu%d", cpu)
    return procs


def _stop_warmup_load(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def settle_if_configured(
    cpus: Iterable[int], applied: AppliedFrequency, env: Any, *, settle_config: Mapping[str, Any] | None,
) -> Mapping[int, int | None] | None:
    """ARC-161: envoltura de conveniencia para los llamadores reales
    (calibration.py, runner.py) -- lee `manifest.frequency_settle` y no hace
    nada si `enabled=False`/ausente (mismo criterio "ausente = deshabilitado"
    que turbo/uncore/gpu, nunca se infiere habilitado). Retorna None cuando
    no se esperó nada, o el último `scaling_cur_freq` observado por CPU
    cuando sí se esperó y asentó.

    ARC-165: cuando hay un objetivo numérico que asentar (no REF/
    native_governor), envuelve el sondeo con un warm-up real en `cpus` (ver
    `_start_warmup_load`) -- sin esto, un CPU inactivo nunca puede confirmar
    un candado alto (ARC-164), sin importar cuánto se espere. El warm-up se
    detiene siempre antes de retornar, incluso si el sondeo lanza
    FrequencyControlError por timeout.
    """
    config = settle_config or {}
    if not config.get("enabled", False):
        return None
    cpus = list(cpus)
    kwargs = dict(
        tolerance_fraction=float(config["tolerance_fraction"]),
        timeout_seconds=float(config["timeout_seconds"]),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 0.2)),
    )
    if applied.requested_khz is None:
        return wait_for_frequency_settled(cpus, applied.requested_khz, env, **kwargs)
    warmup_procs = _start_warmup_load(cpus)
    try:
        return wait_for_frequency_settled(cpus, applied.requested_khz, env, **kwargs)
    finally:
        _stop_warmup_load(warmup_procs)


def restore_original_state(original: OriginalState, env: Any) -> bool:
    """FRQ-04: idempotent; verifies every write by reading it back.

    Returns True only if every attribute that was snapshotted now reads back
    as its original value. Always attempts every CPU, even if an earlier one
    failed, because this can run from a signal handler with no second chance.

    ARC-94 (segunda ronda): dos correcciones de robustez.
    (a) min/max se escriben con ``_write_range_safe`` (el mismo orden
    protegido que ``_apply_bounded``/``_apply_native_governor`` ya usan)
    en vez de min-luego-max sin condición -- defensivo ante cualquier
    estado intermedio que no sea un único punto pinneado.
    (b) cada CPU queda envuelto en su propio try/except: antes, una
    excepción (p.ej. ``PermissionError`` si el permiso real no cubre un
    atributo que se creía escribible) en el CPU N interrumpía el bucle
    ANTES de intentar restaurar N+1 en adelante, contradiciendo la promesa
    del propio docstring ("always attempts every CPU") -- crítico porque
    esta función puede ejecutarse desde un manejador de SIGINT/SIGTERM sin
    segunda oportunidad.
    """
    if original.strategy == STRATEGY_UNAVAILABLE or not original.per_cpu:
        return True
    if not getattr(env, "frequency_write_capable", False):
        # FRQ-06 held throughout the campaign: nothing was ever written, so
        # there is nothing to restore.
        return True

    all_ok = True
    for cpu, state in original.per_cpu.items():
        try:
            if state.min_freq_khz is not None and state.max_freq_khz is not None:
                min_path = _attr_path(env, cpu, _MIN_ATTR)
                max_path = _attr_path(env, cpu, _MAX_ATTR)
                if min_path is not None and max_path is not None:
                    all_ok = _write_range_safe(
                        min_path, max_path, state.min_freq_khz, state.max_freq_khz, cpu=cpu,
                    ) and all_ok
            if state.setspeed_khz is not None:
                setspeed_path = _setspeed_path(env, cpu)
                if setspeed_path is not None and setspeed_path.exists():
                    all_ok = (
                        _write_and_verify(setspeed_path, str(state.setspeed_khz), attr=_SETSPEED_ATTR, cpu=cpu)
                        and all_ok
                    )
            # ARC-95: mismo bug que _apply_native_governor ya corrigió en
            # ARC-94 -- bounded_range (intel_pstate/amd-pstate) nunca
            # escribe scaling_governor, así que restaurarlo aquí exige un
            # permiso (escritura de governor) que P1 no cubre. Solo
            # discrete_bounds (acpi-cpufreq) alguna vez lo cambió
            # (_apply_discrete lo fija a "userspace") y necesita
            # restaurarlo de verdad.
            if state.governor is not None and original.strategy == STRATEGY_DISCRETE:
                governor_path = _attr_path(env, cpu, _GOVERNOR_ATTR)
                if governor_path is not None:
                    all_ok = _write_and_verify(governor_path, state.governor, attr=_GOVERNOR_ATTR, cpu=cpu) and all_ok
        except OSError:
            logger.exception("freqctl: restore_original_state falló en cpu%d, continuando con el resto", cpu)
            all_ok = False
    if not all_ok:
        logger.error("freqctl: restore_original_state no verificó todos los atributos por lectura")
    return all_ok


def install_emergency_handlers(restore: Callable[[], bool | None]) -> None:
    """FRQ-05: atexit, SIGINT and SIGTERM all call `restore`.

    `restore` should be a closure over the campaign's OriginalState/env (e.g.
    ``lambda: restore_original_state(original, env)``), so a crash or Ctrl-C
    mid-campaign still leaves the node in its original frequency state.
    """
    atexit.register(restore)

    def _handler(signum: int, frame: Any) -> None:
        try:
            restore()
        finally:
            # Rearmar SIEMPRE la disposición del sistema y reenviar la señal.
            # No se restaura el handler heredado: un proceso iniciado en
            # background por una shell no interactiva puede heredar SIGINT
            # como SIG_IGN. Rearmar ese valor haría que os.kill() se ignorara
            # y que la campaña continuara después de restaurar sysfs, hallazgo
            # reproducido en hardware real por la prueba de caos FRQ-05
            # (paccaA100, 2026-08-14).
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def read_observed_frequency_khz(env: Any, cpu: int) -> int | None:
    """FRQ-10 support: scaling_cur_freq for one CPU, for postprocess.py to
    attach per-window. Read-only; safe under any frequency_write_capable."""
    return _read_int(cur_freq_path(env, cpu))


def read_governors(cpus: Iterable[int], env: Any) -> dict[int, str | None]:
    """Lee `scaling_governor` de cada CPU (expandido a hermanos SMT, mismo
    criterio que `apply_frequency`/`snapshot_original_state`) -- de solo
    lectura, no requiere `frequency_write_capable`.

    Añadido para fase4_evaluacion/governors.py (§5.1 del plan de
    realineación: comparar el agente contra `ondemand`/`schedutil`, no solo
    contra "lo que el nodo ya tuviera puesto"). No modifica ningún
    comportamiento existente -- es aditivo, como `read_observed_frequency_khz`.
    """
    expanded = _expand_with_smt_siblings(cpus, env)
    return {cpu: _read_text(_attr_path(env, cpu, _GOVERNOR_ATTR)) for cpu in expanded}


def set_governor(cpus: Iterable[int], governor: str, env: Any) -> dict[int, bool]:
    """Escribe `scaling_governor = governor` en cada CPU (expandido a
    hermanos SMT) y verifica por relectura -- misma disciplina que el resto
    del módulo (`_write_and_verify`, con sus reintentos ante rechazo
    transitorio bajo carga, ARC-108).

    A propósito NO restaura nada por sí solo ni se integra con
    `snapshot_original_state`/`restore_original_state`: esas dos funciones
    solo restauran `scaling_governor` cuando `strategy == STRATEGY_DISCRETE`
    (la única vía por la que `apply_frequency` lo cambiaba antes de que
    existiera esta función, ver el comentario ARC-95 en
    `restore_original_state`) -- mezclar esta función con ese mecanismo
    arriesgaría que un gobernador cambiado aquí no se restaurara en una
    ruta de señal/crash que asume ese invariante. `fase4_evaluacion/governors.py`
    hace su propio snapshot con `read_governors()` antes de llamar aquí, y
    restaura llamando a esta misma función con el valor leído.

    Devuelve el resultado por CPU (True = verificado por relectura); no
    lanza por un CPU individual fallido, para que el llamador pueda decidir
    si un fallo parcial es aceptable o debe abortar el escenario completo.
    """
    if not getattr(env, "frequency_write_capable", False):
        raise FrequencyControlError(
            "freqctl.set_governor: env.frequency_write_capable es False -- "
            "sin permiso de escritura detectado en este nodo"
        )
    expanded = _expand_with_smt_siblings(cpus, env)
    results: dict[int, bool] = {}
    for cpu in expanded:
        governor_path = _attr_path(env, cpu, _GOVERNOR_ATTR)
        if governor_path is None:
            logger.error("freqctl.set_governor: cpu%d no tiene scaling_governor en frequency_control_paths", cpu)
            results[cpu] = False
            continue
        results[cpu] = _write_and_verify(governor_path, governor, attr=_GOVERNOR_ATTR, cpu=cpu)
    return results
