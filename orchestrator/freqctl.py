from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import signal
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


def _cur_freq_path(env: Any, cpu: int) -> Path | None:
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


def _write_and_verify(path: Path, value: str, *, attr: str, cpu: int) -> bool:
    """FRQ-02/FRQ-04: never assume a write succeeded; reread and compare."""
    _write_text(path, value)
    observed = _read_text(path)
    if observed != value:
        logger.error(
            "freqctl: %s en cpu%d no coincide tras escribir (esperado=%r, leido=%r)",
            attr, cpu, value, observed,
        )
        return False
    return True


def _nearest_available(target_khz: float, available_khz: Iterable[int]) -> int:
    return min(available_khz, key=lambda value: abs(value - target_khz))


def _target_khz(level: Any, available_khz: Iterable[int]) -> int:
    values = list(available_khz)
    low, high = min(values), max(values)
    fraction = getattr(level, "fraction", None)
    if fraction is None:
        raise ValueError(f"freqctl: nivel {getattr(level, 'id', '?')!r} es fixed pero no declara fraction")
    return round(low + float(fraction) * (high - low))


def snapshot_original_state(cpus: Iterable[int], env: Any) -> OriginalState:
    """FRQ-01: read-only snapshot, taken exactly once at campaign start.

    Safe to call regardless of frequency_write_capable: it never writes.
    """
    cpus = tuple(cpus)
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
        per_cpu_applied[cpu] = _read_int(_cur_freq_path(env, cpu))
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
    """Apply one FrequencyLevel to exactly `cpus` (FRQ-09: never node-wide).

    FRQ-06: if frequency_write_capable is False, this never touches sysfs and
    reports strategy="unavailable" regardless of env.frequency_control_strategy.
    """
    cpus = tuple(cpus)
    if not cpus:
        raise ValueError("freqctl: apply_frequency requiere al menos un CPU")

    if not getattr(env, "frequency_write_capable", False):
        return _apply_unavailable(level.id)

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
            if state.governor is not None:
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

    previous_handlers: dict[int, Any] = {}

    def _handler(signum: int, frame: Any) -> None:
        try:
            restore()
        finally:
            # Re-arm the default disposition and re-raise so the process
            # actually terminates with the expected signal semantics instead
            # of silently swallowing SIGINT/SIGTERM.
            signal.signal(signum, previous_handlers.get(signum, signal.SIG_DFL))
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handler)


def read_observed_frequency_khz(env: Any, cpu: int) -> int | None:
    """FRQ-10 support: scaling_cur_freq for one CPU, for postprocess.py to
    attach per-window. Read-only; safe under any frequency_write_capable."""
    return _read_int(_cur_freq_path(env, cpu))
