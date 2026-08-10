from __future__ import annotations

from dataclasses import dataclass
import logging
import subprocess
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

STRATEGY_LOCKED_CLOCKS = "locked_clocks"
STRATEGY_UNAVAILABLE = "unavailable"

# ARC-87: GPU no tiene un análogo de scaling_governor/scaling_min_freq que
# pueda os.access()-earse como en CPU (E09) -- el mecanismo real es
# nvmlDeviceSetGpuLockedClocks, invocado aquí vía `nvidia-smi -lgc/-rgc`
# (ARC-62 confirmó que la vía histórica sin privilegios, Applications
# Clocks, está deprecada en este driver; -lgc exige root). Por eso este
# módulo no tiene un OriginalState por-atributo como freqctl.py: la GPU no
# tiene un estado "original" persistente que snapshotear más allá de "sin
# reloj fijado" (comportamiento por defecto del driver), así que restaurar
# siempre significa `nvidia-smi -rgc`, incondicionalmente e idempotente.


class GpuFrequencyControlError(RuntimeError):
    """nvidia-smi reportó una falla real al fijar o restablecer el reloj de GPU."""


@dataclass(frozen=True)
class AppliedGpuFrequency:
    """Espejo de freqctl.AppliedFrequency para el eje de frecuencia de GPU."""

    level_id: str
    strategy: str
    requested_mhz: int | None
    applied_mhz: int | None
    write_skipped_reason: str | None  # "unavailable", o None
    # ARC-94: reloj SM realmente leido tras `-lgc`/`-rgc`, via una consulta
    # independiente (`nvidia-smi --query-gpu=clocks.sm`) -- a diferencia de
    # freqctl.py (que siempre relee y verifica el sysfs escrito antes de
    # confiar en un returncode==0), esta funcion originalmente asumia
    # exito solo por el returncode de `-lgc`/`-rgc`, sin relectura alguna.
    observed_sm_mhz: int | None = None


def _nearest_available(target_mhz: float, available_mhz: Iterable[int]) -> int:
    return min(available_mhz, key=lambda value: abs(value - target_mhz))


def _target_mhz(level: Any, available_mhz: Iterable[int]) -> int:
    values = list(available_mhz)
    low, high = min(values), max(values)
    fraction = getattr(level, "fraction", None)
    if fraction is None:
        raise ValueError(f"gpu_freqctl: nivel {getattr(level, 'id', '?')!r} es fixed pero no declara fraction")
    return round(low + float(fraction) * (high - low))


def _default_run_nvidia_smi(args: list[str], *, gpu_index: int) -> subprocess.CompletedProcess:
    # ARC-104: -lgc/-rgc exigen root en este driver (ARC-62); pacca delega
    # esto vía sudo restringido a la cuenta de ejecución (no root literal),
    # así que la escritura real -- a diferencia de la relectura de solo
    # lectura en _default_query_sm_clock_mhz, que no lo necesita -- debe
    # invocarse con sudo o falla con "Insufficient Permissions" siempre.
    return subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(gpu_index), *args],
        capture_output=True, text=True, timeout=30, check=False,
    )


def _default_query_sm_clock_mhz(gpu_index: int) -> int | None:
    """Relectura independiente del reloj SM real, vía una consulta separada
    de la que aplicó el cambio -- mismo principio que freqctl._write_and_verify
    (nunca confiar solo en el returncode de la escritura)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _apply_unavailable(level_id: str) -> AppliedGpuFrequency:
    # ARC-87 (espejo de FRQ-06): ningún comando de nvidia-smi que escriba se
    # invoca por esta rama.
    return AppliedGpuFrequency(
        level_id=level_id,
        strategy=STRATEGY_UNAVAILABLE,
        requested_mhz=None,
        applied_mhz=None,
        write_skipped_reason="unavailable",
    )


def apply_gpu_frequency(
    level: Any,
    env: Any,
    *,
    gpu_index: int = 0,
    run_nvidia_smi: Callable[..., subprocess.CompletedProcess] = _default_run_nvidia_smi,
    query_sm_clock_mhz: Callable[[int], int | None] = _default_query_sm_clock_mhz,
) -> AppliedGpuFrequency:
    """Fija el reloj de SM de la GPU al valor que implica `level.fraction`
    sobre `env.gpu_available_clocks_mhz`, vía `nvidia-smi -lgc <t>,<t>`
    (min=max=target, mismo patrón de "pin a un solo punto" que
    freqctl._apply_bounded usa para el rango de intel_pstate).

    Nunca toca el dispositivo si `env.gpu_frequency_write_capable` es falso
    (reporta strategy="unavailable" en su lugar, igual que
    freqctl.apply_frequency). Para `mode == "native_governor"` (el nivel de
    referencia REF), en vez de fijar un punto, restablece el comportamiento
    por defecto del driver (`-rgc`) -- GPU no tiene una noción de "governor
    nativo que hay que reproducir" como CPU, el estado nativo simplemente es
    "sin reloj fijado".

    ARC-94: además del returncode de `-lgc`/`-rgc`, se hace una relectura
    independiente (`query_sm_clock_mhz`, por defecto
    `nvidia-smi --query-gpu=clocks.sm`) -- antes de este cambio, el éxito se
    asumía solo por returncode==0, a diferencia de freqctl.py (CPU), que
    siempre relee el sysfs escrito. La relectura de GPU no puede exigir
    igualdad estricta con el target: el reloj SM real cae a un nivel
    ocioso más bajo cuando no hay carga, incluso con el techo fijado por
    `-lgc` -- eso es comportamiento esperado, no una falla. Lo que sí es
    evidencia inequívoca de que el candado no se aplicó es observar un
    reloj **por encima** del techo fijado; ese caso sí bloquea con
    `GpuFrequencyControlError`. Si la consulta de relectura falla o no
    está disponible, se registra `observed_sm_mhz=None` sin bloquear -- es
    una verificación adicional, no un requisito nuevo de la ruta feliz.
    """
    if not getattr(env, "gpu_frequency_write_capable", False):
        return _apply_unavailable(level.id)

    if getattr(level, "mode", None) == "native_governor":
        try:
            result = run_nvidia_smi(["-rgc"], gpu_index=gpu_index)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GpuFrequencyControlError(
                f"gpu_freqctl: nvidia-smi -rgc no se pudo ejecutar para el nivel {level.id!r}: {exc}"
            ) from exc
        if result.returncode != 0:
            raise GpuFrequencyControlError(
                f"gpu_freqctl: nvidia-smi -rgc falló para el nivel {level.id!r}: {result.stderr.strip()}"
            )
        return AppliedGpuFrequency(
            level_id=level.id,
            strategy=STRATEGY_LOCKED_CLOCKS,
            requested_mhz=None,
            applied_mhz=None,
            write_skipped_reason=None,
            observed_sm_mhz=query_sm_clock_mhz(gpu_index),
        )

    available = getattr(env, "gpu_available_clocks_mhz", None)
    if not available:
        raise GpuFrequencyControlError(
            f"gpu_freqctl: apply_gpu_frequency({level.id!r}) requiere gpu_available_clocks_mhz no vacío"
        )
    target = _nearest_available(_target_mhz(level, available), available)
    try:
        result = run_nvidia_smi(["-lgc", f"{target},{target}"], gpu_index=gpu_index)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuFrequencyControlError(
            f"gpu_freqctl: nvidia-smi -lgc {target} no se pudo ejecutar para el nivel {level.id!r}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise GpuFrequencyControlError(
            f"gpu_freqctl: nvidia-smi -lgc {target} falló para el nivel {level.id!r}: {result.stderr.strip()}"
        )
    observed = query_sm_clock_mhz(gpu_index)
    if observed is not None and observed > target:
        raise GpuFrequencyControlError(
            f"gpu_freqctl: apply_gpu_frequency({level.id!r}) relectura {observed}MHz supera "
            f"el techo fijado {target}MHz -- el candado no parece haberse aplicado"
        )
    return AppliedGpuFrequency(
        level_id=level.id,
        strategy=STRATEGY_LOCKED_CLOCKS,
        requested_mhz=target,
        applied_mhz=target,
        write_skipped_reason=None,
        observed_sm_mhz=observed,
    )


def restore_gpu_state(
    env: Any,
    *,
    gpu_index: int = 0,
    run_nvidia_smi: Callable[..., subprocess.CompletedProcess] = _default_run_nvidia_smi,
) -> bool:
    """`nvidia-smi -rgc` incondicional -- idempotente incluso si nunca se
    fijó nada (no hay estado por-atributo que snapshotear, a diferencia de
    freqctl.OriginalState). Nunca lanza: puede llamarse desde un manejador
    de señal sin segunda oportunidad, mismo contrato best-effort que
    freqctl.restore_original_state (devuelve bool, registra el error, no
    interrumpe la restauración de otros subsistemas).

    ARC-95: el "nunca lanza" del docstring anterior no estaba garantizado
    en código -- ``run_nvidia_smi`` (un ``subprocess.run`` real) puede
    levantar ``OSError``/``FileNotFoundError`` (binario ausente) o
    ``subprocess.TimeoutExpired`` sin que nada lo capturara aquí, exactamente
    el tipo de excepción que un manejador de SIGINT/SIGTERM sin segunda
    oportunidad no puede permitirse.
    """
    if not getattr(env, "gpu_frequency_write_capable", False):
        # Igual que FRQ-06 en freqctl: nunca se escribió nada, nada que
        # restaurar.
        return True
    try:
        result = run_nvidia_smi(["-rgc"], gpu_index=gpu_index)
    except (OSError, subprocess.SubprocessError):
        logger.exception("gpu_freqctl: nvidia-smi -rgc no se pudo ejecutar durante la restauración")
        return False
    if result.returncode != 0:
        logger.error("gpu_freqctl: nvidia-smi -rgc no confirmó el reset: %s", result.stderr.strip())
        return False
    return True
