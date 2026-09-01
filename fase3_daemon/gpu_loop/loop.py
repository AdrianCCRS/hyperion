"""Loop de GPU del daemon (§4.1/§4.3 punto 6 del plan de realineación).

No corre por tiempo fijo -- corre POR FASE. La fuente de eventos de fase es
`fase3_daemon/gpu_loop/activity_poller.py` (sondeo de `gpu_util_pct` vía
NVML, ver ese módulo para el porqué): `phase_events` aquí es cualquier
iterable de `PhaseBeginEvent`, para poder probarse sin GPU real y sin
depender de la fuente concreta.

⚠️ **Historial de diseño**: la primera versión de este loop consumía
eventos de un shim `LD_PRELOAD` que interceptaba `cudaLaunchKernel`
(`fase3_daemon/shim/`, eliminado). Se abandonó tras confirmar, compilando
ese shim contra CUDA real y cargándolo contra un kernel de prueba, que la
intercepción nunca se dispara para la sintaxis `<<<>>>` en ningún modo de
enlace de cudart -- `nvcc` resuelve esa llamada en tiempo de compilación,
no a través de la tabla de símbolos dinámicos. `activity_poller.py`
documenta el hallazgo completo y las dos alternativas de intercepción
consideradas y no implementadas (trabajo futuro).

⚠️ **Limitación conocida, documentada también en
`fase2_clasificador/README.md`**: no existe todavía un clasificador de GPU
entrenado (features NVML: `gpu_util_pct`, `gpu_mem_util_pct`, `gpu_power_mw`,
`gpu_sm_clock_mhz`, `gpu_temperature_c`, §2.5 del plan). `classify_fn` es
inyectable a propósito -- este loop no asume ninguna implementación
concreta, para no bloquear el resto del wiring en un modelo que todavía no
existe. Con `classify_fn=None`, usa `_placeholder_classify()` (heurística
de ejemplo por `gpu_util_pct`, NUNCA para producción -- lanza si se intenta
usar fuera de pruebas explícitas, ver su docstring).
"""
from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc import gpu_freqctl  # noqa: E402
from fase3_daemon.gpu_loop.controller import (  # noqa: E402
    GpuClockController,
    GpuClockControllerConfig,
    GpuPhaseDecision,
    GpuPhaseLabel,
)


@dataclass(frozen=True)
class GpuFeatures:
    """Snapshot NVML en el instante de inicio de fase -- las mismas 5
    columnas de §2.5 del plan, ninguna requiere `ncu` (nunca en producción,
    ver gpu_clock_controller.hpp)."""
    gpu_util_pct: float
    gpu_mem_util_pct: float
    gpu_power_mw: float
    gpu_sm_clock_mhz: float
    gpu_temperature_c: float


@dataclass(frozen=True)
class PhaseBeginEvent:
    """Un evento de inicio de fase, emitido por `activity_poller.poll_phase_events()`
    (o, en pruebas, por cualquier iterable inyectado)."""
    now_ns: int
    features: GpuFeatures


_NVIDIA_SMI_FIELDS = (
    "utilization.gpu", "utilization.memory", "power.draw", "clocks.sm", "temperature.gpu",
)


def query_gpu_features(gpu_index: int | str | None = None) -> GpuFeatures | None:
    """Snapshot NVML en vivo vía `nvidia-smi`, en el mismo orden que
    `_NVIDIA_SMI_FIELDS`. Devuelve `None` si la consulta falla (GPU no
    disponible, `nvidia-smi` no encontrado) -- el llamador decide qué
    hacer con una muestra sin features (típicamente: no cambiar de estado,
    nunca inventar un valor). Por consistencia con el resto del proyecto
    (`common/hpc/gpu_freqctl.py`, `common/hpc/gpu_inspector.py`, que ya
    usan subprocess sobre `nvidia-smi` en vez de un binding NVML de
    Python), esta función hace lo mismo -- no introduce `pynvml` como
    dependencia nueva solo para esto."""
    args = ["nvidia-smi", f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}",
            "--format=csv,noheader,nounits"]
    if gpu_index is not None:
        args += ["--id", str(gpu_index)]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != len(_NVIDIA_SMI_FIELDS):
        return None
    try:
        util, mem_util, power_w, sm_clock, temp = (float(p) for p in parts)
    except ValueError:
        return None
    return GpuFeatures(
        gpu_util_pct=util, gpu_mem_util_pct=mem_util,
        gpu_power_mw=power_w * 1000.0,  # nvidia-smi reporta power.draw en W
        gpu_sm_clock_mhz=sm_clock, gpu_temperature_c=temp,
    )


def _placeholder_classify(_features: GpuFeatures) -> GpuPhaseLabel:
    """NO ES UN CLASIFICADOR VÁLIDO -- lanza siempre. Existe únicamente para
    que quede un error explícito y ruidoso si alguien intenta correr el
    loop en producción sin haber inyectado `classify_fn` con el modelo real
    de fase2_clasificador/. Ningún valor por defecto silencioso: un
    clasificador placeholder que "funcionara" (p.ej. siempre compute_bound)
    fallaría en silencio exactamente en el escenario que el objetivo 3 del
    plan pide evitar."""
    raise NotImplementedError(
        "gpu_loop.run() requiere classify_fn: no existe todavía un "
        "clasificador de GPU entrenado (ver fase2_clasificador/README.md, "
        "limitaciones conocidas). Pásalo explícitamente -- p.ej. una función "
        "de prueba, o el modelo real una vez exista."
    )


def build_controller_from_policy(
    policy: dict[str, Any], min_dwell_ns: int, set_clock: Callable[[int], bool],
) -> GpuClockController:
    """Construye el GpuClockController a partir de la tabla de política ya
    derivada (fase3_daemon/policy/derive_policy_table.py) -- nunca
    hardcodea los MHz objetivo en el daemon (§3.5 paso 7 del plan: "nunca
    hardcodear esta tabla en el código del daemon").

    Si la política para una clase es "no_actuar" (incluyendo el caso real
    de hoy, T_transición_gpu no medido), el reloj objetivo de esa clase es
    0 -- interpretado aguas abajo por el llamador como "no bloquear el
    reloj", no como un MHz real a solicitar. Queda a criterio del llamador
    decidir si eso significa restaurar el gobernador nativo o simplemente
    no tocar la GPU en absoluto para esa clase.

    El reloj objetivo viene directamente de `resolved_clock_mhz`
    (mediana del reloj REAL observado en la campaña de barrido, no una
    resolución de fracción hecha aquí) -- ver
    `fase3_daemon/policy/derive_policy_table.py::_median_observed_frequency()`.
    La tabla de política es autocontenida a propósito: no requiere que el
    daemon tenga a mano el manifiesto de campaña que definió qué fracción
    correspondía a "F0" para poder aplicar la política.
    """
    def _target_mhz(key: str) -> int:
        entry = policy.get(key, {})
        if entry.get("action") != "actuar":
            return 0
        resolved = entry.get("resolved_clock_mhz")
        if resolved is None:
            raise ValueError(
                f"{key}: política 'actuar' sin 'resolved_clock_mhz' -- "
                "tabla de política incompleta o generada por una versión "
                "vieja de derive_policy_table.py"
            )
        return int(round(resolved))

    config = GpuClockControllerConfig(
        min_dwell_ns=min_dwell_ns,
        compute_bound_clock_mhz=_target_mhz("gpu-compute_bound"),
        memory_bound_clock_mhz=_target_mhz("gpu-memory_bound"),
    )
    return GpuClockController(config, set_clock)


def make_gpu_freqctl_setter(env: Any, gpu_index: int | str | None = None) -> Callable[[int], bool]:
    """Adaptador real de producción: `GpuClockController` espera
    `Callable[[int], bool]` (MHz -> éxito); `gpu_freqctl.apply_gpu_frequency`
    espera un `level` con `.mode`/`.fraction`. Se construye un nivel
    ``fixed`` sintético con la fracción exacta que da el MHz pedido sobre
    ``env.gpu_available_clocks_mhz`` -- reutiliza toda la lógica de
    relectura/verificación de `gpu_freqctl` en vez de reimplementarla.
    """
    from types import SimpleNamespace

    available = getattr(env, "gpu_available_clocks_mhz", None) or []

    def set_clock(mhz: int) -> bool:
        if not available:
            return False
        lo, hi = min(available), max(available)
        fraction = 1.0 if hi == lo else (mhz - lo) / (hi - lo)
        fraction = max(0.0, min(1.0, fraction))
        level = SimpleNamespace(id=f"gpu_loop_target_{mhz}mhz", mode="fixed", fraction=fraction)
        try:
            applied = gpu_freqctl.apply_gpu_frequency(level, env, gpu_index=gpu_index)
        except gpu_freqctl.GpuFrequencyControlError:
            return False
        return applied.applied_mhz == mhz or applied.strategy != gpu_freqctl.STRATEGY_UNAVAILABLE

    return set_clock


def run(
    phase_events: Iterable[PhaseBeginEvent],
    controller: GpuClockController,
    classify_fn: Callable[[GpuFeatures], GpuPhaseLabel] | None = None,
    *,
    on_decision: Callable[[PhaseBeginEvent, GpuPhaseLabel, GpuPhaseDecision], None] | None = None,
) -> list[GpuPhaseDecision]:
    """Consume `phase_events` uno a uno, clasifica cada uno y aplica la
    decisión del controller. Devuelve la lista de decisiones (para tests e
    inspección); en producción, `on_decision` es el punto de enganche del
    logging estructurado que pide §4.3 punto 10 del plan (features leídas,
    clase inferida, reloj aplicado, tiempo de inferencia/actuación).
    """
    classify = classify_fn or _placeholder_classify
    decisions: list[GpuPhaseDecision] = []
    for event in phase_events:
        label = classify(event.features)
        decision = controller.on_phase_begin(label, event.now_ns)
        decisions.append(decision)
        if on_decision is not None:
            on_decision(event, label, decision)
    return decisions


def now_ns() -> int:
    """Timestamp monotónico en ns, mismo reloj que usa el controller --
    envuelto para que los callers de producción no llamen a
    time.monotonic_ns() directamente en dos lugares distintos."""
    return time.monotonic_ns()
