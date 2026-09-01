"""Detección de fase de GPU por sondeo de actividad NVML (Opción C).

Reemplaza la intercepción de `cudaLaunchKernel` vía `LD_PRELOAD`
(`fase3_daemon/shim/`, eliminado) tras un hallazgo confirmado: esa
intercepción no se dispara para kernels lanzados con la sintaxis estándar
`<<<>>>` en ningún modo de enlace de cudart -- `nvcc` resuelve esa llamada
en tiempo de compilación/enlace, nunca a través de la tabla de símbolos
dinámicos que `LD_PRELOAD` puede alterar (verificado compilando el shim
contra CUDA 13.3 real y cargándolo contra un kernel de prueba real: cero
eventos, confirmado con `nm -D` y una build de depuración).

Elegida sobre las otras dos vías evaluadas (documentadas como trabajo
futuro en `fase3_daemon/README.md`, no implementadas):
  (a) Interceptar a nivel de driver CUDA (`cuLaunchKernel` de `libcuda.so`)
      -- probablemente con el mismo problema, porque cudart resuelve esa
      llamada internamente vía `dlsym()` sobre un handle propio, no a
      través de la tabla de símbolos global que `LD_PRELOAD` altera.
  (b) `CUDA_INJECTION64_PATH` + CUPTI callback API -- el mecanismo oficial
      de NVIDIA para esto, más robusto (funciona sin importar cómo se
      compiló el binario objetivo) pero mucho más pesado de implementar
      (SDK adicional, dominios de callback, gestión de suscriptores).

Esta vía (C) no requiere ninguna instrumentación del binario objetivo --
observa el estado de la GPU desde el proceso del daemon mismo, vía NVML
(reutilizando `query_gpu_features()`, ya construido y probado). A cambio
de esa simplicidad, la frontera de fase no es instantánea: se detecta con
la latencia de `poll_interval_s`, no en el instante exacto del primer
`cudaLaunchKernel`. Esto es aceptable dado el propio diseño de
`gpu_clock_controller.hpp`: ya trabaja a granularidad de fase con
histéresis (`min_dwell_ns`), pensado para no reaccionar al instante --
la latencia de sondeo es pequeña frente al costo real de una transición
de reloj de GPU (T_transición_gpu, §2.4.1, todavía sin medir).
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.gpu_loop.loop import GpuFeatures, PhaseBeginEvent  # noqa: E402

# "Umbral de ruido" (§4.1 del plan, "Señal de coordinación": "el loop de
# GPU la activa mientras gpu_util_pct esté por encima de un umbral de
# ruido") -- mismo concepto que la señal de coordinación CPU-GPU, ahora
# también la base de la detección de fase misma.
DEFAULT_ACTIVITY_THRESHOLD_PCT = 5.0
DEFAULT_POLL_INTERVAL_S = 0.05  # 50 ms


def poll_phase_events(
    query_features_fn: Callable[[], GpuFeatures | None],
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    activity_threshold_pct: float = DEFAULT_ACTIVITY_THRESHOLD_PCT,
    now_fn: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_end: Callable[[int], None] | None = None,
    max_events: int | None = None,
) -> Iterator[PhaseBeginEvent]:
    """Sondea `query_features_fn()` cada `poll_interval_s` segundos y genera
    un `PhaseBeginEvent` en cada transición idle -> activo
    (`gpu_util_pct` cruza `activity_threshold_pct` de abajo hacia arriba).
    La transición activo -> idle se reporta vía `on_end` (para logging de
    duración de fase), no genera un `PhaseBeginEvent` -- mismo contrato que
    tenía el listener de eventos del shim (§4.1: el controller solo actúa
    en fronteras de inicio).

    `query_features_fn` devolviendo `None` (NVML no disponible momentáneamente)
    no cambia el estado interno -- se trata como "sin señal esta muestra",
    nunca se fabrica una transición con datos ausentes.

    `max_events` es solo para pruebas (detiene el generador tras N eventos
    de inicio); en producción se deja en `None` y el generador corre
    indefinidamente. `now_fn`/`sleep_fn` son inyectables para poder probar
    sin reloj real ni esperas reales.
    """
    is_active = False
    emitted = 0
    while max_events is None or emitted < max_events:
        features = query_features_fn()
        if features is not None:
            active_now = features.gpu_util_pct > activity_threshold_pct
            now = now_fn()
            if active_now and not is_active:
                is_active = True
                emitted += 1
                yield PhaseBeginEvent(now_ns=now, features=features)
            elif not active_now and is_active:
                is_active = False
                if on_end is not None:
                    on_end(now)
        sleep_fn(poll_interval_s)
