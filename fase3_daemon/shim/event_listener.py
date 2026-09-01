"""Extremo Python del canal de eventos de fase que emite
fase3_daemon/shim/blocking_sync_shim.cpp -- recibe los datagramas
"BEGIN,<ns>"/"END,<ns>" por el mismo socket de dominio Unix
(HYPERION_GPU_PHASE_SOCKET) y los convierte en
fase3_daemon.gpu_loop.loop.PhaseBeginEvent, consultando NVML en vivo en
cada BEGIN.

A diferencia del resto del proyecto, la parte de socket real (`listen()`)
no se pudo probar contra el shim de verdad -- ese shim no se pudo compilar
en este entorno sin CUDA toolkit (ver fase3_daemon/shim/blocking_sync_shim.cpp).
Sí se probó de punta a punta con un socket Unix real y un cliente que envía
exactamente los mismos mensajes que el shim (ver
fase3_daemon/tests/test_event_listener.py) -- la parte no verificada es
específicamente la interposición CUDA, no el protocolo de socket en sí.

Consulta de features NVML en vivo: por consistencia con el resto del
proyecto (`common/hpc/gpu_freqctl.py`, `common/hpc/gpu_inspector.py`, que
ya usan subprocess sobre `nvidia-smi` en vez de un binding NVML de Python),
`query_gpu_features()` hace lo mismo -- no introduce `pynvml` como
dependencia nueva solo para este módulo.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.gpu_loop.loop import GpuFeatures, PhaseBeginEvent  # noqa: E402

_NVIDIA_SMI_FIELDS = (
    "utilization.gpu", "utilization.memory", "power.draw", "clocks.sm", "temperature.gpu",
)


def query_gpu_features(gpu_index: int | str | None = None) -> GpuFeatures | None:
    """Snapshot NVML en vivo vía `nvidia-smi`, en el mismo orden que
    `_NVIDIA_SMI_FIELDS`. Devuelve None si la consulta falla (GPU no
    disponible, `nvidia-smi` no encontrado) -- el llamador decide qué
    hacer con un evento sin features (típicamente: descartarlo, nunca
    inventar un valor)."""
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


def parse_message(raw: bytes) -> tuple[str, int] | None:
    """Parsea un datagrama "KIND,<ns>\\n" del shim. None si no matchea el
    formato esperado -- un mensaje corrupto/parcial se descarta, nunca
    lanza (este parser corre en el loop principal del listener)."""
    try:
        text = raw.decode("ascii").strip()
        kind, ns_str = text.split(",", 1)
        if kind not in ("BEGIN", "END"):
            return None
        return kind, int(ns_str)
    except (UnicodeDecodeError, ValueError):
        return None


def listen(
    socket_path: str,
    query_features_fn: Callable[[], GpuFeatures | None] = query_gpu_features,
    *,
    on_end: Callable[[int], None] | None = None,
    max_events: int | None = None,
) -> Iterator[PhaseBeginEvent]:
    """Escucha `socket_path` (socket de dominio Unix, datagrama -- mismo
    tipo que usa el shim en C++) y genera un `PhaseBeginEvent` por cada
    "BEGIN" recibido con features NVML válidas. Los eventos "END" se
    reportan vía `on_end` (para logging/duración de fase) pero no generan
    `PhaseBeginEvent` -- el controller solo actúa en fronteras de inicio.

    `max_events` es solo para pruebas (detiene el loop tras N eventos BEGIN
    exitosos); en producción se deja en None y el generador corre
    indefinidamente hasta que el proceso del daemon termine.
    """
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    try:
        emitted = 0
        while max_events is None or emitted < max_events:
            raw, _addr = server.recvfrom(256)
            parsed = parse_message(raw)
            if parsed is None:
                continue
            kind, ns = parsed
            if kind == "END":
                if on_end is not None:
                    on_end(ns)
                continue
            features = query_features_fn()
            if features is None:
                # Sin snapshot NVML válido, no se puede clasificar esta
                # fase -- se descarta el evento en vez de inventar features.
                continue
            emitted += 1
            yield PhaseBeginEvent(now_ns=ns, features=features)
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
