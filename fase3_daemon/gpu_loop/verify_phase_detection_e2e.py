#!/usr/bin/env python3
"""Bloque C3 (Plan_Fase3_Daemon.md §0.1): verifica de punta a punta que
`activity_poller.poll_phase_events()` detecta una fase de actividad real
de GPU -- contra un kernel real de terceros lanzado como subproceso
(checksum verificado, mismo rigor que `fase3_daemon/composite_apps`), NO
contra `GpuFeatures` sintéticas como hacen todos los tests unitarios
existentes de `test_activity_poller.py`.

Hasta ahora la detección de fase (Opción C, sondeo NVML) solo se había
probado con datos inyectados -- nunca contra hardware real, un kernel CUDA
real y el sondeo real vía `nvidia-smi`. Este script cierra esa brecha.

Mecánica: un hilo aparte corre `poll_phase_events()` con
`query_gpu_features()` real (mismo `common/hpc` que usa `run_daemon.py`),
mientras el hilo principal lanza el kernel como subproceso y mide su
inicio/fin real con reloj monotónico. Se arranca el sondeo ANTES de
lanzar el kernel (para no perder la transición idle->activo) y se lo deja
correr un margen después de que el kernel termina (para capturar la
transición activo->idle, que llega con la latencia de `poll_interval_s`).
Reutiliza `should_continue` (Bloque C, ítem C7) para detener el generador
de forma limpia en vez de un `for` con `break` sobre un iterador infinito.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc.catalog import KernelEntry, load_catalog, verify_binary  # noqa: E402
from fase3_daemon.gpu_loop.activity_poller import poll_phase_events  # noqa: E402
from fase3_daemon.gpu_loop.loop import PhaseBeginEvent, query_gpu_features  # noqa: E402

DEFAULT_CATALOG_PATH = _REPO_ROOT / "fase1_telemetria" / "catalog" / "catalog.yaml"
DEFAULT_KERNEL_ID = "gpu_dgemm_n4096"


@dataclass
class E2EResult:
    process_begin_ns: int
    process_end_ns: int
    begin_events: list[PhaseBeginEvent]
    end_ns_values: list[int]
    success: bool
    exit_code: int

    @property
    def detected_begin(self) -> bool:
        return len(self.begin_events) >= 1

    @property
    def detected_end(self) -> bool:
        return len(self.end_ns_values) >= 1

    @property
    def begin_latency_ns(self) -> int | None:
        """Cuánto tardó el sondeo en detectar el inicio real del kernel,
        respecto al instante en que se lanzó el subproceso. Puede ser
        negativo (el kernel tarda en subir su gpu_util_pct por encima del
        umbral -- el sondeo detecta la actividad DESPUÉS de que el proceso
        arrancó, nunca antes)."""
        if not self.begin_events:
            return None
        return self.begin_events[0].now_ns - self.process_begin_ns

    @property
    def end_latency_ns(self) -> int | None:
        if not self.end_ns_values:
            return None
        return self.end_ns_values[0] - self.process_end_ns


def _check_success(entry: KernelEntry, exit_code: int, stdout: str) -> bool:
    """Mismo criterio que `composite_apps/composite_known.py::_check_success`
    (RUN-05) -- ver ese módulo para por qué se reimplementa en memoria en
    vez de importar de `fase1_telemetria/runner.py`."""
    import re
    check = entry.success_check
    check_type = check.get("type")
    if check_type == "exit_code":
        return exit_code == check.get("expected", 0)
    if check_type == "stdout_regex":
        return re.search(check["pattern"], stdout) is not None
    return False


def run_e2e(
    entry: KernelEntry,
    *,
    node_id: str,
    kernels_root: Path,
    gpu_index: int | str | None,
    poll_interval_s: float,
    activity_threshold_pct: float,
    end_margin_s: float,
    poller_warmup_s: float = 1.0,
    query_features_fn=None,
    run_fn=subprocess.run,
    sleep_fn=time.sleep,
    now_fn=time.monotonic_ns,
    monotonic_fn=time.monotonic,
) -> E2EResult:
    """`query_features_fn`/`run_fn`/`sleep_fn`/`now_fn`/`monotonic_fn` son
    inyectables a propósito -- para poder probar la mecánica de
    coordinación entre hilos (arrancar el sondeo antes del kernel,
    detenerlo con margen después) sin GPU ni subprocesos reales (ver
    `tests/test_verify_phase_detection_e2e.py`, corrible en cualquier
    máquina). En producción, `query_features_fn` es
    `lambda: query_gpu_features(gpu_index)` (NVML real vía `nvidia-smi`) y
    `run_fn` es `subprocess.run` sobre el binario real."""
    resolved_exec_path = str(kernels_root / entry.exec_path)
    if not verify_binary(replace(entry, exec_path=resolved_exec_path), node_id=node_id):
        raise RuntimeError(
            f"C02: {entry.id!r} no pasó la verificación de checksum en node_id={node_id!r} "
            f"(exec_path resuelto: {resolved_exec_path})"
        )
    query_fn = query_features_fn or (lambda: query_gpu_features(gpu_index))

    begin_events: list[PhaseBeginEvent] = []
    end_ns_values: list[int] = []
    stop_polling_at = {"deadline": None}  # float | None, en la escala de monotonic_fn()

    def should_continue() -> bool:
        deadline = stop_polling_at["deadline"]
        return deadline is None or monotonic_fn() < deadline

    def poll() -> None:
        for event in poll_phase_events(
            query_fn, poll_interval_s=poll_interval_s, activity_threshold_pct=activity_threshold_pct,
            on_end=end_ns_values.append, should_continue=should_continue,
            now_fn=now_fn, sleep_fn=sleep_fn,
        ):
            begin_events.append(event)

    poller_thread = threading.Thread(target=poll, daemon=True)
    poller_thread.start()
    # Deja que el sondeo tome al menos una lectura idle real ANTES de
    # lanzar el kernel -- si arrancan a la vez, el primer PhaseBeginEvent
    # podría perderse por una carrera con la primera lectura del poller.
    sleep_fn(poller_warmup_s)

    argv = [resolved_exec_path, *shlex.split(entry.exec_args)]
    begin_ns = now_fn()
    proc = run_fn(argv, cwd=str(kernels_root), capture_output=True, text=True)
    end_ns = now_fn()

    # Deja correr el sondeo end_margin_s mas para capturar la transicion
    # activo->idle (llega con latencia de poll_interval_s tras el fin real).
    stop_polling_at["deadline"] = monotonic_fn() + end_margin_s
    # join() es threading real -- siempre con timeout de reloj de PARED
    # real (nunca inyectable), pero con sleep_fn/monotonic_fn falsos en
    # pruebas el hilo termina casi de inmediato de todos modos.
    poller_thread.join(timeout=end_margin_s + poll_interval_s * 4 + 5.0)

    return E2EResult(
        process_begin_ns=begin_ns, process_end_ns=end_ns,
        begin_events=begin_events, end_ns_values=end_ns_values,
        success=_check_success(entry, proc.returncode, proc.stdout), exit_code=proc.returncode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--kernel-id", default=DEFAULT_KERNEL_ID)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--kernels-root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-index", default=None)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--activity-threshold-pct", type=float, default=5.0)
    parser.add_argument("--end-margin-s", type=float, default=2.0)
    args = parser.parse_args()

    catalog = load_catalog(str(args.catalog))
    if args.kernel_id not in catalog:
        print(f"kernel {args.kernel_id!r} no está en el catálogo", file=sys.stderr)
        return 2
    entry = catalog[args.kernel_id]
    if entry.device != "gpu":
        print(f"kernel {args.kernel_id!r} no es device=gpu (es {entry.device!r})", file=sys.stderr)
        return 2

    print(f"lanzando {entry.id} (runtime esperado ~{entry.expected_runtime_seconds}s) "
          f"con sondeo NVML real cada {args.poll_interval_s}s...")
    result = run_e2e(
        entry, node_id=args.node_id, kernels_root=args.kernels_root, gpu_index=args.gpu_index,
        poll_interval_s=args.poll_interval_s, activity_threshold_pct=args.activity_threshold_pct,
        end_margin_s=args.end_margin_s,
    )

    def _fmt_ms(ns: int | None) -> str:
        return f"{ns / 1e6:.1f}ms" if ns is not None else "n/a"

    process_duration_s = (result.process_end_ns - result.process_begin_ns) / 1e9
    print(f"proceso: éxito={result.success} exit_code={result.exit_code} duración={process_duration_s:.2f}s")
    print(f"fase detectada (inicio): {result.detected_begin} "
          f"(n={len(result.begin_events)}, latencia={_fmt_ms(result.begin_latency_ns)})")
    print(f"fase detectada (fin): {result.detected_end} "
          f"(n={len(result.end_ns_values)}, latencia={_fmt_ms(result.end_latency_ns)})")

    ok = result.success and result.detected_begin and result.detected_end
    print(f"verificación C3 (detección de fase de GPU de punta a punta): {'OK' if ok else 'FALLÓ'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
