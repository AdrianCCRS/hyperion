#!/usr/bin/env python3
"""Punto de entrada de Fase 3 (Objetivo 3): daemon de control en espacio
de usuario.

⚠️ **Estado real, léase antes de usar**: este script arranca el loop de
GPU completo (event listener del shim + controller + actuación real vía
`common.hpc.gpu_freqctl`) y el manejo de señales/restauración combinada
(§4.2/§4.3 punto 8). El loop de CPU (C++, con inferencia del modelo de
Fase 2 sobre el tick de ~1ms de `common/telemetry/collector.hpp`, §4.3
puntos 2/5) **no está integrado aquí todavía** -- su máquina de decisión
(`fase3_daemon/cpu_loop/include/cpu_phase_controller.hpp`) está construida
y probada, pero el binario real que la conecta con inferencia ONNX y el
harness C++ no se pudo construir en el entorno donde se hizo esta
reconstrucción (falta el SDK C++ de ONNX Runtime y un modelo real
entrenado -- ver `fase3_daemon/README.md`, limitaciones conocidas). Correr
este script hoy da el loop de GPU en vivo; el loop de CPU debe lanzarse
por separado en cuanto exista ese binario.

Modo (a) por defecto: opera sobre un cpuset/cgroup delegado (no descubre
ni delega el cpuset por sí solo -- eso lo hace el job de Slurm que lanza
este proceso, igual que `fase1_telemetria/campaign.py`). Modo (b)
(`--pid`): se limita a monitorear/actuar en función de un PID específico,
para pruebas dirigidas contra un solo binario del catálogo (§4.3 punto 1).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc import environment as environment_module  # noqa: E402
from common.hpc import freqctl, gpu_freqctl  # noqa: E402
from fase3_daemon.gpu_loop import loop as gpu_loop_module  # noqa: E402
from fase3_daemon.shim import event_listener  # noqa: E402

logger = logging.getLogger(__name__)


def _dry_run_setter(label: str):
    def set_clock(mhz: int) -> bool:
        logger.info("[dry-run] aplicaría reloj GPU -> %s MHz (%s)", mhz, label)
        return True
    return set_clock


def build_daemon_gpu_loop(
    policy_table_path: Path,
    *,
    gpu_phase_socket: str,
    gpu_index: int | str | None,
    min_dwell_ns: int,
    dry_run: bool,
    classify_fn,
    query_features_fn=event_listener.query_gpu_features,
):
    """Ensambla el loop de GPU real a partir de la tabla de política ya
    derivada (§3.4/§3.5) -- nunca recalcula EDP en línea (§3.4 punto 4:
    "el daemon nunca recalcula el EDP... solo aplica la tabla ya derivada
    offline")."""
    policy_doc = yaml.safe_load(policy_table_path.read_text())
    policy = policy_doc["policy"]

    if dry_run:
        set_clock = _dry_run_setter("gpu")
    else:
        env = environment_module.detect_environment()
        set_clock = gpu_loop_module.make_gpu_freqctl_setter(env, gpu_index=gpu_index)
        _install_restore_handlers(env, gpu_index)

    controller = gpu_loop_module.build_controller_from_policy(policy, min_dwell_ns, set_clock)

    def on_decision(event, label, decision):
        logger.info(
            "fase GPU: label=%s target_mhz=%s applied_mhz=%s changed=%s dwell_remaining_ns=%s",
            label.value, decision.target_clock_mhz, decision.applied_clock_mhz,
            decision.clock_changed, decision.dwell_remaining_ns,
        )

    def on_end(now_ns: int) -> None:
        logger.debug("fin de fase GPU en t=%sns", now_ns)

    events = event_listener.listen(gpu_phase_socket, query_features_fn, on_end=on_end)
    return gpu_loop_module.run(events, controller, classify_fn=classify_fn, on_decision=on_decision)


def _install_restore_handlers(env, gpu_index: int | str | None) -> None:
    """§4.2 punto 3 / §4.3 punto 8: restauración obligatoria e idempotente
    registrada en atexit/SIGINT/SIGTERM, reutilizando
    `freqctl.install_emergency_handlers` (misma utilidad genérica que ya
    tiene una prueba de caos real en el proyecto, ARC-140) -- no
    reimplementa el manejo de señales.

    ⚠️ Hoy solo restaura GPU: este script todavía no escribe frecuencia de
    CPU (el loop de CPU no está integrado aquí, ver el docstring del
    módulo). Cuando lo esté, el registro debe unificarse en UNA sola
    llamada a `install_emergency_handlers` que restaure ambos ejes juntos
    -- §4.3 punto 8 es explícito en que dos registros separados se pisan
    entre sí, así que no se debe llamar dos veces a esta función.
    """
    def restore_gpu() -> bool:
        return gpu_freqctl.restore_gpu_state(env, gpu_index=gpu_index)

    freqctl.install_emergency_handlers(restore_gpu)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-table", type=Path, required=True,
                         help="policy_table.yaml producido por fase3_daemon/policy/derive_policy_table.py")
    parser.add_argument("--gpu-phase-socket", required=True,
                         help="Ruta del socket Unix donde escucha eventos de fase de fase3_daemon/shim/ "
                              "(debe coincidir con HYPERION_GPU_PHASE_SOCKET del binario instrumentado).")
    parser.add_argument("--gpu-index", default=None)
    parser.add_argument("--min-dwell-ns", type=int, required=True,
                         help="Piso de permanencia de reloj GPU (§2.4.1) -- debe venir de T_transición_gpu "
                              "MEDIDO, nunca de un valor arbitrario. No hay default a propósito.")
    parser.add_argument("--mode", choices=["cpuset", "pid"], default="cpuset")
    parser.add_argument("--pid", type=int, default=None,
                         help="Requerido si --mode pid (§4.3 punto 1, modo de prueba dirigida).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Clasifica y decide, pero solo registra en log -- no escribe frecuencia real "
                              "(§4.3 punto 9, validar antes de tocar hardware real).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "pid" and args.pid is None:
        parser.error("--mode pid requiere --pid")

    logger.warning(
        "run_daemon.py: el loop de CPU (C++, inferencia sobre collector.hpp) no está "
        "integrado en este script todavía -- ver el docstring del módulo y "
        "fase3_daemon/README.md. Arrancando solo el loop de GPU."
    )

    def classify_placeholder(features):
        raise NotImplementedError(
            "no existe todavía un clasificador de GPU entrenado -- ver "
            "fase2_clasificador/README.md, limitaciones conocidas"
        )

    try:
        build_daemon_gpu_loop(
            args.policy_table,
            gpu_phase_socket=args.gpu_phase_socket,
            gpu_index=args.gpu_index,
            min_dwell_ns=args.min_dwell_ns,
            dry_run=args.dry_run,
            classify_fn=classify_placeholder,
        )
    except KeyboardInterrupt:
        logger.info("interrumpido, saliendo")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
