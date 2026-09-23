#!/usr/bin/env python3
"""Punto de entrada de Fase 3 (Objetivo 3): daemon de control en espacio
de usuario.

⚠️ **Estado real, léase antes de usar**: este script arranca el loop de
GPU completo (sondeo de actividad NVML vía `gpu_loop/activity_poller.py`
+ clasificador real `gpu_loop/classifier.py::HistoricalGpuClassifier`
+ controller + actuación real vía `common.hpc.gpu_freqctl`) y el manejo de
señales/restauración combinada (§4.2/§4.3 punto 8). La fuente de eventos
de fase es sondeo, no intercepción de `cudaLaunchKernel` -- esa vía se
intentó y se retiró tras confirmar que no funciona para la sintaxis
`<<<>>>` de CUDA, ver `fase3_daemon/gpu_loop/activity_poller.py` para el
hallazgo completo. El loop de CPU (C++, con inferencia del modelo de
Fase 2 sobre el tick de ~1ms de `common/telemetry/collector.hpp`, §4.3
puntos 2/5) **no está integrado aquí todavía** -- su máquina de decisión
(`fase3_daemon/cpu_loop/include/cpu_phase_controller.hpp`) está construida
y probada, pero el binario real que la conecta con inferencia ONNX y el
harness C++ no se pudo construir en el entorno donde se hizo esta
reconstrucción (falta un modelo real entrenado -- el SDK C++ de ONNX
Runtime ya está disponible, ver `fase3_daemon/README.md`, limitaciones
conocidas). Correr este script hoy da el loop de GPU en vivo; el loop de
CPU debe lanzarse por separado en cuanto exista ese binario.

**Brazos del experimento de Fase 4** (`Plan_Fase3_Daemon.md` §0.1): `--dry-run`
es el brazo *sombra* -- clasifica y decide exactamente igual que en
producción, se detiene justo antes de escribir el reloj real. Sin esa
bandera es el brazo *activo*. El brazo *base* es, simplemente, no correr
este script. Hoy la política de `memory_bound` en GPU es la única
`actuar` (F1, +8.9% EDP), y el brazo *activo* para esa clase sigue
bloqueado por H1 (candado de reloj averiado) -- el resto del daemon
(clasificación, decisión, registro) es válido y puede probarse en *sombra*
sin esperar a que H1 se repare.

Modo (a) por defecto: opera sobre un cpuset/cgroup delegado (no descubre
ni delega el cpuset por sí solo -- eso lo hace el job de Slurm que lanza
este proceso, igual que `fase1_telemetria/campaign.py`) y corre mientras
dure la asignación, sin atarse a ningún proceso concreto. Modo (b)
(`--pid`, Bloque C ítem C7): ata el CICLO DE VIDA del loop a un PID
objetivo -- se detiene solo cuando ese proceso termina (`pid_alive()`,
verificado antes de arrancar y en cada iteración de sondeo), para pruebas
dirigidas contra un solo binario del catálogo (§4.3 punto 1). ⚠️ Esto NO
recorta las variables NVML a lo que ese proceso consume: `query_gpu_features()`
sigue siendo una lectura de TODO el dispositivo (mismo límite estructural
documentado para el clasificador GPU, ver `gpu_loop/classifier.py`) --
`--pid` en este script solo decide CUÁNDO parar, no filtra CUÁLES
muestras cuentan.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc import environment as environment_module  # noqa: E402
from common.hpc import freqctl, gpu_freqctl  # noqa: E402
from fase3_daemon.decision_log import ARMS, DecisionLogWriter, DecisionRecord  # noqa: E402
from fase3_daemon.gpu_loop import activity_poller  # noqa: E402
from fase3_daemon.gpu_loop.coordination import GpuActiveSignalWriter  # noqa: E402
from fase3_daemon.gpu_loop import loop as gpu_loop_module  # noqa: E402
from fase3_daemon.gpu_loop.classifier import HistoricalGpuClassifier  # noqa: E402

_DEFAULT_MODELS_DIR = _REPO_ROOT / "fase2_clasificador" / "models"

logger = logging.getLogger(__name__)


def pid_alive(pid: int) -> bool:
    """Comprueba si `pid` sigue vivo, sin enviar ninguna señal real
    (`kill(pid, 0)` -- el kernel solo valida permisos/existencia, POSIX
    estándar, mismo mecanismo que usa `psutil`/`os.kill` en cualquier
    daemon Unix). Wiring del modo (b) `--pid` (Bloque C ítem C7,
    Plan_Fase3_Daemon.md §4.3 punto 1): el loop de GPU se detiene solo
    cuando el proceso objetivo de la prueba dirigida termina, en vez de
    seguir sondeando NVML indefinidamente contra un PID muerto."""
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # El proceso existe pero pertenece a otro usuario -- no debería
        # pasar en la cuenta compartida de pacca (todo corre como
        # latorresn), pero "existe sin permiso" sigue siendo "vivo".
        return True
    return True


def _dry_run_setter(label: str):
    def set_clock(mhz: int) -> bool:
        logger.info("[dry-run] aplicaría reloj GPU -> %s MHz (%s)", mhz, label)
        return True
    return set_clock


def build_daemon_gpu_loop(
    policy_table_path: Path,
    *,
    gpu_index: int | str | None,
    min_dwell_ns: int,
    dry_run: bool,
    classify_fn,
    poll_interval_s: float = activity_poller.DEFAULT_POLL_INTERVAL_S,
    activity_threshold_pct: float = activity_poller.DEFAULT_ACTIVITY_THRESHOLD_PCT,
    query_features_fn=None,
    max_events: int | None = None,
    sleep_fn=time.sleep,
    on_sample=None,
    arm: str = "sombra",
    decision_log: DecisionLogWriter | None = None,
    should_continue=None,
    gpu_active_signal: GpuActiveSignalWriter | None = None,
    delegated_cpus: str = "0",
    gpu_settle_s: float | None = None,
    min_active_s: float = 0.0,
    on_active_start=None,
):
    """Ensambla el loop de GPU real a partir de la tabla de política ya
    derivada (§3.4/§3.5) -- nunca recalcula EDP en línea (§3.4 punto 4:
    "el daemon nunca recalcula el EDP... solo aplica la tabla ya derivada
    offline")."""
    if arm not in ARMS or arm == "base":
        raise ValueError(
            f"arm={arm!r} inválido para build_daemon_gpu_loop -- debe ser 'sombra' o 'activo' "
            "('base' significa, literalmente, no correr este script, ver Plan_Fase3_Daemon.md §0.1)"
        )
    if (arm == "sombra") != dry_run:
        raise ValueError(
            f"arm={arm!r} inconsistente con dry_run={dry_run!r} -- 'sombra' implica dry_run=True, "
            "'activo' implica dry_run=False (requisito 1 §0.1: sombra hace el mismo trabajo que "
            "activo y se detiene justo antes de escribir)"
        )

    policy_doc = yaml.safe_load(policy_table_path.read_text())
    policy = policy_doc["policy"]

    if dry_run:
        set_clock = _dry_run_setter("gpu")
    else:
        # detect_environment exige `delegated_cpus` (posicional). Antes se llamaba sin argumentos, lo que
        # habria lanzado TypeError al arrancar el brazo 'activo': la prueba lo enmascaraba con un
        # `lambda: fake_env` que aceptaba cero argumentos. El loop de GPU no toca CPU: el valor solo sirve
        # para que la deteccion de entorno tenga un cpuset que leer.
        env = environment_module.detect_environment(delegated_cpus)
        set_clock = gpu_loop_module.make_gpu_freqctl_setter(env, gpu_index=gpu_index, settle_s=gpu_settle_s)
        _install_restore_handlers(env, gpu_index)

    controller = gpu_loop_module.build_controller_from_policy(policy, min_dwell_ns, set_clock)

    def on_decision(event, label, decision):
        if gpu_active_signal is not None:
            gpu_active_signal.write(True)
        logger.info(
            "fase GPU: label=%s target_mhz=%s applied_mhz=%s changed=%s dwell_remaining_ns=%s",
            label.value, decision.target_clock_mhz, decision.applied_clock_mhz,
            decision.clock_changed, decision.dwell_remaining_ns,
        )
        if decision_log is not None:
            decision_log.write(DecisionRecord(
                ts_ns=event.now_ns,
                arm=arm,
                device="gpu",
                label=label.value,
                confidence=None,  # HistoricalGpuClassifier.classify() no expone proba, ver su docstring
                features={
                    "gpu_util_pct": event.features.gpu_util_pct,
                    "gpu_mem_util_pct": event.features.gpu_mem_util_pct,
                    "gpu_power_mw": event.features.gpu_power_mw,
                    "gpu_sm_clock_mhz": event.features.gpu_sm_clock_mhz,
                    "gpu_temperature_c": event.features.gpu_temperature_c,
                },
                policy_action="actuar" if decision.target_clock_mhz else "no_actuar",
                target_freq_khz=decision.target_clock_mhz * 1000,
                applied_freq_khz=decision.applied_clock_mhz * 1000,
                written=decision.clock_changed and not decision.clock_setter_failed and not dry_run,
                write_failed=decision.clock_changed and decision.clock_setter_failed,
                inference_time_ns=decision.inference_time_ns,
                actuation_time_ns=decision.actuation_time_ns,
                extra={"dwell_remaining_ns": decision.dwell_remaining_ns},
            ))

    def on_end(now_ns: int) -> None:
        if gpu_active_signal is not None:
            gpu_active_signal.write(False)
        logger.debug("fin de fase GPU en t=%sns", now_ns)

    def _active_start(now_ns: int) -> None:
        # La senal de coordinacion CPU-GPU sube al INICIO de la actividad, no al tomar la decision (que
        # con min_active_s > 0 llega segundos despues): el piso de CPU debe proteger la fase desde el arranque.
        if gpu_active_signal is not None:
            gpu_active_signal.write(True)
        if on_active_start is not None:
            on_active_start(now_ns)

    query_fn = query_features_fn or (lambda: gpu_loop_module.query_gpu_features(gpu_index))
    events = activity_poller.poll_phase_events(
        query_fn, poll_interval_s=poll_interval_s,
        activity_threshold_pct=activity_threshold_pct, on_end=on_end,
        max_events=max_events, sleep_fn=sleep_fn, on_sample=on_sample,
        should_continue=should_continue, min_active_s=min_active_s, on_active_start=_active_start,
    )
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
                         help="policy_table.yaml producido por fase3_daemon/policy/build_policy_table.py")
    parser.add_argument("--gpu-index", default=None)
    parser.add_argument("--min-active-s", type=float, default=3.0,
                         help="La clase de cada fase de GPU se decide tras esta ventana de actividad SOSTENIDA, con "
                              "mediana/std solo sobre esa fase (default 3 s). 0 = decidir en el flanco de subida "
                              "con una muestra instantanea (medido: acierta 2 de 6, job 7606). Retrasa la "
                              "actuacion esa misma cantidad.")
    parser.add_argument("--gpu-settle-s", type=float, default=None,
                         help="Espera de asentamiento de NVML dentro de cada cambio de reloj (default: la de gpu_freqctl, "
                              "1.5 s). Medido (job 7599): el comando cuesta ~50 ms y el reloj llega ~80 ms despues; un "
                              "valor menor reduce el costo por cambio, pero es decision explicita.")
    parser.add_argument("--delegated-cpus", default="0",
                         help="cpuset que lee detect_environment() (solo lectura). El loop de GPU no escribe en CPU; "
                              "el valor solo tiene que existir en el nodo (default '0').")
    parser.add_argument("--min-dwell-ns", type=int, required=True,
                         help="Piso de permanencia de reloj GPU (§2.4.1) -- debe venir de T_transición_gpu "
                              "MEDIDO, nunca de un valor arbitrario. No hay default a propósito.")
    parser.add_argument("--poll-interval-s", type=float, default=activity_poller.DEFAULT_POLL_INTERVAL_S,
                         help="Cadencia de sondeo NVML para detectar fronteras de fase "
                              f"(default {activity_poller.DEFAULT_POLL_INTERVAL_S}s, ver gpu_loop/activity_poller.py).")
    parser.add_argument("--activity-threshold-pct", type=float,
                         default=activity_poller.DEFAULT_ACTIVITY_THRESHOLD_PCT,
                         help="Umbral de gpu_util_pct para considerar la GPU activa "
                              f"(default {activity_poller.DEFAULT_ACTIVITY_THRESHOLD_PCT}%%).")
    parser.add_argument("--mode", choices=["cpuset", "pid"], default="cpuset",
                         help="'cpuset' (default): corre mientras dure la asignación de Slurm, sin "
                              "atarse a ningún proceso. 'pid': ata el ciclo de vida del loop a --pid -- "
                              "se detiene solo cuando ese proceso termina (Bloque C C7). No recorta las "
                              "variables NVML al proceso (siguen siendo del dispositivo completo).")
    parser.add_argument("--pid", type=int, default=None,
                         help="Requerido si --mode pid (§4.3 punto 1, modo de prueba dirigida).")
    parser.add_argument("--arm", choices=["sombra", "activo"], required=True,
                         help="Brazo de primera clase del experimento de Fase 4 (Plan_Fase3_Daemon.md SS0.1, "
                              "requisito 1). 'sombra': clasifica y decide exactamente igual que 'activo', pero "
                              "se detiene justo antes de escribir el reloj real. 'activo': escribe de verdad. "
                              "El brazo 'base' es, literalmente, no correr este script. Sin default: elegir el "
                              "brazo a propósito, nunca por accidente.")
    parser.add_argument("--log-path", type=Path, default=None,
                         help="Ruta del registro JSONL de decisiones (requisito 2 SS0.1) -- una linea por fase "
                              "de GPU, mismo esquema que el lado CPU (decision_log.py/decision_log.hpp). Si se "
                              "omite, no se escribe registro estructurado (solo el log de texto habitual).")
    parser.add_argument("--gpu-active-signal-path", type=Path, default=None,
                         help="Ruta del archivo de senal de coordinacion CPU-GPU (Bloque C, item C4) -- este "
                              "loop escribe '1'/'0' atomicamente en cada transicion de fase; cpu_loop_main "
                              "(C++, proceso separado) lo lee cada tick via --gpu-active-signal-path propio. Si "
                              "se omite, no se escribe (cpu_loop_main sigue viendo gpu_active=false, el default "
                              "seguro de antes de que esta senal existiera).")
    parser.add_argument("--models-dir", type=Path, default=_DEFAULT_MODELS_DIR,
                         help=f"Directorio con {{nombre}}.joblib + {{nombre}}.metadata.json del candidato GPU "
                              f"(default: {_DEFAULT_MODELS_DIR}).")
    parser.add_argument("--model-name", default="gpu_random_forest_historical_20260923_sin_reloj",
                         help="Nombre base del candidato exportado (sin extensión).")
    parser.add_argument("--classifier-window", type=int, default=None,
                         help="Muestras NVML del buffer movil para mediana/std (default: "
                              "HistoricalGpuClassifier.DEFAULT_WINDOW_SIZE). Ver gpu_loop/classifier.py "
                              "para la discusion de por que es una aproximacion causal.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "pid" and args.pid is None:
        parser.error("--mode pid requiere --pid")

    should_continue = None
    if args.mode == "pid":
        if not pid_alive(args.pid):
            parser.error(f"--mode pid: el proceso {args.pid} no existe (ya terminó antes de arrancar)")
        should_continue = lambda: pid_alive(args.pid)  # noqa: E731
        logger.info("modo pid: el loop se detendrá cuando el proceso %d termine", args.pid)

    logger.warning(
        "run_daemon.py: el loop de CPU (C++, inferencia sobre collector.hpp) no está "
        "integrado en este script todavía -- ver el docstring del módulo y "
        "fase3_daemon/README.md. Arrancando solo el loop de GPU."
    )

    classifier_kwargs = {}
    if args.classifier_window is not None:
        classifier_kwargs["window_size"] = args.classifier_window
    classifier = HistoricalGpuClassifier.from_export_dir(
        args.models_dir, name=args.model_name, **classifier_kwargs,
    )
    logger.info(
        "clasificador GPU cargado: %s (variables=%s, brazo=%s)",
        args.model_name, classifier._feature_names, args.arm,
    )

    decision_log = DecisionLogWriter(args.log_path) if args.log_path is not None else None
    gpu_active_signal = (
        GpuActiveSignalWriter(args.gpu_active_signal_path)
        if args.gpu_active_signal_path is not None else None
    )
    try:
        build_daemon_gpu_loop(
            args.policy_table,
            gpu_index=args.gpu_index,
            min_dwell_ns=args.min_dwell_ns,
            dry_run=(args.arm == "sombra"),
            classify_fn=classifier.classify,
            on_sample=classifier.record_sample,
            poll_interval_s=args.poll_interval_s,
            activity_threshold_pct=args.activity_threshold_pct,
            arm=args.arm,
            decision_log=decision_log,
            should_continue=should_continue,
            gpu_active_signal=gpu_active_signal,
            delegated_cpus=args.delegated_cpus,
            gpu_settle_s=args.gpu_settle_s,
            min_active_s=args.min_active_s,
            on_active_start=classifier.reset_window,
        )
    except KeyboardInterrupt:
        logger.info("interrumpido, saliendo")
        return 130
    finally:
        if decision_log is not None:
            decision_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
