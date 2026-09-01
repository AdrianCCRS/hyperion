#!/usr/bin/env bash
# Punto de entrada raíz: lanza las 4 fases en secuencia si así se quiere.
# Cada fase sigue siendo invocable sola vía su propio run_*.py -- este
# script es una conveniencia, no una capa nueva de lógica: delega en los
# mismos launchers documentados en cada README de fase.
#
# ⚠️ Estado real (ver cada README de fase para el detalle completo): Fase 1
# y Fase 2 corren de punta a punta. Fase 3 no tiene todavía el loop de CPU
# real (falta el SDK C++ de ONNX Runtime + un modelo entrenado real) ni el
# shim CUDA compilado (este entorno de reconstrucción no tiene el CUDA
# toolkit) -- este script arranca el daemon en --dry-run si se le pide.
# Fase 4 genera el reporte a partir de datos ya producidos, no orquesta
# corridas nuevas automáticamente. Ver fase3_daemon/README.md y
# fase4_evaluacion/README.md para el procedimiento manual completo mientras
# esas piezas se completan.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HYPERION_PYTHON:-python3}"

log() { printf '[run_all] %s\n' "$*" >&2; }
fail() { printf '[run_all] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Uso: ./run_all.sh <comando> [opciones]

Comandos:
  check-readiness              Chequeo de solo lectura de permisos (perf/RAPL/NVML/cpuset).
  fase1 -- <args>               fase1_telemetria/run_campaign.py (ver --help propio).
  fase2 -- <args>               fase2_clasificador/run_training.py.
  fase3 -- <args>               fase3_daemon/run_daemon.py.
  fase4 -- <args>               fase4_evaluacion/run_evaluation.py.
  all --manifest M --node-id N --reference-kernel-ref K --campaign-dir D --campaign-id C
                                Fase 1 (run-campaign) -> Fase 2 (entrenar+serializar,
                                lee el output de Fase 1) -> avisa qué falta para Fase 3/4
                                en vez de fingir que también corrieron.
  test                          pytest de las 4 fases + common/, y ctest de las
                                partes en C++ que sí compilan en este entorno.

Cada fase también es invocable directamente, sin este script -- ver el
README.md de cada una para el uso completo:
  python3 fase1_telemetria/run_campaign.py --help
  python3 fase2_clasificador/run_training.py --help
  python3 fase3_daemon/run_daemon.py --help
  python3 fase4_evaluacion/run_evaluation.py --help
USAGE
}

cmd_check_readiness() {
    "$PYTHON" "$REPO_ROOT/common/readiness/check_node_readiness.py" "$@"
}

cmd_fase1() {
    (cd "$REPO_ROOT" && "$PYTHON" fase1_telemetria/run_campaign.py "$@")
}

cmd_fase2() {
    "$PYTHON" "$REPO_ROOT/fase2_clasificador/run_training.py" "$@"
}

cmd_fase3() {
    "$PYTHON" "$REPO_ROOT/fase3_daemon/run_daemon.py" "$@"
}

cmd_fase4() {
    "$PYTHON" "$REPO_ROOT/fase4_evaluacion/run_evaluation.py" "$@"
}

cmd_test() {
    log "pytest: common + las 4 fases"
    "$PYTHON" -m pytest \
        "$REPO_ROOT/common/tests" \
        "$REPO_ROOT/fase1_telemetria/tests" \
        "$REPO_ROOT/fase2_clasificador/tests" \
        "$REPO_ROOT/fase3_daemon/tests" \
        "$REPO_ROOT/fase4_evaluacion/tests" \
        -q

    log "cmake+ctest: common/telemetry (harness C++, requiere CUDA toolkit para WITH_GPU=ON)"
    cmake -S "$REPO_ROOT/common/telemetry" -B "$REPO_ROOT/common/telemetry/build" >/dev/null
    cmake --build "$REPO_ROOT/common/telemetry/build" -j >/dev/null
    ctest --test-dir "$REPO_ROOT/common/telemetry/build" --output-on-failure

    log "cmake+ctest: fase3_daemon/cpu_loop (máquina de decisión, sin dependencia de ONNX/CUDA)"
    cmake -S "$REPO_ROOT/fase3_daemon/cpu_loop" -B "$REPO_ROOT/fase3_daemon/cpu_loop/build" >/dev/null
    cmake --build "$REPO_ROOT/fase3_daemon/cpu_loop/build" -j >/dev/null
    ctest --test-dir "$REPO_ROOT/fase3_daemon/cpu_loop/build" --output-on-failure

    log "todo verde."
}

cmd_all() {
    local manifest="" node_id="" reference_kernel_ref="" campaign_dir="" campaign_id=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --manifest) manifest="$2"; shift 2 ;;
            --node-id) node_id="$2"; shift 2 ;;
            --reference-kernel-ref) reference_kernel_ref="$2"; shift 2 ;;
            --campaign-dir) campaign_dir="$2"; shift 2 ;;
            --campaign-id) campaign_id="$2"; shift 2 ;;
            *) fail "opción desconocida para 'all': $1" ;;
        esac
    done
    [[ -n "$manifest" && -n "$node_id" && -n "$reference_kernel_ref" && -n "$campaign_dir" && -n "$campaign_id" ]] \
        || fail "'all' requiere --manifest --node-id --reference-kernel-ref --campaign-dir --campaign-id"

    log "=== Fase 1: run-campaign ==="
    (cd "$REPO_ROOT" && "$PYTHON" fase1_telemetria/run_campaign.py run-campaign \
        --manifest "$manifest" --node-id "$node_id" --reference-kernel-ref "$reference_kernel_ref")

    log "=== Fase 2: entrenamiento + serialización ==="
    "$PYTHON" "$REPO_ROOT/fase2_clasificador/run_training.py" \
        --campaign-dir "$campaign_dir" --campaign-id "$campaign_id" \
        --output-dir "$REPO_ROOT/fase2_clasificador/models"

    log "=== Fase 3/4: no se lanzan automáticamente ==="
    log "Fase 3 requiere el loop de CPU real (no construido, ver fase3_daemon/README.md) y"
    log "la tabla de política derivada a mano (fase3_daemon/policy/derive_policy_table.py)."
    log "Fase 4 requiere windows.csv de los 3+1 escenarios, corridos por separado"
    log "(ver el procedimiento en fase4_evaluacion/README.md)."
    log "Fase 1 y Fase 2 completadas -- el resto es manual por ahora, no un fallo de este script."
}

main() {
    [[ $# -ge 1 ]] || { usage; exit 1; }
    local command="$1"; shift
    case "$command" in
        check-readiness) cmd_check_readiness "$@" ;;
        fase1) [[ "${1:-}" == "--" ]] && shift; cmd_fase1 "$@" ;;
        fase2) [[ "${1:-}" == "--" ]] && shift; cmd_fase2 "$@" ;;
        fase3) [[ "${1:-}" == "--" ]] && shift; cmd_fase3 "$@" ;;
        fase4) [[ "${1:-}" == "--" ]] && shift; cmd_fase4 "$@" ;;
        all) cmd_all "$@" ;;
        test) cmd_test ;;
        -h|--help|help) usage ;;
        *) usage; fail "comando desconocido: $command" ;;
    esac
}

main "$@"
