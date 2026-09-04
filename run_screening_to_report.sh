#!/usr/bin/env bash
# Fase 1: desde validación de paccaA100 hasta el informe de utilidad de los
# kernels tentativos. No genera ni ejecuta la campaña fina definitiva.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HYPERION_PYTHON:-python3}"
KERNEL_ROOT="${HOME}/hyperion-kernels"
RESULTS_ROOT="${HOME}/hyperion-results/screening"
TAG="screen_$(date +%Y%m%d)"
NODE_ID="pacca-a100"
REFERENCE_KERNEL="npb_mg"
STAGE="all"
NCU_COUNTS="5,20,50"
NCU_BINARY="ncu"
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${GPU_INDEX%%,*}"
TRANSITION_WORKLOAD=""
FORCE_NCU=0
NCU_KERNELS=()

usage() {
  cat <<'EOF'
Uso:
  bash run_screening_to_report.sh [opciones]

Opciones:
  --stage prepare|validate|screen-cpu|transition|ncu|screen-gpu|screen|warmup|report|all
  --tag ID                       Identificador reproducible de esta ejecución.
  --kernel-root DIR              Raíz externa con bin/ y datos de kernels.
  --results-root DIR             Raíz de resultados (se crea DIR/TAG).
  --node-id ID                   Clave de checksum del catálogo (pacca-a100).
  --reference-kernel-ref ID      Referencia de estabilidad de la campaña.
  --ncu-launch-counts LISTA      Límites crecientes, default 5,20,50.
  --ncu PATH                     Ejecutable ncu.
  --transition-workload CMD      Carga CUDA sostenida de >=12 s.
  --force-ncu                    Repite perfiles aunque ya exista el JSON.
  --ncu-kernel ID                Reperfilar solo este kernel (repetible).

El modo por etapas permite reanudar después de una reserva Slurm o ampliar
solo los puntos ncu de un kernel. `screen-cpu` es independiente de ncu;
`screen-gpu` nunca comienza sin un manifiesto filtrado por reportes ncu con
roofline_label_eligible=true. `screen` ejecuta ambos cribados.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --kernel-root) KERNEL_ROOT="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --node-id) NODE_ID="$2"; shift 2 ;;
    --reference-kernel-ref) REFERENCE_KERNEL="$2"; shift 2 ;;
    --ncu-launch-counts) NCU_COUNTS="$2"; shift 2 ;;
    --ncu) NCU_BINARY="$2"; shift 2 ;;
    --transition-workload) TRANSITION_WORKLOAD="$2"; shift 2 ;;
    --force-ncu) FORCE_NCU=1; shift ;;
    --ncu-kernel) NCU_KERNELS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Opción desconocida: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  prepare|validate|screen-cpu|transition|ncu|screen-gpu|screen|warmup|report|all) ;;
  *) printf 'Etapa inválida: %s\n' "$STAGE" >&2; exit 2 ;;
esac

RUN_ROOT="${RESULTS_ROOT%/}/${TAG}"
WORKFLOW="$RUN_ROOT/workflow.json"

log() { printf '\n[screening:%s] %s\n' "$STAGE" "$*" >&2; }
need_file() { [[ -f "$1" ]] || { printf 'Falta %s. Ejecute primero --stage %s.\n' "$1" "$2" >&2; exit 2; }; }
run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

workflow_value() {
  "$PYTHON" - "$WORKFLOW" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
}

manifest_kernels() {
  "$PYTHON" - "$1" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
for item in doc["kernels"]:
    print(item["kernel_ref"] if isinstance(item, dict) else item)
PY
}

validate_workflow_identity() {
  local saved_kernel saved_node current_kernel
  saved_kernel="$(workflow_value kernel_root)"
  saved_node="$(workflow_value node_id)"
  current_kernel="$(cd "$KERNEL_ROOT" 2>/dev/null && pwd)" || {
    printf 'No existe kernel-root: %s\n' "$KERNEL_ROOT" >&2; exit 2;
  }
  [[ "$current_kernel" == "$saved_kernel" ]] || {
    printf 'kernel-root cambió: workflow=%s, argumento=%s\n' "$saved_kernel" "$current_kernel" >&2; exit 2;
  }
  [[ "$NODE_ID" == "$saved_node" ]] || {
    printf 'node-id cambió: workflow=%s, argumento=%s\n' "$saved_node" "$NODE_ID" >&2; exit 2;
  }
}

stage_prepare() {
  log "Preparando catálogo de trabajo y manifiestos reproducibles"
  [[ -d "$KERNEL_ROOT" ]] || { printf 'No existe kernel-root: %s\n' "$KERNEL_ROOT" >&2; exit 2; }
  "$PYTHON" -m fase1_telemetria.screening_workflow prepare \
    --results-root "$RESULTS_ROOT" --tag "$TAG" --node-id "$NODE_ID" \
    --kernel-root "$KERNEL_ROOT" >/dev/null
  log "Workflow: $WORKFLOW"
}

stage_validate() {
  need_file "$WORKFLOW" prepare
  log "Readiness, build WITH_GPU=ON, tests y diagnóstico de manifiestos"
  "$PYTHON" common/readiness/check_node_readiness.py
  cmake -S common/telemetry -B common/telemetry/build -DWITH_GPU=ON
  cmake --build common/telemetry/build -j
  ctest --test-dir common/telemetry/build --output-on-failure
  local cpu_manifest gpu_manifest
  cpu_manifest="$(workflow_value cpu_manifest)"
  gpu_manifest="$(workflow_value gpu_candidates_manifest)"
  "$PYTHON" fase1_telemetria/run_campaign.py diagnose \
    --manifest "$cpu_manifest" --output-dir "$RUN_ROOT/diagnose/cpu" --use-allowed-cpus
  "$PYTHON" fase1_telemetria/run_campaign.py diagnose \
    --manifest "$gpu_manifest" --output-dir "$RUN_ROOT/diagnose/gpu" --use-allowed-cpus
  "$PYTHON" -m fase1_telemetria.screening_workflow check-binaries --workflow "$WORKFLOW"
}

stage_transition() {
  need_file "$WORKFLOW" prepare
  local diagnostic probe clocks mid min workload cadence_dir matrix_dir q
  diagnostic="$RUN_ROOT/diagnose/gpu/startup_diagnostic.json"
  need_file "$diagnostic" validate
  probe="$SCRIPT_DIR/common/telemetry/build/gpu_clock_transition_probe"
  need_file "$probe" validate
  clocks="$($PYTHON - "$diagnostic" <<'PY'
import json, sys
values = json.load(open(sys.argv[1]))["environment"].get("gpu_available_clocks_mhz") or []
print(",".join(str(x) for x in sorted(set(map(int, values)))))
PY
)"
  [[ -n "$clocks" ]] || { printf 'Diagnóstico sin gpu_available_clocks_mhz.\n' >&2; exit 2; }
  read -r mid min < <("$PYTHON" - "$clocks" <<'PY'
import sys
v=sorted(map(int, sys.argv[1].split(',')))
target=v[0]+0.5*(v[-1]-v[0])
print(min(v,key=lambda x:abs(x-target)), v[0])
PY
)
  workload="$TRANSITION_WORKLOAD"
  if [[ -z "$workload" ]]; then
    workload="$KERNEL_ROOT/bin/gpu_phasic --phase-seconds 1 --total-seconds 20 --size-mib 2048 --seed 20260806"
  fi
  cadence_dir="$(workflow_value transition_dir)/cadence"
  matrix_dir="$(workflow_value transition_dir)/matrix"
  mkdir -p "$cadence_dir" "$matrix_dir"
  log "Etapa A: cadencia NVML 5/10/50/100 ms; mid=${mid}MHz"
  for interval in 5000000 10000000 50000000 100000000; do
    "$probe" --workload-cmd "$workload" --gpu "$GPU_INDEX" \
      --from-clock REF --to-clock "$mid" --tolerance-mhz 15 \
      --probe-interval-ns "$interval" --dry-run-actuation \
      --warmup-ns 2000000000 --workload-min-active-ns 6000000000 \
      --max-wait-ns 1000000000 --out-dir "$cadence_dir/q_${interval}"
  done
  "$PYTHON" -m fase1_telemetria.gpu_transition.cadence_sweep "$cadence_dir" \
    --out "$cadence_dir/cadence_sweep.json"
  "$PYTHON" -m fase1_telemetria.screening_workflow apply-cadence \
    --workflow "$WORKFLOW" --cadence-report "$cadence_dir/cadence_sweep.json"
  q="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["q_produccion_ns"])' "$cadence_dir/cadence_sweep.json")"

  log "Etapa B: matriz de transición dirigida REF/F3/F6, tres réplicas"
  run_pair() {
    local from="$1" to="$2" label="$3" repetition
    for repetition in 1 2 3; do
      "$probe" --workload-cmd "$workload" --gpu "$GPU_INDEX" \
        --from-clock "$from" --to-clock "$to" --tolerance-mhz 15 \
        --stable-consecutive 3 --probe-interval-ns "$q" \
        --warmup-ns 2000000000 --workload-min-active-ns 6000000000 \
        --request-at-ns 5000000000 --max-wait-ns 3000000000 \
        --replicate-id "$repetition" --label "$label" \
        --out-dir "$matrix_dir/$label/r$repetition"
    done
  }
  run_pair REF "$mid" REF_to_F3
  run_pair REF "$min" REF_to_F6
  run_pair "$mid" "$min" F3_to_F6
  run_pair "$min" "$mid" F6_to_F3
  "$PYTHON" -m fase1_telemetria.gpu_transition.aggregate_transition_matrix \
    "$matrix_dir" --require-pair "REF->$mid" --require-pair "REF->$min" \
    --require-pair "$mid->$min" --require-pair "$min->$mid" \
    --output "$matrix_dir/transition_matrix_aggregate.json"
}

stage_ncu() {
  need_file "$WORKFLOW" prepare
  need_file "$(workflow_value transition_dir)/cadence/cadence_sweep.json" transition
  need_file "$(workflow_value transition_dir)/matrix/transition_matrix_aggregate.json" transition
  log "F1-GPU-004: ncu ANTES del cribado GPU"
  local args=(--workflow "$WORKFLOW" --launch-counts "$NCU_COUNTS" --ncu "$NCU_BINARY")
  [[ "$FORCE_NCU" == 1 ]] && args+=(--force)
  local kernel
  for kernel in "${NCU_KERNELS[@]}"; do args+=(--kernel "$kernel"); done
  "$PYTHON" -m fase1_telemetria.screening_workflow ncu "${args[@]}"
}

stage_screen_cpu() {
  need_file "$WORKFLOW" prepare
  local cpu_manifest
  cpu_manifest="$(workflow_value cpu_manifest)"
  log "Cribado CPU (OI viva mediante uncore_imc)"
  (cd "$KERNEL_ROOT" && "$PYTHON" "$SCRIPT_DIR/fase1_telemetria/run_campaign.py" run-campaign \
    --manifest "$cpu_manifest" --node-id "$NODE_ID" --reference-kernel-ref "$REFERENCE_KERNEL")
}

stage_screen_gpu() {
  need_file "$WORKFLOW" prepare
  local gpu_manifest
  gpu_manifest="$(workflow_value gpu_eligible_manifest)"
  need_file "$gpu_manifest" ncu
  log "Cribado GPU exclusivamente con candidatos habilitados por ncu"
  (cd "$KERNEL_ROOT" && "$PYTHON" "$SCRIPT_DIR/fase1_telemetria/run_campaign.py" run-campaign \
    --manifest "$gpu_manifest" --node-id "$NODE_ID" --reference-kernel-ref "$REFERENCE_KERNEL")
}

calibrate_and_reprocess() {
  local device="$1" manifest="$2" campaign_dir="$3" warmup_dir="$4"
  local kernel_args=() kernel
  while IFS= read -r kernel; do kernel_args+=(--kernel "$kernel"); done < <(manifest_kernels "$manifest")
  "$PYTHON" -m fase1_telemetria.warmup_calibration \
    --campaign-dir "$campaign_dir" "${kernel_args[@]}" --device "$device" \
    --out-dir "$warmup_dir" --catalog "$(workflow_value catalog)" --apply
  "$PYTHON" -m fase1_telemetria.repostprocess_campaign \
    --manifest "$manifest" --node-id "$NODE_ID" --catalog-path "$(workflow_value catalog)"
}

stage_warmup() {
  need_file "$WORKFLOW" prepare
  local cpu_manifest gpu_manifest
  cpu_manifest="$(workflow_value cpu_manifest)"
  gpu_manifest="$(workflow_value gpu_eligible_manifest)"
  log "Calibrando warmup y reprocesando las mismas corridas, sin relanzar kernels"
  calibrate_and_reprocess cpu "$cpu_manifest" "$(workflow_value cpu_campaign_dir)" "$(workflow_value warmup_cpu_dir)"
  calibrate_and_reprocess gpu "$gpu_manifest" "$(workflow_value gpu_campaign_dir)" "$(workflow_value warmup_gpu_dir)"
}

stage_report() {
  need_file "$WORKFLOW" prepare
  need_file "$(workflow_value warmup_cpu_dir)/warmup_calibration.json" warmup
  need_file "$(workflow_value warmup_gpu_dir)/warmup_calibration.json" warmup
  local cpu_campaign gpu_campaign
  cpu_campaign="$(workflow_value cpu_campaign_dir)"
  gpu_campaign="$(workflow_value gpu_campaign_dir)"
  log "Cobertura por dispositivo e informe de utilidad de candidatos"
  "$PYTHON" fase2_clasificador/run_phase_coverage.py \
    --campaign-dir "$cpu_campaign" --campaign-id "$(workflow_value cpu_campaign_id)" \
    --device cpu --output-dir "$(workflow_value coverage_cpu_dir)"
  "$PYTHON" fase2_clasificador/run_phase_coverage.py \
    --campaign-dir "$gpu_campaign" --campaign-id "$(workflow_value gpu_campaign_id)" \
    --device gpu --output-dir "$(workflow_value coverage_gpu_dir)"
  "$PYTHON" -m fase1_telemetria.tentative_kernel_report --workflow "$WORKFLOW"
}

cd "$SCRIPT_DIR"
if run_stage prepare || [[ "$STAGE" == all ]]; then stage_prepare; fi
if [[ "$STAGE" != prepare ]]; then need_file "$WORKFLOW" prepare; validate_workflow_identity; fi
if run_stage validate; then stage_validate; fi
# CPU no depende de ncu. En `all` se ejecuta antes de ocupar la GPU; para
# reservas separadas puede lanzarse explícitamente con `--stage screen-cpu`.
if [[ "$STAGE" == all || "$STAGE" == screen || "$STAGE" == screen-cpu ]]; then stage_screen_cpu; fi
if run_stage transition; then stage_transition; fi
if run_stage ncu; then stage_ncu; fi
if [[ "$STAGE" == all || "$STAGE" == screen || "$STAGE" == screen-gpu ]]; then stage_screen_gpu; fi
if run_stage warmup; then stage_warmup; fi
if run_stage report; then stage_report; fi

log "Etapa solicitada completada. La campaña fina NO se generó ni ejecutó."
