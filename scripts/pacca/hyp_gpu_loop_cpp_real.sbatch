#!/bin/bash
# Verificacion REAL del daemon de GPU en C++ (gpu_loop_main) en paccaA100. Escribe el reloj de la GPU de verdad.
#   1. Costo de CPU del sondeo con la GPU ociosa: daemon Python (run_daemon.py, sombra) vs C++ (sombra), 30 s cada uno.
#   2. Caos con el candado puesto: SIGTERM y SIGINT durante una fase memory; el reloj debe volver al nativo.
#   3. Compuesta REAL del catalogo (gpu_dgemm_n4096 + gpu_rajaperf_stream_triad) con el daemon en sombra y en activo:
#      clasificacion contra las fronteras reales y escritura/liberacion del reloj.
# Solo con el nodo libre. Sin --time. No intervenir a mano sobre el job en vuelo.
#SBATCH --job-name=hyp_gpu_loop_cpp_real
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_loop_cpp_real_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_loop_cpp_real_%j.err

set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
KERNEL_ROOT="$HOME/hyperion-kernels"
CPP="$ROOT/fase3_daemon/gpu_loop_cpp"
WORK="$CPP/real_work"
GPU=0
mkdir -p "$WORK"; rm -f "$WORK"/*.jsonl "$WORK"/*.out "$WORK"/*.txt 2>/dev/null
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate"
NVCC=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/bin/nvcc
BIN="$CPP/build/gpu_loop_main"
COMMON=(--model "$CPP/gpu_rf_sin_reloj.onnx" --features "$CPP/gpu_rf_sin_reloj.features.txt" --gpu-index $GPU)
DWELL_NS=3700000000; SETTLE=0.3
clk() { nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits; }
cleanup() { pkill -P $$ gpu_phase_target 2>/dev/null || true; pkill -P $$ -f "run_daemon.py|gpu_loop_main" 2>/dev/null || true; sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true; }
trap cleanup EXIT
hostname; git -C "$ROOT" log --oneline -1
[ -x "$BIN" ] || { echo "falta $BIN (correr hyp_gpu_loop_cpp_build_test.sbatch)" >&2; exit 1; }
sudo -n nvidia-smi -i $GPU -rgc || { echo "sudo -n nvidia-smi -rgc NO permitido" >&2; exit 1; }
"$NVCC" -O2 -arch=sm_80 -o "$WORK/gpu_phase_target" "$ROOT/fase3_daemon/gpu_loop/tools/gpu_phase_target.cu" || exit 1
T="$WORK/gpu_phase_target"
launch() {  # $1 = arm ; resto = flags extra. exec via el lanzador de politica (traduce policy_table.yaml)
  local arm=$1; shift
  python3 "$CPP/launch_gpu_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" --binary "$BIN" --arm "$arm" -- \
    "${COMMON[@]}" --min-dwell-ns $DWELL_NS --settle-s $SETTLE "$@"
}
ticks() { awk '{print $14+$15+$16+$17}' /proc/$1/stat; }   # utime+stime+cutime+cstime (ticks de 10 ms)

echo; echo "=== 1. costo de CPU del sondeo, GPU ociosa, 30 s, brazo sombra ==="
sleep 40 & SP=$!
python3 "$ROOT/fase3_daemon/run_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" --arm sombra --mode pid --pid $SP \
  --min-dwell-ns $DWELL_NS --gpu-index $GPU --delegated-cpus 0 > "$WORK/idle_python.out" 2>&1 & PP=$!
sleep 2; t0=$(ticks $PP); sleep 30; t1=$(ticks $PP); kill $PP 2>/dev/null; wait $PP 2>/dev/null; kill $SP 2>/dev/null
echo "Python (run_daemon.py + nvidia-smi por sondeo): $(( t1 - t0 )) ticks = $(awk -v a=$((t1-t0)) 'BEGIN{printf "%.2f", a/100}') s de CPU en 30 s = $(awk -v a=$((t1-t0)) 'BEGIN{printf "%.1f", a/30}')% de un nucleo"
sleep 40 & SP=$!
launch sombra --target-pid $SP > "$WORK/idle_cpp.out" 2>&1 & CP=$!
sleep 2; t0=$(ticks $CP); sleep 30; t1=$(ticks $CP); kill $CP 2>/dev/null; wait $CP 2>/dev/null; kill $SP 2>/dev/null
echo "C++ (gpu_loop_main, NVML directo):               $(( t1 - t0 )) ticks = $(awk -v a=$((t1-t0)) 'BEGIN{printf "%.2f", a/100}') s de CPU en 30 s = $(awk -v a=$((t1-t0)) 'BEGIN{printf "%.1f", a/30}')% de un nucleo"

echo; echo "=== 2. caos con el candado puesto (carga sintetica con huecos: fases de 20 s) ==="
chaos_round() {  # $1 = TERM|INT
  local sig=$1
  "$T" alternate 20 400 2 > "$WORK/phases_chaos_$sig.txt" 2>&1 & TGT=$!
  launch activo --target-pid $TGT --log-path "$WORK/chaos_$sig.jsonl" > "$WORK/chaos_$sig.out" 2>&1 & DP=$!
  local locked=""
  for i in $(seq 1 40); do
    sleep 1; c=$(clk)
    if [ "$c" -le 1275 ] && [ "$c" -ge 1245 ]; then locked=$c; echo "SIG$sig: candado observado a t=${i}s: clocks.sm=$c"; break; fi
    kill -0 $DP 2>/dev/null || { echo "el daemon murio antes de fijar el reloj" >&2; tail -20 "$WORK/chaos_$sig.out"; kill $TGT 2>/dev/null; return 1; }
  done
  [ -n "$locked" ] || { echo "SIG$sig: el daemon NO fijo el reloj en 40 s (puede haber clasificado la 1a fase como compute)" >&2; tail -8 "$WORK/chaos_$sig.out"; kill $DP $TGT 2>/dev/null; return 1; }
  kill -$sig $DP
  for i in $(seq 1 40); do kill -0 $DP 2>/dev/null || break; sleep 0.5; done
  if kill -0 $DP 2>/dev/null; then echo "!!! el daemon NO atendio SIG$sig en 20 s" >&2; kill -9 $DP; kill $TGT 2>/dev/null; return 1; fi
  wait $DP; local rc=$?
  sleep 3; local after; after=$(clk)
  grep -E "^(deteniendo|actuador GPU|resumen)" "$WORK/chaos_$sig.out"
  kill $TGT 2>/dev/null; wait $TGT 2>/dev/null
  if [ "$after" -gt 1300 ] && [ $rc -eq 0 ]; then echo "tras SIG$sig: clocks.sm=$after, codigo de salida $rc -> reloj RESTAURADO"; return 0; fi
  echo "!!! tras SIG$sig: clocks.sm=$after, codigo $rc: NO restaurado" >&2; return 1
}
chaos_round TERM || RC=1
sleep 3
chaos_round INT || RC=1

echo; echo "=== 3. compuesta real: gpu_dgemm_n4096 + gpu_rajaperf_stream_triad x3, sombra y activo ==="
run_arm() {
  local arm=$1
  ( cd "$KERNEL_ROOT" && python3 "$ROOT/fase3_daemon/composite_apps/composite_known.py" --node-id pacca-a100 \
      --kernels-root "$KERNEL_ROOT" --sequence gpu_dgemm_n4096 gpu_rajaperf_stream_triad --cycles 3 \
      --boundaries-out "$WORK/phases_$arm.jsonl" > "$WORK/composite_$arm.out" 2>&1 ) & local cp=$!
  launch $arm --target-pid $cp --log-path "$WORK/real_$arm.jsonl" > "$WORK/real_$arm.out" 2>&1 & local dp=$!
  wait $cp; sleep 2; kill -TERM $dp 2>/dev/null; wait $dp 2>/dev/null
  echo "--- brazo $arm ---"
  cut -c1-140 "$WORK/composite_$arm.out"
  python3 - "$WORK/phases_$arm.jsonl" "$WORK/real_$arm.jsonl" "$arm" <<'PY'
import json, sys
phases = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
try: dec = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
except FileNotFoundError: dec = []
ok = n = 0
for r in dec:
    ph = next((p for p in phases if p["begin_ns"] <= r["ts_ns"] <= p["end_ns"]), None)
    hint = ph["phase_label_hint"] if ph else None
    good = hint in ("compute_bound", "memory_bound") and hint == r["label"]
    if hint in ("compute_bound", "memory_bound"): n += 1; ok += good
    print(f"  {(ph['kernel_id'] if ph else '(entre fases)'):30s} verdad={hint} clase={r['label']} conf={r['confidence']:.2f} {'OK' if good else 'ERROR'} "
          f"objetivo_khz={r['target_freq_khz']} escrito={r['written']} actuacion_ms={(r.get('actuation_time_ns') or 0)/1e6:.0f} "
          f"inferencia_us={(r.get('inference_time_ns') or 0)/1e3:.0f} t_desde_inicio_fase_s={(r['ts_ns']-ph['begin_ns'])/1e9 if ph else float('nan'):.1f}")
print(f"brazo {sys.argv[3]}: clasificacion {ok}/{n} ({len(phases)} fases, {len(dec)} decisiones)")
PY
  sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1
  echo "reloj tras el brazo: $(clk) MHz"
}
run_arm sombra
run_arm activo
echo; [ "${RC:-0}" = 0 ] && echo "=== FIN: OK ===" || { echo "=== FIN: CON FALLOS ===" >&2; exit 1; }
