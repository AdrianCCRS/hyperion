#!/bin/bash
# FASE 4: matriz de experimentos daemon vs REF. Ejes: alcance {cpu, gpu, joint} x kernels {known, unseen} x brazo.
#   cpu   : base, sombra, activo, activo_f0                   (solo daemon de CPU)         3 repeticiones
#   gpu   : base, sombra, activo                              (solo daemon de GPU)         3 repeticiones
#   (diagnostico, solo si se piden en ONLY_ARMS: sombra_pin, sombra_ort1, sombra_pin_ort1 = sombra con el consumidor fijado a --consumer-cpu y/o ORT con 1 hilo sin espera activa)
#   joint : base, sombra, activo (con piso), activo_nofloor   (ambos daemons, app que alterna CPU y GPU)  5 repeticiones
# base = REF (gobernador nativo con turbo, sin daemon). Por celda se registra duracion, energia de CPU (RAPL, ambos paquetes)
# y de GPU (NVML) POR SEPARADO (el EDP del nodo se calcula luego), mas las fronteras de fase y las decisiones de cada daemon
# para puntuar clasificacion. Orden aleatorizado, reanudable (salta celdas ya hechas). Toca turbo, frecuencias y reloj de GPU:
# nodo libre, restaura desde un snapshot. NUNCA local. Sin --time. No intervenir a mano.
# Variables: INTERVAL_NS (1000000, muestreo del daemon de CPU) CYCLES (2) GAP_S (1) REPS_GENERAL (3) REPS_JOINT (5) ONLY_SCOPES ONLY_SETS ONLY_ARMS (listas separadas por espacio)
#            OUT (carpeta de resultados; para reanudar, misma carpeta)
#SBATCH --job-name=hyp_fase4_matrix
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/fase4_matrix_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/fase4_matrix_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"
INTERVAL_NS="${INTERVAL_NS:-1000000}"; CYCLES="${CYCLES:-2}"; GAP_S="${GAP_S:-1}"; REPS_GENERAL="${REPS_GENERAL:-3}"; REPS_JOINT="${REPS_JOINT:-5}"
ONLY_SCOPES="${ONLY_SCOPES:-cpu gpu joint}"; ONLY_SETS="${ONLY_SETS:-known unseen}"; ONLY_ARMS="${ONLY_ARMS:-base sombra activo activo_f0 activo_nofloor}"
OUT="${OUT:-$HOME/hyperion-results/final/fase4_matrix_${SLURM_JOB_ID:-manual}}"; mkdir -p "$OUT/cells"
ORT="$HOME/.conda/envs/hyperion-cpu-onnx"; CMAKE="$ORT/bin/cmake"; CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
C=/sys/devices/system/cpu; CPUS="0 1 2 3 4 5 16 17 18 19 20 21"; GPU=0; FLOOR_KHZ=3200000
R0=/sys/class/powercap/intel-rapl:0; R1=/sys/class/powercap/intel-rapl:1; RMAX=$(cat $R0/max_energy_range_uj)
hostname; git -C "$ROOT" log --oneline -1; echo "salida: $OUT  ciclos=$CYCLES hueco=${GAP_S}s reps=$REPS_GENERAL/$REPS_JOINT"

BUILD="$OUT/build"; mkdir -p "$BUILD"
if [ ! -x "$BUILD/cpu_build/cpu_loop_main" ] || [ ! -x "$BUILD/gpu_build/gpu_loop_main" ]; then
  "$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD/cpu_build" -DONNXRUNTIME_ROOT="$ORT" >/dev/null && "$CMAKE" --build "$BUILD/cpu_build" -j 8 --target cpu_loop_main 2>&1 | grep -E "error|Built target"
  "$CMAKE" -S "$ROOT/fase3_daemon/gpu_loop_cpp" -B "$BUILD/gpu_build" -DONNXRUNTIME_ROOT="$ORT" -DNVML_INCLUDE_DIR="$CUDA/include" \
    -DNVML_LIB="$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so" >/dev/null && "$CMAKE" --build "$BUILD/gpu_build" -j 8 --target gpu_loop_main 2>&1 | grep -E "error|Built target"
fi
CPU_BIN="$BUILD/cpu_build/cpu_loop_main"; GPU_BIN="$BUILD/gpu_build/gpu_loop_main"
[ -x "$CPU_BIN" ] && [ -x "$GPU_BIN" ] || { echo "no se construyeron los binarios" >&2; exit 1; }
g++ -O2 -o "$BUILD/gpu_energy_read" "$ROOT/fase3_daemon/composite_apps/gpu_energy_read.cpp" -I"$CUDA/include" "$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so" || exit 1
GER="$BUILD/gpu_energy_read"; "$GER" $GPU >/dev/null || { echo "sin lectura de energia de GPU" >&2; exit 1; }

# politicas: GPU tal como esta versionada; CPU experimental F0 base / F1 en memory (activo) o F0 en ambas (activo_f0)
python3 - "$ROOT/fase3_daemon/policy_table.yaml" "$OUT" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
def cpu(c, m):
    e = dict(d); e["policy"] = dict(d["policy"])
    e["policy"]["cpu-compute_bound"] = {"action": "actuar_experimental", "chosen_level": "F0", "resolved_freq_khz": 3200000}
    e["policy"]["cpu-memory_bound"] = {"action": "actuar_experimental", "chosen_level": m[0], "resolved_freq_khz": m[1]}
    yaml.safe_dump(e, open(f"{sys.argv[2]}/policy_{c}.yaml", "w"))
cpu("activo", ("F1", 2900000)); cpu("activo_f0", ("F0", 3200000))
PY

state() { echo "no_turbo=$(cat $C/intel_pstate/no_turbo)"; for c in $CPUS; do d=$C/cpu$c/cpufreq; echo "cpu$c min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq)"; done; }
state > "$OUT/state_inicial.txt"; ORIG_NO_TURBO=$(head -1 "$OUT/state_inicial.txt" | cut -d= -f2)
restore_cpu() {
  while read -r line; do case "$line" in cpu*) c=${line%% *}; c=${c#cpu}; mn=$(echo "$line" | sed -n 's/.*min=\([0-9]*\).*/\1/p'); mx=$(echo "$line" | sed -n 's/.*max=\([0-9]*\).*/\1/p'); d=$C/cpu$c/cpufreq; echo "$mx" > $d/scaling_max_freq; echo "$mn" > $d/scaling_min_freq;; esac; done < "$OUT/state_inicial.txt"
  sudo -n /usr/local/bin/set_turbo_state "$ORIG_NO_TURBO" || true
}
cleanup() { pkill -P $$ -f "cpu_loop_main|gpu_loop_main|composite_" 2>/dev/null || true; state > "$OUT/state_ahora.txt"
  diff -q "$OUT/state_inicial.txt" "$OUT/state_ahora.txt" >/dev/null 2>&1 || restore_cpu; sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true; }
trap cleanup EXIT
sudo -n nvidia-smi -i $GPU -rgc || { echo "sudo -n nvidia-smi -rgc NO permitido" >&2; exit 1; }

RES="$OUT/results.csv"
[ -f "$RES" ] || echo "cell,scope,set,arm,rep,wall_s,e_cpu_j,e_gpu_j,app_rc,rc_cpu_daemon,rc_gpu_daemon,state_ok" > "$RES"
python3 - "$ONLY_SCOPES" "$ONLY_SETS" "$ONLY_ARMS" "$REPS_GENERAL" "$REPS_JOINT" > "$OUT/cells.txt" <<'PY'
import random, sys
scopes, sets, arms = (set(a.split()) for a in sys.argv[1:4]); rg, rj = int(sys.argv[4]), int(sys.argv[5])
plan = {"cpu": (["base", "sombra", "activo", "activo_f0", "sombra_pin", "sombra_ort1", "sombra_pin_ort1"], rg), "gpu": (["base", "sombra", "activo"], rg),
        "joint": (["base", "sombra", "activo", "activo_nofloor"], rj)}
cells = [(sc, st, arm, r) for sc, (al, n) in plan.items() if sc in scopes for st in ("known", "unseen") if st in sets
         for arm in al if arm in arms for r in range(1, n + 1)]
random.Random(20260924).shuffle(cells)
for c in cells: print(*c)
PY
echo "celdas planificadas: $(wc -l < "$OUT/cells.txt")"

app_cmd() {  # $1 scope $2 set -> imprime el comando de la aplicacion
  local A="$ROOT/fase3_daemon/composite_apps"
  case "$1:$2" in
    cpu:known)   echo "$A/composite_known.py" ;;
    cpu:unseen)  echo "$A/composite_unseen.py" ;;
    gpu:known)   echo "$A/composite_gpu_known.py" ;;
    gpu:unseen)  echo "$A/composite_gpu_unseen.py" ;;
    joint:*)     echo "$A/composite_joint.py --set $2" ;;
  esac
}
run_cell() {
  local scope=$1 set=$2 arm=$3 rep=$4 id="$1_$2_$3_$4" D="$OUT/cells/$1_$2_$3_$4"
  mkdir -p "$D"; local SIG="$D/sig" pol dm_arm=sombra floor=$FLOOR_KHZ
  case "$arm" in activo|activo_f0|activo_nofloor) dm_arm=activo;; esac
  local xflags=()  # variantes de diagnostico del sobrecosto del daemon de CPU (solo brazo sombra)
  case "$arm" in sombra_pin) xflags=(--pin-consumer);; sombra_ort1) xflags=(--ort-threads 1);; sombra_pin_ort1) xflags=(--pin-consumer --ort-threads 1);; esac
  [ "$arm" = activo_nofloor ] && floor=0
  pol="$OUT/policy_activo.yaml"; [ "$arm" = activo_f0 ] && pol="$OUT/policy_activo_f0.yaml"; [ "$arm" = sombra ] && pol="$OUT/policy_activo.yaml"
  [ "$arm" = activo_nofloor ] && pol="$OUT/policy_activo.yaml"
  sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1; sleep 5
  local cpu_d="" gpu_d="" rc_c=NA rc_g=NA
  local sigflag=(); [ "$scope" = joint ] && sigflag=(--gpu-active-signal-path "$SIG")
  local cmd; cmd=$(app_cmd "$scope" "$set")
  ( cd "$KERNEL_ROOT" && exec taskset -c 0-3 python3 $cmd --node-id pacca-a100 --kernels-root "$KERNEL_ROOT" --cycles "$CYCLES" --gap-s "$GAP_S" \
      --boundaries-out "$D/phases.jsonl" > "$D/app.out" 2>&1 ) & local COMP=$!
  # el brazo base no arranca daemons; los demas, segun el alcance
  kill -STOP $COMP   # la app espera a que los daemons esten listos
  if [ "$arm" != base ] && { [ "$scope" = gpu ] || [ "$scope" = joint ]; }; then
    ( exec python3 "$ROOT/fase3_daemon/gpu_loop_cpp/launch_gpu_daemon.py" --policy-table "$pol" --binary "$GPU_BIN" --arm $dm_arm -- \
        --model "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.onnx" --features "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.features.txt" \
        --gpu-index $GPU --min-dwell-ns 3700000000 --settle-s 0.3 --target-pid $COMP "${sigflag[@]}" --log-path "$D/gpu_decisions.jsonl" ) > "$D/gpu_daemon.out" 2>&1 & gpu_d=$!
  fi
  if [ "$arm" != base ] && { [ "$scope" = cpu ] || [ "$scope" = joint ]; }; then
    ( exec python3 "$ROOT/fase3_daemon/cpu_loop/launch_cpu_daemon.py" --policy-table "$pol" --binary "$CPU_BIN" --arm $dm_arm -- \
        --model "$ROOT/fase3_daemon/cpu_loop/xgboost_cpu.onnx" --perf-cpus 0,1,2,3 --collector-cpu 6 --consumer-cpu 7 --target-pid $COMP --interval-ns $INTERVAL_NS "${xflags[@]}" \
        "${sigflag[@]}" --gpu-active-floor-khz $floor --log-path "$D/cpu_decisions.jsonl" ) > "$D/cpu_daemon.out" 2>&1 & cpu_d=$!
  fi
  sleep 3
  [ -n "$gpu_d" ] && ! kill -0 $gpu_d 2>/dev/null && echo "  !!! daemon GPU murio al arrancar ($id)" >&2
  [ -n "$cpu_d" ] && ! kill -0 $cpu_d 2>/dev/null && echo "  !!! daemon CPU murio al arrancar ($id)" >&2
  local e0 f0 g0 t0; e0=$(cat $R0/energy_uj); f0=$(cat $R1/energy_uj); g0=$("$GER" $GPU); t0=$(date +%s.%N)
  kill -CONT $COMP; wait $COMP; local app_rc=$?
  local t1 e1 f1 g1; t1=$(date +%s.%N); e1=$(cat $R0/energy_uj); f1=$(cat $R1/energy_uj); g1=$("$GER" $GPU)
  sleep 1
  [ -n "$cpu_d" ] && kill -TERM $cpu_d 2>/dev/null; [ -n "$gpu_d" ] && kill -TERM $gpu_d 2>/dev/null
  [ -n "$cpu_d" ] && { wait $cpu_d; rc_c=$?; }; [ -n "$gpu_d" ] && { wait $gpu_d; rc_g=$?; }
  state > "$D/state_final.txt"; local ok=1; diff -q "$OUT/state_inicial.txt" "$D/state_final.txt" >/dev/null 2>&1 || { ok=0; restore_cpu; }
  local wall ecpu egpu
  wall=$(python3 -c "print(round($t1-$t0,3))")
  ecpu=$(python3 -c "m=$RMAX; d=lambda a,b:(b-a)%m; print(round((d($e0,$e1)+d($f0,$f1))/1e6,2))")
  egpu=$(python3 -c "print(round(($g1-$g0)/1e3,2))")
  echo "$id,$scope,$set,$arm,$rep,$wall,$ecpu,$egpu,$app_rc,$rc_c,$rc_g,$ok" | tee -a "$RES"
}
while read -r scope set arm rep; do
  grep -q "^${scope}_${set}_${arm}_${rep}," "$RES" && continue
  run_cell "$scope" "$set" "$arm" "$rep"
done < "$OUT/cells.txt"
echo "=== FIN: $(($(wc -l < "$RES")-1)) celdas en $RES ==="
