#!/bin/bash
# CloverLeaf: cinco bloques REF/sombra/activo, RF congelado, umbral 0.90.
#SBATCH --job-name=hyp_cloverleaf_confirm
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err
set -euo pipefail

ROOT=${HYPERION_ROOT:-$HOME/hyperion_c8}
OUT=${OUT:-$HOME/hyperion-results/final/cloverleaf_confirm_${SLURM_JOB_ID}}
SRC=$HOME/hyperion-kernels/external_src/CloverLeaf_CUDAFortran
INPUT=$ROOT/scripts/pacca/cloverleaf_screen_6000.in
BIN=$HOME/hyperion-results/final/cloverleaf_shadow_7689/build/gpu_loop_main
CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
MODEL=$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.onnx
FEATURES=$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.features.txt
POLICY=$ROOT/fase3_daemon/policy_table.yaml
R0=/sys/class/powercap/intel-rapl:0
R1=/sys/class/powercap/intel-rapl:1
RMAX=$(cat "$R0/max_energy_range_uj")
mkdir -p "$OUT"
source "$HOME/hyperion-venv/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

test -x "$SRC/clover_leaf" && test -x "$BIN"
test "$(git -C "$SRC" rev-parse HEAD)" = 6558a980d6094a70e815d445d0f561c11c0c8ef6
test "$(sha256sum "$INPUT" | awk '{print $1}')" = 77991ce329268acf6195219a37906b1d3fdc719d588452498f5e0ee2db6576eb
sha256sum "$INPUT" > "$OUT/input.sha256"
git -C "$SRC" diff > "$OUT/source_patch.diff"
sha256sum "$SRC/clover_leaf" "$BIN" "$MODEL" "$FEATURES" "$POLICY" > "$OUT/binaries_and_policy.sha256"
g++ -O2 -o "$OUT/gpu_energy_read" "$ROOT/fase3_daemon/composite_apps/gpu_energy_read.cpp" \
  -I"$CUDA/include" "$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"
GER=$OUT/gpu_energy_read

python3 - "$OUT/plan.txt" <<'PY'
import random, sys
rng = random.Random(20260924)
with open(sys.argv[1], 'w') as out:
    for block in range(1, 6):
        arms = ['ref', 'sombra', 'activo']
        rng.shuffle(arms)
        for arm in arms:
            out.write(f'{block} {arm}\n')
PY
echo 'block,arm,wall_s,e_cpu_j,e_gpu_j,kinetic,app_rc,daemon_rc,state_ok' > "$OUT/results.csv"

app_pid=''; daemon_pid=''; sampler_pid=''
cleanup() {
  [ -z "$sampler_pid" ] || { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
  [ -z "$daemon_pid" ] || { kill -TERM "$daemon_pid" 2>/dev/null || true; wait "$daemon_pid" 2>/dev/null || true; }
  [ -z "$app_pid" ] || { kill -TERM "$app_pid" 2>/dev/null || true; wait "$app_pid" 2>/dev/null || true; }
  sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT
clock_state() {
  nvidia-smi -i 0 --query-gpu=clocks.applications.graphics,clocks.applications.memory \
    --format=csv,noheader,nounits | tr -d ' '
}

while read -r block arm; do
  d=$OUT/b${block}_$arm
  mkdir -p "$d"
  cp "$INPUT" "$d/clover.in"
  sudo -n nvidia-smi -i 0 -rgc >/dev/null
  before=$(clock_state)
  printf '%s\n' "$before" > "$d/clock_before.txt"
  (cd "$d" && exec "$SRC/clover_leaf") > "$d/app.out" 2>&1 &
  app_pid=$!
  kill -STOP "$app_pid"
  daemon_pid=''; daemon_rc=NA
  if [ "$arm" != ref ]; then
    (exec python3 "$ROOT/fase3_daemon/gpu_loop_cpp/launch_gpu_daemon.py" \
      --policy-table "$POLICY" --binary "$BIN" --arm "$arm" -- \
      --model "$MODEL" --features "$FEATURES" --gpu-index 0 \
      --min-dwell-ns 3700000000 --settle-s 0.3 --target-pid "$app_pid" \
      --threshold 0.90 --log-path "$d/gpu_decisions.jsonl") > "$d/daemon.out" 2>&1 &
    daemon_pid=$!
  fi
  sleep 3
  if [ -n "$daemon_pid" ] && ! kill -0 "$daemon_pid" 2>/dev/null; then
    echo "daemon GPU salió antes de iniciar $block/$arm" >&2
    exit 1
  fi
  e0=$(cat "$R0/energy_uj"); f0=$(cat "$R1/energy_uj")
  g0=$("$GER" 0); t0=$(date +%s.%N)
  nvidia-smi -i 0 --query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw \
    --format=csv,noheader,nounits -lms 250 > "$d/nvml.csv" 2>&1 &
  sampler_pid=$!
  kill -CONT "$app_pid"
  app_rc=0; wait "$app_pid" || app_rc=$?
  app_pid=''
  t1=$(date +%s.%N); e1=$(cat "$R0/energy_uj"); f1=$(cat "$R1/energy_uj"); g1=$("$GER" 0)
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  sampler_pid=''
  if [ -n "$daemon_pid" ]; then
    daemon_rc=0; wait "$daemon_pid" || daemon_rc=$?
    daemon_pid=''
  fi
  reset_rc=0; sudo -n nvidia-smi -i 0 -rgc >/dev/null || reset_rc=$?
  after=$(clock_state)
  printf '%s\n' "$after" > "$d/clock_after.txt"
  state_ok=0
  [ "$reset_rc" = 0 ] && [ "$before" = "$after" ] && state_ok=1
  read -r wall cpu gpu < <(python3 - "$t0" "$t1" "$e0" "$e1" "$f0" "$f1" "$g0" "$g1" "$RMAX" <<'PY'
import math, sys
t0,t1,e0,e1,f0,f1,g0,g1,m = map(float,sys.argv[1:])
wall=t1-t0
cpu=((e1-e0)%m+(f1-f0)%m)/1e6
gpu=(g1-g0)/1e3
assert all(math.isfinite(x) and x>0 for x in (wall,cpu,gpu))
print(f'{wall:.6f} {cpu:.6f} {gpu:.6f}')
PY
  )
  kinetic=$(awk '/^ step:/{v=$8} END{print v}' "$d/clover.out")
  test -n "$kinetic"
  echo "$block,$arm,$wall,$cpu,$gpu,$kinetic,$app_rc,$daemon_rc,$state_ok" | tee -a "$OUT/results.csv"
  [ "$app_rc" = 0 ] && { [ "$daemon_rc" = 0 ] || [ "$daemon_rc" = NA ]; } && [ "$state_ok" = 1 ] || exit 1
done < "$OUT/plan.txt"

python3 - "$OUT/results.csv" "$OUT/numeric_check.json" <<'PY'
import csv, json, math, sys
rows=list(csv.DictReader(open(sys.argv[1])))
assert len(rows)==15
ref=[float(r['kinetic']) for r in rows if r['arm']=='ref']
assert len(ref)==5 and all(math.isfinite(x) for x in ref)
base=sum(ref)/len(ref)
checks=[{'block':r['block'],'arm':r['arm'],'relative_error':abs(float(r['kinetic'])-base)/abs(base)} for r in rows]
json.dump(checks,open(sys.argv[2],'w'),indent=2)
assert all(x['relative_error']<=1e-8 for x in checks)
PY
