#!/bin/bash
# Sondeo del escenario D de la Fase 4 (aplicacion real de terceros: LAMMPS 2023 con paquete GPU del modulo del cluster).
# Responde dos preguntas sin medir energia: (1) por que el agente de GPU no abrio fase con myocyte (utilizacion de GPU cada 0.25 s
# durante la secuencia de la aplicacion E-B) y (2) como se ve cada benchmark de LAMMPS a escala de segundos (utilizacion, reloj,
# potencia) y que clase decide el agente en sombra. No toca frecuencias (el agente va en brazo sombra). NUNCA local. Sin --time.
#SBATCH --job-name=hyp_fase4_D_probe
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"; OUT="${OUT:-$HOME/hyperion-results/final/fase4_D_probe}"; mkdir -p "$OUT"
CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0; BENCH=/usr/local/src/lammps/lammps-23Jun2022/bench
STEPS="${STEPS:-20000}"; MAXS="${MAXS:-75}"
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
GPU_BIN="$HOME/hyperion-results/final/fase4_E/build/gpu_build/gpu_loop_main"; [ -x "$GPU_BIN" ] || echo "!!! sin binario del agente de GPU: $GPU_BIN" >&2
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
sampler() { nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,power.draw --format=csv,noheader -lms 250 > "$1" & echo $!; }

# (1) secuencia de la aplicacion E-B (BabelStream, LavaMD, myocyte) con utilizacion de GPU y sin agente
S=$(sampler "$OUT/util_eb.csv")
( cd "$KERNEL_ROOT" && python3 "$ROOT/fase3_daemon/composite_apps/composite_gpu_memdom.py" --set unseen --node-id pacca-a100 \
    --kernels-root "$KERNEL_ROOT" --cycles 1 --gap-s 1 --boundaries-out "$OUT/phases_eb.jsonl" ) > "$OUT/eb.out" 2>&1
kill $S 2>/dev/null; sleep 3

# (2) LAMMPS: cada benchmark con el paquete GPU, entradas escaladas y acotadas en tiempo, con el agente de GPU en sombra
module load lammps/2023_GPU 2>&1 || true
which lmp; lmp -h 2>&1 | grep -i -m3 "gpu\|MPI" || true
for b in lj eam chain rhodo; do
  W="$OUT/lammps_$b"; rm -rf "$W"; mkdir -p "$W"; cp -r "$BENCH"/* "$W"/ 2>/dev/null; cd "$W"
  sed -i -E "s/^run[[:space:]]+100$/run\t\t$STEPS/" "in.$b"
  VARS=""; [ "$b" = lj ] || [ "$b" = eam ] && VARS="-var x 4 -var y 4 -var z 4"
  S=$(sampler "$OUT/util_$b.csv")
  T0=$(date +%s.%N)
  timeout $MAXS lmp -sf gpu -pk gpu 1 -in "in.$b" $VARS -log "$W/log.lammps" > "$W/stdout.txt" 2> "$W/stderr.txt" &
  L=$!; sleep 2
  if [ -x "$GPU_BIN" ]; then
    ( exec python3 "$ROOT/fase3_daemon/gpu_loop_cpp/launch_gpu_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" --binary "$GPU_BIN" --arm sombra -- \
        --model "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.onnx" --features "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.features.txt" \
        --gpu-index 0 --min-dwell-ns 3700000000 --settle-s 0.3 --target-pid $L --log-path "$W/gpu_decisions.jsonl" ) > "$W/gpu_daemon.out" 2>&1 & D=$!
  fi
  wait $L; RC=$?; T1=$(date +%s.%N)
  [ -n "${D:-}" ] && { kill -TERM $D 2>/dev/null; wait $D 2>/dev/null; }
  kill $S 2>/dev/null; sleep 3
  echo "lammps $b rc=$RC duracion=$(python3 -c "print(round($T1-$T0,1))")s"; tail -3 "$W/stdout.txt"
done
cd "$OUT"; echo "=== FIN sondeo D ==="
