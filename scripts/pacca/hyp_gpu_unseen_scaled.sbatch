#!/bin/bash
# Candidatos a fase larga de UN solo kernel para la compuesta B de GPU: familias Rodinia inéditas con parámetros mayores
# (lavamd -boxes1d, myocyte pasos). Mide duración y si el daemon de GPU en C++ (sombra) decide, y con qué clase.
# Exploratorio: mismo binario del catálogo, argumentos escalados. Sin --time.
#SBATCH --job-name=hyp_gpu_unseen_scaled
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_unseen_scaled_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_unseen_scaled_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"; CPP="$ROOT/fase3_daemon/gpu_loop_cpp"
WORK="$CPP/real_work_scaled"; mkdir -p "$WORK"; rm -f "$WORK"/* 2>/dev/null
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
BIN="$CPP/build/gpu_loop_main"
hostname; git -C "$ROOT" log --oneline -1
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 || true
try() {  # $1 = binario, resto = argumentos
  local name="$1"; shift; local tag="${name}_$(echo "$*" | tr ' ' '_')"
  ( cd "$KERNEL_ROOT" && exec "bin/$name" "$@" ) > "$WORK/$tag.kout" 2>&1 & local kp=$!
  ( exec "$BIN" --arm sombra --model "$CPP/gpu_rf_sin_reloj.onnx" --features "$CPP/gpu_rf_sin_reloj.features.txt" --gpu-index 0 \
      --min-dwell-ns 3700000000 --settle-s 0.3 --poll-interval-ms 50 --compute-clock-mhz 0 --memory-clock-mhz 1260 \
      --target-pid $kp --log-path "$WORK/$tag.jsonl" ) > "$WORK/$tag.dout" 2>&1 & local dp=$!
  local t0=$(date +%s.%N); wait $kp; local rc=$?; local t1=$(date +%s.%N)
  sleep 1; kill -TERM $dp 2>/dev/null; wait $dp 2>/dev/null
  printf "%s %s: rc=%s %.1f s | " "$name" "$*" "$rc" "$(echo "$t1 - $t0" | bc)"
  python3 - "$WORK/$tag.jsonl" <<'PY'
import json, sys
try: d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError: d = []
print("decisiones:", [(r["label"], round(r["confidence"], 2), round(r["features"].get("phase_active_for_s", 0), 1)) for r in d] or "ninguna")
PY
}
try rodinia_lavamd -boxes1d 70
try rodinia_lavamd -boxes1d 100
try rodinia_lavamd -boxes1d 130
try rodinia_myocyte 100000 1 0
try rodinia_myocyte 400000 1 0
try rodinia_myocyte 1000000 1 0
