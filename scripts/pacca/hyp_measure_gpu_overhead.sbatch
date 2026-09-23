#!/bin/bash
# Bloque C6 (Plan_Fase3_Daemon.md SS0.1): mide la sobrecarga real del loop
# de GPU -- clasificador real, tabla de politica real, kernel de terceros
# real (gpu_dgemm_n4096, checksum verificado), 5 ciclos. Complementa la
# sobrecarga ya medida del lado CPU (Bloque B, cpu_loop_latency_bench).
# NUNCA local, ver memoria feedback-never-run-compute-locally.
#SBATCH --job-name=hyp_measure_gpu_overhead
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/measure_gpu_overhead_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/measure_gpu_overhead_%j.err

set -euo pipefail
ROOT="$HOME/hyperion"
KERNELS_ROOT="$HOME/hyperion-kernels"

# Mismo entorno real que las campanas (ver hyp_composite_known_smoke.sbatch).
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"

hostname
python3 --version

echo "=== tests unitarios (sin GPU/clasificador real) ==="
python3 -m pytest "$ROOT/fase3_daemon/gpu_loop/tests/test_measure_overhead.py" -q

echo "=== medicion real: clasificador + politica + gpu_dgemm_n4096, 5 ciclos ==="
OUT="$ROOT/fase3_daemon/gpu_loop/gpu_overhead_report_${SLURM_JOB_ID}.json"
python3 "$ROOT/fase3_daemon/gpu_loop/measure_overhead.py" \
  --node-id pacca-a100 \
  --kernels-root "$KERNELS_ROOT" \
  --cycles 5 \
  --out "$OUT"

echo "=== reporte registrado ==="
cat "$OUT"
