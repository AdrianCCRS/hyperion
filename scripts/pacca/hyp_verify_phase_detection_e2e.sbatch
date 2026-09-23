#!/bin/bash
# Bloque C3 (Plan_Fase3_Daemon.md SS0.1): verifica de punta a punta que
# activity_poller.poll_phase_events() detecta una fase de actividad de GPU
# REAL -- lanza gpu_dgemm_n4096 (checksum verificado, ~10s) como
# subproceso real mientras un hilo sondea NVML de verdad via nvidia-smi.
# NUNCA local (sin GPU en el laptop del usuario), ver memoria
# feedback-never-run-compute-locally.
#SBATCH --job-name=hyp_verify_phase_detection_e2e
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/verify_phase_detection_e2e_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/verify_phase_detection_e2e_%j.err

set -euo pipefail
ROOT="$HOME/hyperion"
KERNELS_ROOT="$HOME/hyperion-kernels"

# Mismo entorno real que las campanas (ver hyp_composite_known_smoke.sbatch
# para el porque -- jobs 7568-7570: sin esto, python3 y las bibliotecas
# reales de los kernels no se resuelven).
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"

hostname
python3 --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== tests unitarios (sin GPU/subproceso real) ==="
python3 -m pytest "$ROOT/fase3_daemon/gpu_loop/tests/test_verify_phase_detection_e2e.py" -q

echo "=== verificacion C3: gpu_dgemm_n4096 real + sondeo NVML real ==="
python3 "$ROOT/fase3_daemon/gpu_loop/verify_phase_detection_e2e.py" \
  --node-id pacca-a100 \
  --kernels-root "$KERNELS_ROOT" \
  --poll-interval-s 0.05 \
  --end-margin-s 2.0
