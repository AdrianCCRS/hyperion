#!/bin/bash
# Duracion de candidatos a aplicacion compuesta B de GPU (sin phase_label_hint): ERT probes, BabelStream, cuBLAS dgemm de
# calibracion, gemm nativo. No toca frecuencias. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_gpu_survey_b
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_survey_b_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_survey_b_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
python3 "$ROOT/fase3_daemon/composite_apps/survey_kernels.py" --node-id pacca-a100 --kernels-root "$KERNEL_ROOT" \
  gpu_ert_probe_fp32 gpu_ert_probe_fp64 gpu_stream_bw gpu_dgemm_calibration gpu_gemm_native_n4096
