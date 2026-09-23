#!/bin/bash
# Sondeo de duracion de kernels de GPU candidatos para las aplicaciones compuestas de GPU (Fase 3, C2): cada uno solo,
# 1 ciclo, con el driver composite_known.py (checksum verificado). No toca frecuencias. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_gpu_kernel_survey
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_kernel_survey_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_kernel_survey_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
KERNEL_ROOT="$HOME/hyperion-kernels"
WORK="$ROOT/fase3_daemon/composite_apps/survey_work"; mkdir -p "$WORK"
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
# vistas en el entrenamiento (A): compute y memory largos; NO vistas (B): familias Rodinia del conjunto sellado de agosto
for K in gpu_cutlass_simt_dgemm_n4096 gpu_cutlass_simt_conv2d rodinia_kmeans_gpu_N350000_D34_K200 rodinia_heartwall \
         gpu_rajaperf_stream_triad gpu_rajaperf_stream_copy dual_axpy_gpu_N1280000000 dual_spmv_gpu_N200000000 dual_stencil_gpu_N36864 \
         rodinia_lavamd rodinia_myocyte rodinia_backprop rodinia_dwt2d rodinia_dwt2d_s8192 gpu_rajaperf_reduce3_int gpu_rajaperf_indexlist_3loop; do
  ( cd "$KERNEL_ROOT" && python3 "$ROOT/fase3_daemon/composite_apps/composite_known.py" --node-id pacca-a100 \
      --kernels-root "$KERNEL_ROOT" --sequence $K --cycles 1 --boundaries-out "$WORK/survey_$K.jsonl" 2>&1 \
      | grep -E "^\[ciclo|FALL|Error|no encontrados|C02|RuntimeError" | head -2 ) || echo "  $K: no se pudo correr"
done
