#!/bin/bash
# Escenario E de la Fase 4: duracion de los candidatos a fases memory_bound de GPU (catalogo y argumentos escalados).
# No toca frecuencias. NUNCA local. Sin --time. Plan_Fase4_Escenario_E.md, paso 1.
#SBATCH --job-name=hyp_fase4_E_survey
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/fase4_E_survey_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/fase4_E_survey_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
python3 "$ROOT/fase3_daemon/composite_apps/survey_kernels.py" --node-id pacca-a100 --kernels-root "$KERNEL_ROOT" \
  dual_stencil_gpu_N36864 dual_spmv_gpu_N200000000 dual_axpy_gpu_N1280000000 gpu_cutlass_simt_dgemm_n4096 rodinia_lavamd \
  "gpu_stream_bw::--arraysize 100000000 --numtimes 3000" "gpu_stream_bw::--arraysize 100000000 --numtimes 5000" \
  gpu_rajaperf_indexlist_3loop gpu_rajaperf_reduce3_int \
  "rodinia_backprop::4194240" "rodinia_backprop::8388480" "rodinia_myocyte::400000 1 0" "rodinia_myocyte::1000000 1 0"
