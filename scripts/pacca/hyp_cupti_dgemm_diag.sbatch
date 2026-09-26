#!/bin/bash
# Diagnostico corto (~1 min): por que CUPTI Activity registra 0 eventos con
# gpu_cutlass_simt_dgemm (campana gpu_cupti_20260923, 84/84 rechazadas por I10).
# Compara LD_PRELOAD sobre el ELF directo, sobre el wrapper bash y sobre el
# ELF de un kernel de control (gaussian) que si funciono. Sin --time.
#SBATCH --job-name=hyp_cupti_dgemm_diag
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cupti_dgemm_diag_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cupti_dgemm_diag_%j.err
set -uo pipefail
K="$HOME/hyperion-kernels"
LIB="$HOME/hyperion/common/telemetry/build-cupti/libhyperion_cupti_activity.so"
OUT="$HOME/hyperion-results/final/cupti_dgemm_diag_$SLURM_JOB_ID"
mkdir -p "$OUT"
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
ELF="$K/libexec/gpu/cutlass_simt_dgemm_bench"
ARGS="--size 2048 --iterations 40"
run() { # nombre, comando...
  local n=$1; shift
  echo "=== $n"
  HYPERION_CUPTI_ACTIVITY_FILE="$OUT/$n.csv" LD_PRELOAD="$LIB" "$@" 2>&1 | tail -3
  echo "lineas_csv=$(wc -l < "$OUT/$n.csv" 2>/dev/null || echo NOFILE)"
}
run A_elf_directo "$ELF" $ARGS
run B_wrapper_bash "$K/bin/gpu_cutlass_simt_dgemm" $ARGS
run C_control_gaussian "$K/bin/rodinia_gaussian" -s 1024
echo "=== D dependencias del ELF"; ldd "$ELF" | grep -i "cuda\|cupti\|nvidia" || echo "sin libcudart/libcuda dinamicos"
nm -D "$ELF" 2>/dev/null | grep -ci cudaLaunch || true
