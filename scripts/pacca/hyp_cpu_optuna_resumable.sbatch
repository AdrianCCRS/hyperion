#!/bin/bash
# Búsqueda CPU final reanudable. No usar --time: la campaña se controla
# manualmente y el estado queda persistido tras cada trial.
#SBATCH --job-name=hyp_cpu_optuna_resumable
#SBATCH --partition=normal
#SBATCH --nodelist=pacca05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_optuna_resumable_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_optuna_resumable_%j.err

set -euo pipefail

export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Evita paralelismo anidado: los ensembles usan exactamente --n-jobs; las
# bibliotecas numéricas no deben abrir pools adicionales por cada worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$HOME/hyperion-venv/bin/activate"

RUN_ROOT="$HOME/hyperion-results/final/training/cpu_optuna_resumable_20260917"
PROGRESS="$RUN_ROOT/progress.json"
STORAGE="$RUN_ROOT/optuna_trials.sqlite3"
MODELS="$HOME/hyperion/fase2_clasificador/models/cpu_optuna_resumable_20260917"
mkdir -p "$RUN_ROOT" "$MODELS" "$HOME/hyperion-results/final/logs"

KERNELS="npb_bt,npb_mg,npb_cg,npb_sp,npb_ft,npb_lu,dgemm_n2048,rodinia_lavamd_omp,rajaperf_polybench_3mm_omp,cpu_rajaperf_stream_mul,cpu_rajaperf_stream_triad,cpu_rajaperf_stream_add,cpu_rajaperf_lcals_first_sum,cpu_rajaperf_lcals_tridiag_elim,cpu_rajaperf_polybench_jacobi_1d,cpu_rajaperf_polybench_fdtd_2d,cpu_rajaperf_basic_daxpy,cpu_rajaperf_basic_init3,cpu_lulesh,cpu_hpcg,cpu_gap_pr,cpu_cholmod,phasic_p010,phasic_p100,phasic_p1000,dual_gemm_cpu_N2048,dual_fft_cpu_N128,dual_fft_cpu_N1024,dual_fft_cpu_N4096,dual_axpy_cpu_N100000,dual_axpy_cpu_N3162278,dual_axpy_cpu_N10000000,dual_stencil_cpu_N256,dual_stencil_cpu_N512,dual_stencil_cpu_N1024,dual_stencil_cpu_N1536,dual_cholesky_cpu_N256,dual_cholesky_cpu_N1024,dual_cholesky_cpu_N2048,dual_spmv_cpu_N100000,dual_spmv_cpu_N1000000,hpccg_cpu_N46,dual_fft_cpu_N472"
LEVELS="REF,F0,F1,F2,F3,F4,F5,F6,F7,F8"

echo "=== Búsqueda CPU Optuna reanudable ==="
date --iso-8601=seconds
echo "run_root=$RUN_ROOT"
echo "progress=$PROGRESS"
echo "storage=$STORAGE"
echo "cpus=96 n_jobs=96 per_run_sample=2000 n_trials=30"

python3 "$HOME/hyperion/fase2_clasificador/training/train_phase.py" \
  --campaign-dir "$HOME/hyperion-results/final/campaigns/cpu" \
  --campaign-id pacca_cpu_final_20260913 \
  --kernels "$KERNELS" \
  --levels "$LEVELS" \
  --per-run-sample 2000 \
  --n-trials 30 \
  --n-jobs 96 \
  --optuna-storage "$STORAGE" \
  --progress-file "$PROGRESS" \
  --output-dir "$MODELS"

echo CPU_OPTUNA_RESUMABLE_DONE
date --iso-8601=seconds
