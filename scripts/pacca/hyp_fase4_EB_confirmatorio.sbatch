#!/bin/bash
# Repetición ampliada de E-B (familias inéditas de memoria): compara REF, sombra y daemon GPU activo con bloques aleatorizados.
# Mismos kernels que E-B; solo aumenta las repeticiones para poder comparar sin que la bimodalidad de myocyte se mezcle con el brazo.
# No usar localmente. No tiene --time: se detiene manualmente solo si el usuario lo ordena.
#SBATCH --job-name=hyp_fase4_EB_confirm
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

ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
export ONLY_SCOPES=gpumem
export ONLY_SETS=unseen
export ONLY_ARMS="base sombra activo_gpu"
export REPS_GENERAL=8
export CYCLES_GPUMEM=1
export GAP_S=1
export RANDOM_SEED=20260927
export ORDER_MODE=balanced_blocks
export OUT="${OUT:-$HOME/hyperion-results/final/fase4_EB_confirmatorio}"

exec bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
