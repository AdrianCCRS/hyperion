#!/bin/bash
# Confirmación prospectiva de E-A: compara REF, sombra, daemon GPU activo y F1 fijo en la misma aplicación.
# No usar localmente. No tiene --time: se detiene manualmente solo si el usuario lo ordena.
#SBATCH --job-name=hyp_fase4_EA_confirm
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
export ONLY_SETS=known
export ONLY_ARMS="base sombra activo_gpu fijo_gpu_f1"
export REPS_GENERAL=5
export CYCLES_GPUMEM=1
export GAP_S=1
export RANDOM_SEED=20260925
export ORDER_MODE=balanced_blocks
export OUT="${OUT:-$HOME/hyperion-results/final/fase4_EA_confirmatorio}"

exec bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
