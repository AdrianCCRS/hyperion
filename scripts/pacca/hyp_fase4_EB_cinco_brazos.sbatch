#!/bin/bash
# E-B (familias inéditas de memoria) con los cinco brazos del escenario E: REF, REF sin turbo, sombra, agente GPU activo y activo con
# el agente de CPU en observación, en bloques aleatorizados. Completa las mediciones que faltaban de REF sin turbo y de activo con agente
# de CPU para que E-B se presente igual que E-A. La fase de myocyte tiene variabilidad propia (~41 s o ~50 s), por eso son 8 bloques.
# No usar localmente. Sin --time por política operativa de pacca; se detiene solo con scancel si hace falta.
#SBATCH --job-name=hyp_fase4_EB_cinco
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
export ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs"
export REPS_GENERAL=8
export CYCLES_GPUMEM=1
export GAP_S=1
export RANDOM_SEED=20260928
export ORDER_MODE=balanced_blocks
export OUT="${OUT:-$HOME/hyperion-results/final/fase4_EB_cinco_brazos_${SLURM_JOB_ID}}"

exec bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
