#!/bin/bash
# E-B (familias inéditas de memoria): solo los dos brazos del escenario E que faltaban, REF sin turbo y agente GPU activo con el agente de
# CPU en observación, en bloques aleatorizados. REF, sombra y activo ya están medidos (etapa E, repetición de 8 bloques y diagnóstico) y no
# se repiten. Con las 3 repeticiones de la etapa E quedan 11 por brazo, como el brazo activo. La fase de myocyte tiene variabilidad propia
# (~41 s o ~50 s), por eso son 8 bloques.
# No usar localmente. Sin --time por política operativa de pacca; se detiene solo con scancel si hace falta.
#SBATCH --job-name=hyp_fase4_EB_faltantes
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
export ONLY_ARMS="base_noturbo activo_gpu_cpuobs"
export REPS_GENERAL=8
export CYCLES_GPUMEM=1
export GAP_S=1
export RANDOM_SEED=20260928
export ORDER_MODE=balanced_blocks
export OUT="${OUT:-$HOME/hyperion-results/final/fase4_EB_faltantes_${SLURM_JOB_ID}}"

exec bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
