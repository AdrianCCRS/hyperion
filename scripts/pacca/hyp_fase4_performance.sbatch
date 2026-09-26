#!/bin/bash
# Fase 4 completa bajo el estado nativo actual del nodo, sin holder.
# El job termina al completar la serie o falla de forma visible en Slurm.
# Sin --time por política operativa de pacca.
#SBATCH --job-name=hyp_fase4_performance
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err

set -euo pipefail

ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
RESULTS_ROOT="$HOME/hyperion-results/final"
BASE="${OUT_BASE:-$RESULTS_ROOT/fase4_performance_${SLURM_JOB_ID}}"
mkdir -p "$BASE"

state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$BASE/campaign_status.txt"; }

run_stage() {
    local name=$1
    shift
    state "INICIO $name"
    if "$@" >"$BASE/${name}.log" 2>&1; then
        state "OK $name"
    else
        state "FALLO $name; ver $BASE/${name}.log"
        return 1
    fi
}

run_matrix() {
    local name=$1
    shift
    env OUT="$BASE/$name" "$@" bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

[[ -f "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" ]] || {
    state "FALLO no existe launcher de matriz en $ROOT"
    exit 1
}

state "ASIGNACION TOMADA nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT base=$BASE"
for cpu in 0 1 2 3 4 5 16 17 18 19 20 21; do
    state "cpu$cpu governor=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor)"
done

run_stage A run_matrix A \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=5 \
    ONLY_SCOPES="cpu gpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor"
run_stage C run_matrix C \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor"
run_stage noturbo run_matrix noturbo \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo activo"
run_stage E run_matrix E \
    CYCLES=2 GAP_S=1 CYCLES_GPUMEM=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="gpumem" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs fijo_gpu_f1"
run_stage EA_confirmatorio env OUT="$BASE/EA_confirmatorio" HYPERION_ROOT="$ROOT" \
    bash "$ROOT/scripts/pacca/hyp_fase4_EA_confirmatorio.sbatch"
run_stage D run_matrix D \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="lammps" ONLY_SETS="known unseen" LAMMPS_SETS="lj eam chain rhodo" \
    ONLY_ARMS="base sombra activo_gpu activo_gpu_cpuobs" \
    STEPS_lj=4400 STEPS_eam=3200 STEPS_chain=8000 STEPS_rhodo=1500

state "TODAS LAS ETAPAS TERMINARON"
