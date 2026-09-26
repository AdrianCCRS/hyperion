#!/bin/bash
# Reejecuta la evaluación de Fase 4 en una sola asignación exclusiva.
# Cada etapa conserva un directorio nuevo y el holder queda vivo tras un fallo
# o al terminar, para diagnosticar o liberar la asignación manualmente.
# No usar localmente. Sin --time.
#SBATCH --job-name=hyp_fase4_complete
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
RESULTS_ROOT="$HOME/hyperion-results/final"
BASE="${OUT_BASE:-$RESULTS_ROOT/fase4_remeasure_${SLURM_JOB_ID}}"
STATUS="$BASE/holder_status.txt"
STOP="$BASE/HOLDER_STOP"

mkdir -p "$BASE"
rm -f "$STOP"

state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"; }

hold() {
  state "PAUSA: $1"
  state "adjuntar: srun --jobid=$SLURM_JOB_ID --pty bash"
  state "liberar: touch $STOP o scancel $SLURM_JOB_ID"
  while [[ ! -f "$STOP" ]]; do sleep 60; done
  state "HOLDER_STOP detectado"
  exit 2
}

run_stage() {
  local name="$1"
  shift
  state "INICIO $name"
  if "$@" >"$BASE/${name}.log" 2>&1; then
    state "OK $name"
  else
    state "FALLO $name, ver $BASE/${name}.log"
    hold "$name"
  fi
}

matrix() {
  bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

run_a() {
  env OUT="$BASE/A" CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=5 \
    ONLY_SCOPES="cpu gpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor" \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

run_c() {
  env OUT="$BASE/C" CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor" \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

run_noturbo() {
  env OUT="$BASE/noturbo" CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo activo" \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

run_e() {
  env OUT="$BASE/E" CYCLES=2 GAP_S=1 CYCLES_GPUMEM=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="gpumem" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs fijo_gpu_f1" \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

run_ea() {
  env OUT="$BASE/EA_confirmatorio" HYPERION_ROOT="$ROOT" \
    bash "$ROOT/scripts/pacca/hyp_fase4_EA_confirmatorio.sbatch"
}

run_d() {
  env OUT="$BASE/D" CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="lammps" ONLY_SETS="known unseen" LAMMPS_SETS="lj eam chain rhodo" \
    ONLY_ARMS="base sombra activo_gpu activo_gpu_cpuobs" \
    STEPS_lj=4400 STEPS_eam=3200 STEPS_chain=8000 STEPS_rhodo=1500 \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
}

[[ -d "$ROOT" ]] || hold "no existe ROOT=$ROOT"
[[ -f "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" ]] || hold "no existe launcher de matriz"

state "ASIGNACION TOMADA nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT base=$BASE"
taskset -c 0-5 bash -c 'grep "^Cpus_allowed_list:" /proc/self/status' | tee -a "$STATUS"

run_stage A run_a
run_stage C run_c
run_stage noturbo run_noturbo
run_stage E run_e
run_stage EA_confirmatorio run_ea
run_stage D run_d

state "TODAS LAS ETAPAS TERMINARON"
state "adjuntar: srun --jobid=$SLURM_JOB_ID --pty bash"
state "liberar: touch $STOP o scancel $SLURM_JOB_ID"
while [[ ! -f "$STOP" ]]; do sleep 60; done
state "HOLDER_STOP detectado"
