#!/bin/bash
# Corrida larga del escenario E de la Fase 4, protegida por la compuerta del pre-vuelo (Plan_Fase4_Escenario_E.md, pasos 4 y 5).
# Si el pre-vuelo (fase4_E_preflight) no cumple los criterios, termina antes de medir y sin tocar nada. Variables (via --export):
#   RUN_KIND = E        matriz E:   ONLY_SCOPES=gpumem, brazos base base_noturbo sombra activo_gpu activo_gpu_cpuobs -> fase4_E
#   RUN_KIND = noturbo  complemento REF sin turbo para A y C: cpu y joint, brazos base base_noturbo activo -> fase4_C_noturbo
# NUNCA local. Sin --time. Someter con: sbatch -J hyp_fase4_E_matrix (o hyp_fase4_C_noturbo) --export=ALL,RUN_KIND=...
#SBATCH --job-name=hyp_fase4_E_run
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
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; RES="$HOME/hyperion-results/final"; source "$HOME/hyperion-venv/bin/activate"
hostname; git -C "$ROOT" log --oneline -1
GATE_FLAG=""; [ "${RUN_KIND:-}" = noturbo ] && GATE_FLAG="--noturbo"  # el complemento sin turbo solo exige base_noturbo sano
python3 "$ROOT/fase4_evaluacion/gate_preflight_E.py" "$RES/fase4_E_preflight" $GATE_FLAG || { echo "compuerta cerrada: no se mide nada" >&2; exit 2; }
export CYCLES_GPUMEM=1 REPS_GENERAL=3 REPS_JOINT=3
case "${RUN_KIND:?RUN_KIND=E|noturbo}" in
  E)       export ONLY_SCOPES="gpumem" ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs" OUT="$RES/fase4_E" ;;
  noturbo) export ONLY_SCOPES="cpu joint" ONLY_ARMS="base base_noturbo activo" OUT="$RES/fase4_C_noturbo" ;;
  *) echo "RUN_KIND desconocido" >&2; exit 1 ;;
esac
exec bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
