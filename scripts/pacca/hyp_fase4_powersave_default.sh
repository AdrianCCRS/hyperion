#!/bin/bash
# Ejecuta la misma serie de Fase 4 bajo intel_pstate powersave con EPP default.
# Debe invocarse dentro de una asignación exclusiva ya obtenida. No usa sbatch.

set -euo pipefail

ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
OUT_BASE="${OUT_BASE:?defina OUT_BASE para aislar esta réplica}"
CPUFREQ=/sys/devices/system/cpu
CPUS="0 1 2 3 4 5 16 17 18 19 20 21"

original_governor=$(cat "$CPUFREQ/cpu0/cpufreq/scaling_governor")
original_epp=$(cat "$CPUFREQ/cpufreq/policy0/energy_performance_preference")

restore_governor() {
    sudo -n /usr/local/bin/set_cpu_gov "$original_governor" "$original_epp" || {
        echo "No se pudo restaurar governor=$original_governor epp=$original_epp" >&2
        return 1
    }
}
trap restore_governor EXIT

sudo -n /usr/local/bin/set_cpu_gov powersave default
for cpu in $CPUS; do
    observed=$(cat "$CPUFREQ/cpu$cpu/cpufreq/scaling_governor")
    [ "$observed" = powersave ] || {
        echo "cpu$cpu conserva governor=$observed; se esperaba powersave" >&2
        exit 1
    }
done
observed_epp=$(cat "$CPUFREQ/cpufreq/policy0/energy_performance_preference")
[ "$observed_epp" = default ] || {
    echo "EPP observado=$observed_epp; se esperaba default" >&2
    exit 1
}

run_matrix() {
    local name=$1
    shift
    echo "[$(date -Is)] INICIO $name"
    env OUT="$OUT_BASE/$name" "$@" bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
    echo "[$(date -Is)] OK $name"
}

run_matrix A \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=5 \
    ONLY_SCOPES="cpu gpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor"
run_matrix C \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor"
run_matrix noturbo \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo activo"
run_matrix E \
    CYCLES=2 GAP_S=1 CYCLES_GPUMEM=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="gpumem" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs fijo_gpu_f1"

echo "[$(date -Is)] INICIO EA_confirmatorio"
env OUT="$OUT_BASE/EA_confirmatorio" HYPERION_ROOT="$ROOT" \
    bash "$ROOT/scripts/pacca/hyp_fase4_EA_confirmatorio.sbatch"
echo "[$(date -Is)] OK EA_confirmatorio"

run_matrix D \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="lammps" ONLY_SETS="known unseen" LAMMPS_SETS="lj eam chain rhodo" \
    ONLY_ARMS="base sombra activo_gpu activo_gpu_cpuobs" \
    STEPS_lj=4400 STEPS_eam=3200 STEPS_chain=8000 STEPS_rhodo=1500
