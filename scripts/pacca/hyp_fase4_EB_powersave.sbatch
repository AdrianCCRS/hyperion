#!/bin/bash
# E-B (familias inéditas de memoria) bajo intel_pstate powersave con EPP default, para contrastar con el estado nativo performance del resto de la
# Fase 4. Mismos cinco brazos, kernels y 8 bloques aleatorizados que la E-B en performance (REF, REF sin turbo, sombra, activo y activo con agente
# de CPU en observación). Restaura governor/EPP originales al terminar (trap). Si no se puede fijar powersave, NO mide: se queda en pausa para
# diagnosticar (patrón holder), sin soltar el nodo. Liberar con: touch $OUT/HOLDER_STOP o scancel.
# No usar localmente. Sin --time por política operativa de pacca; se detiene solo con scancel si hace falta.
#SBATCH --job-name=hyp_fase4_EB_powersave
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
export OUT="${OUT:-$HOME/hyperion-results/final/fase4_EB_powersave_${SLURM_JOB_ID}}"
mkdir -p "$OUT"
CPUFREQ=/sys/devices/system/cpu
CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
STOP="$OUT/HOLDER_STOP"; rm -f "$STOP"
state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT/powersave_status.txt"; }

original_governor=$(cat "$CPUFREQ/cpu0/cpufreq/scaling_governor")
original_epp=$(cat "$CPUFREQ/cpu0/cpufreq/energy_performance_preference")
restore() { sudo -n /usr/local/bin/set_cpu_gov "$original_governor" "$original_epp" && state "restaurado governor=$original_governor epp=$original_epp"; }
trap restore EXIT

hold() {
    state "PAUSA: $1"; state "adjuntar: srun --jobid=$SLURM_JOB_ID --pty bash | liberar: touch $STOP o scancel $SLURM_JOB_ID"
    while [[ ! -f "$STOP" ]]; do sleep 60; done
    exit 2
}
dump() { for c in $CPUS; do echo "cpu$c gov=$(cat $CPUFREQ/cpu$c/cpufreq/scaling_governor 2>&1) epp=$(cat $CPUFREQ/cpu$c/cpufreq/energy_performance_preference 2>&1)"; done; }

state "nodo=$(hostname) job=$SLURM_JOB_ID original: governor=$original_governor epp=$original_epp"
ok=0
for try in 1 2 3; do
    sudo -n /usr/local/bin/set_cpu_gov powersave default 2>&1 | tee -a "$OUT/powersave_status.txt"
    sleep 2
    if [ "$(dump | grep -vc 'gov=powersave epp=default')" -eq 0 ]; then ok=1; break; fi
    state "intento $try: no todos los cores quedaron en powersave/default"
done
dump | tee -a "$OUT/powersave_status.txt"
[ "$ok" = 1 ] || hold "no se pudo fijar powersave/default en los 12 cores"
state "powersave/default verificado en los 12 cores; inicia E-B"

export ONLY_SCOPES=gpumem ONLY_SETS=unseen
export ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs"
export REPS_GENERAL=8 CYCLES_GPUMEM=1 GAP_S=1
export RANDOM_SEED=20260929 ORDER_MODE=balanced_blocks
bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"
rc=$?
state "matriz terminó rc=$rc; governor final antes de restaurar: $(cat $CPUFREQ/cpu0/cpufreq/scaling_governor)"
[ "$rc" -eq 0 ] || hold "matriz falló rc=$rc (governor sigue en powersave hasta soltar)"
