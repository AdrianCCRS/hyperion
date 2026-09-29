#!/bin/bash
# Fase 4 bajo intel_pstate powersave (EPP default), contraparte de la serie performance del job 7696 en las etapas que van a
# figuras: A, E, confirmatorio E-A, D y CloverLeaf (C y noturbo se omiten, son controles de repetibilidad).
# Estado de CPU en los 12 núcleos delegados: governor powersave, EPP default, turbo apagado y rango 0.8-3.2 GHz. Igual que
# 7696 en turbo (apagado) y techo (3.2 GHz); lo único distinto es que el governor elige la frecuencia dentro del rango
# (7696 tenía min=max=3.2 GHz, donde el governor no tiene margen).
# Antes de medir: pre-vuelo corto (agente de CPU bajo powersave, que nunca se probó, y LAMMPS). Si algo falla, NO suelta el
# nodo: queda en pausa para diagnosticar (patrón holder). Al terminar restaura governor, EPP, turbo y min/max por núcleo
# desde el snapshot tomado al inicio.
# Adjuntar shell (solo inspección mientras mide): srun --jobid=<id> --overlap --pty bash
# Liberar una pausa: touch $OUT_BASE/HOLDER_STOP o scancel. Sin --time por política operativa de pacca.
#SBATCH --job-name=hyp_fase4_powersave
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
export OUT_BASE="${OUT_BASE:-$HOME/hyperion-results/final/fase4_powersave_${SLURM_JOB_ID}}"
mkdir -p "$OUT_BASE"
C=/sys/devices/system/cpu
CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
MIN_KHZ=800000 MAX_KHZ=3200000
STOP="$OUT_BASE/HOLDER_STOP"; rm -f "$STOP"
state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT_BASE/campaign_status.txt"; }
dump() { echo "no_turbo=$(cat $C/intel_pstate/no_turbo)"; for c in $CPUS; do d=$C/cpu$c/cpufreq
    echo "cpu$c gov=$(cat $d/scaling_governor) epp=$(cat $d/energy_performance_preference) min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq)"; done; }

dump > "$OUT_BASE/cpu_state_original.txt"
original_governor=$(cat $C/cpu0/cpufreq/scaling_governor)
original_epp=$(cat $C/cpu0/cpufreq/energy_performance_preference)
original_no_turbo=$(cat $C/intel_pstate/no_turbo)
restore() {
    sudo -n /usr/local/bin/set_cpu_gov "$original_governor" "$original_epp"
    sudo -n /usr/local/bin/set_turbo_state "$original_no_turbo"
    while read -r line; do case "$line" in cpu*)
        c=${line%% *}; c=${c#cpu}; mn=$(echo "$line" | sed -n 's/.*min=\([0-9]*\).*/\1/p'); mx=$(echo "$line" | sed -n 's/.*max=\([0-9]*\).*/\1/p')
        d=$C/cpu$c/cpufreq; echo "$mx" > $d/scaling_max_freq; echo "$mn" > $d/scaling_min_freq; echo "$mx" > $d/scaling_max_freq;; esac
    done < "$OUT_BASE/cpu_state_original.txt"
    dump > "$OUT_BASE/cpu_state_restaurado.txt"
    if diff -q <(sort "$OUT_BASE/cpu_state_original.txt") <(sort "$OUT_BASE/cpu_state_restaurado.txt") >/dev/null; then
        state "estado de CPU restaurado al original"
    else
        state "AVISO: el estado restaurado difiere del original"; diff "$OUT_BASE/cpu_state_original.txt" "$OUT_BASE/cpu_state_restaurado.txt" >> "$OUT_BASE/campaign_status.txt"
    fi
}
trap restore EXIT

hold() {
    state "PAUSA: $1"; state "adjuntar: srun --jobid=$SLURM_JOB_ID --overlap --pty bash | liberar: touch $STOP o scancel $SLURM_JOB_ID"
    while [[ ! -f "$STOP" ]]; do sleep 60; done
    exit 2
}

state "nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT original: governor=$original_governor epp=$original_epp no_turbo=$original_no_turbo"

# EPP default: intel_pstate (modo active) relee el preset real de la plataforma (balance_performance bajo powersave en
# paccaA100, verificado en 7779), no la cadena "default"; se exige EPP uniforme en los 12 núcleos.
ok=0
for try in 1 2 3; do
    sudo -n /usr/local/bin/set_cpu_gov powersave default >> "$OUT_BASE/campaign_status.txt" 2>&1
    sudo -n /usr/local/bin/set_turbo_state 1
    for c in $CPUS; do d=$C/cpu$c/cpufreq; echo $MAX_KHZ > $d/scaling_max_freq; echo $MIN_KHZ > $d/scaling_min_freq; done
    sleep 2
    govs=$(dump | grep -oP 'gov=\K\S+' | sort -u); epps=$(dump | grep -oP 'epp=\K\S+' | sort -u)
    mins=$(dump | grep -oP 'min=\K\S+' | sort -u); maxs=$(dump | grep -oP 'max=\K\S+' | sort -u)
    if [ "$govs" = powersave ] && [ "$(echo "$epps" | wc -l)" -eq 1 ] && [ "$(cat $C/intel_pstate/no_turbo)" = 1 ] \
       && [ "$mins" = $MIN_KHZ ] && [ "$maxs" = $MAX_KHZ ]; then ok=1; break; fi
    state "intento $try: gov=[$govs] epp=[$epps] no_turbo=$(cat $C/intel_pstate/no_turbo) min=[$mins] max=[$maxs]"
done
dump | tee "$OUT_BASE/cpu_state_powersave.txt" >> "$OUT_BASE/campaign_status.txt"
[ "$ok" = 1 ] || hold "no se pudo fijar powersave/default, turbo off y 0.8-3.2 GHz de forma uniforme en los 12 núcleos"
state "powersave verificado: EPP=$epps no_turbo=1 min=$MIN_KHZ max=$MAX_KHZ"

run_stage() {
    local name=$1; shift
    state "INICIO $name"
    if "$@" > "$OUT_BASE/$name.log" 2>&1; then state "OK $name"; else hold "FALLO $name; ver $OUT_BASE/$name.log"; fi
    # el estado de CPU tiene que seguir en powersave después de cada etapa
    dump > "$OUT_BASE/cpu_state_tras_$name.txt"
    diff -q "$OUT_BASE/cpu_state_powersave.txt" "$OUT_BASE/cpu_state_tras_$name.txt" >/dev/null \
        || hold "el estado de CPU cambió tras $name; ver cpu_state_tras_$name.txt"
}
run_matrix() { local name=$1; shift; env OUT="$OUT_BASE/$name" "$@" bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch"; }
# columnas: ...,app_rc($9),rc_cpu_daemon,rc_gpu_daemon,state_ok($NF); exige al menos una celda, app_rc=0 y state_ok=1
all_ok() { awk -F, 'NR>1 {n++; if ($9!=0 || $NF!=1) bad=1} END {exit (bad || n==0)}' "$1"; }

# Pre-vuelo: agente de CPU activo (escribe min/max) bajo powersave y una celda de LAMMPS.
run_stage preflight run_matrix preflight \
    CYCLES=1 GAP_S=1 REPS_GENERAL=1 REPS_JOINT=1 \
    ONLY_SCOPES="cpu joint" ONLY_SETS="known" ONLY_ARMS="base activo"
all_ok "$OUT_BASE/preflight/results.csv" || hold "pre-vuelo con app_rc!=0 o state_ok=0; ver preflight/results.csv"
run_stage preflight_lammps run_matrix preflight_lammps \
    CYCLES=1 GAP_S=1 REPS_GENERAL=1 REPS_JOINT=1 \
    ONLY_SCOPES="lammps" ONLY_SETS="known" LAMMPS_SETS="lj" ONLY_ARMS="base" STEPS_lj=4400
all_ok "$OUT_BASE/preflight_lammps/results.csv" || hold "pre-vuelo LAMMPS falló; ver preflight_lammps/results.csv"
state "pre-vuelo OK"

run_stage A run_matrix A \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=5 \
    ONLY_SCOPES="cpu gpu joint" ONLY_SETS="known unseen" \
    ONLY_ARMS="base sombra activo activo_f0 activo_nofloor"
run_stage E run_matrix E \
    CYCLES=2 GAP_S=1 CYCLES_GPUMEM=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="gpumem" ONLY_SETS="known unseen" \
    ONLY_ARMS="base base_noturbo sombra activo_gpu activo_gpu_cpuobs fijo_gpu_f1"
run_stage EA_confirmatorio env OUT="$OUT_BASE/EA_confirmatorio" HYPERION_ROOT="$ROOT" \
    bash "$ROOT/scripts/pacca/hyp_fase4_EA_confirmatorio.sbatch"
run_stage D run_matrix D \
    CYCLES=2 GAP_S=1 REPS_GENERAL=3 REPS_JOINT=3 \
    ONLY_SCOPES="lammps" ONLY_SETS="known unseen" LAMMPS_SETS="lj eam chain rhodo" \
    ONLY_ARMS="base sombra activo_gpu activo_gpu_cpuobs" \
    STEPS_lj=4400 STEPS_eam=3200 STEPS_chain=8000 STEPS_rhodo=1500
run_stage cloverleaf env OUT="$OUT_BASE/cloverleaf" HYPERION_ROOT="$ROOT" \
    bash "$ROOT/scripts/pacca/hyp_cloverleaf_confirm.sbatch"

state "TODAS LAS ETAPAS TERMINARON"
