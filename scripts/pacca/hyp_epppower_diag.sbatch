#!/bin/bash
# Diagnostico exploratorio: ¿EPP=power se comporta distinto de EPP=default (balance_performance, medido en el job 7789) bajo
# intel_pstate/powersave en esta plataforma? NO es para el libro todavia.
# Fase 1 (unos 2 min): carga sintetica sostenida (busy loop) en los 12 nucleos delegados bajo EPP=balance_performance y bajo
#   EPP=power, muestreando scaling_cur_freq y potencia de paquete (RAPL). Sin tocar el framework de campanas.
# Gate manual: el job se pausa despues del diagnostico. Si el resultado es negativo (no hay diferencia de frecuencia/potencia
#   entre los dos EPP), cancelar aqui con `touch $OUT_BASE/HOLDER_STOP` o `scancel <jobid>`. Si hay diferencia real, continuar
#   con `touch $OUT_BASE/CONTINUE` para medir el confirmatorio E-A (mismo diseno que EA_confirmatorio de 7789) bajo EPP=power
#   y comparar el EDP contra el 0.974 (IC95 0.968-0.980) ya medido con EPP=default.
# AUTO_CONTINUE=1 omite el gate y pasa directo al confirmatorio (a plena carga el EPP no cambia nada; su efecto bajo
#   carga parcial solo se ve en el confirmatorio).
# Sin --time por politica operativa de pacca (cuenta compartida latorresn). Adjuntar shell de solo inspeccion mientras corre:
#   srun --jobid=<id> --overlap --pty bash
#SBATCH --job-name=hyp_epppower_diag
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
export OUT_BASE="${OUT_BASE:-$HOME/hyperion-results/final/epppower_diag_${SLURM_JOB_ID}}"
mkdir -p "$OUT_BASE"
C=/sys/devices/system/cpu
CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
MIN_KHZ=800000 MAX_KHZ=3200000
RAPL=/sys/class/powercap/intel-rapl:0
STOP="$OUT_BASE/HOLDER_STOP"; CONT="$OUT_BASE/CONTINUE"; rm -f "$STOP" "$CONT"
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
    state "PAUSA: $1"
    state "adjuntar: srun --jobid=$SLURM_JOB_ID --overlap --pty bash | liberar: touch $STOP o scancel $SLURM_JOB_ID"
    while [[ ! -f "$STOP" ]]; do sleep 60; done
    exit 2
}

gate() {
    state "PAUSA (gate manual): $1"
    state "continuar con EA_confirmatorio bajo EPP=power: touch $CONT | terminar aqui: touch $STOP o scancel $SLURM_JOB_ID"
    while [[ ! -f "$CONT" && ! -f "$STOP" ]]; do sleep 30; done
    if [[ -f "$STOP" ]]; then state "detenido en el gate por decision del usuario"; exit 0; fi
    state "continuando tras el gate"
}

set_state() {
    local epp="$1" ok=0
    for try in 1 2 3; do
        sudo -n /usr/local/bin/set_cpu_gov powersave "$epp" >> "$OUT_BASE/campaign_status.txt" 2>&1
        sudo -n /usr/local/bin/set_turbo_state 1
        for c in $CPUS; do d=$C/cpu$c/cpufreq; echo $MAX_KHZ > $d/scaling_max_freq; echo $MIN_KHZ > $d/scaling_min_freq; done
        sleep 2
        local govs epps mins maxs
        govs=$(dump | grep -oP 'gov=\K\S+' | sort -u); epps=$(dump | grep -oP 'epp=\K\S+' | sort -u)
        mins=$(dump | grep -oP 'min=\K\S+' | sort -u); maxs=$(dump | grep -oP 'max=\K\S+' | sort -u)
        if [ "$govs" = powersave ] && [ "$(echo "$epps" | wc -l)" -eq 1 ] && [ "$(cat $C/intel_pstate/no_turbo)" = 1 ] \
           && [ "$mins" = $MIN_KHZ ] && [ "$maxs" = $MAX_KHZ ]; then ok=1; break; fi
        state "intento $try (epp=$epp): gov=[$govs] epp=[$epps] no_turbo=$(cat $C/intel_pstate/no_turbo) min=[$mins] max=[$maxs]"
    done
    [ "$ok" = 1 ] || hold "no se pudo fijar powersave/$epp, turbo off y 0.8-3.2 GHz de forma uniforme"
    state "estado verificado: epp=$(dump | grep -oP 'epp=\K\S+' | sort -u) no_turbo=1 min=$MIN_KHZ max=$MAX_KHZ"
}

diag() {
    local epp="$1" secs="${2:-20}"
    set_state "$epp"
    local pids=()
    for c in $CPUS; do taskset -c "$c" bash -c 'while :; do :; done' & pids+=($!); done
    sleep 2
    local e0 t0 e1 t1 freqs=()
    e0=$(cat "$RAPL/energy_uj"); t0=$(date +%s.%N)
    for _ in $(seq 1 "$secs"); do freqs+=("$(cat $C/cpu0/cpufreq/scaling_cur_freq)"); sleep 1; done
    e1=$(cat "$RAPL/energy_uj"); t1=$(date +%s.%N)
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    local dt de watts range
    dt=$(echo "$t1 - $t0" | bc)
    de=$((e1 - e0))
    if [ "$de" -lt 0 ]; then range=$(cat "$RAPL/max_energy_range_uj"); de=$((de + range)); fi
    watts=$(echo "scale=2; $de / 1000000 / $dt" | bc)
    state "diag epp=$epp cpu0_freq_khz=[${freqs[*]}] potencia_paquete_w=$watts dt_s=$dt"
    dump >> "$OUT_BASE/cpu_state_diag_$epp.txt"
}

state "nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT original: governor=$original_governor epp=$original_epp no_turbo=$original_no_turbo"

diag balance_performance 20
diag power 20

if [[ "${AUTO_CONTINUE:-0}" = 1 ]]; then
    state "diagnostico terminado (AUTO_CONTINUE=1, sin gate manual); ver cpu0_freq_khz y potencia_paquete_w de cada EPP"
else
    gate "diagnostico terminado; ver campaign_status.txt (cpu0_freq_khz y potencia_paquete_w de cada EPP)"
fi

# Solo se llega aqui si se tocó CONTINUE: confirmatorio E-A completo bajo EPP=power, mismo diseño que EA_confirmatorio del job 7789.
set_state power
state "INICIO EA_confirmatorio_epppower"
if env OUT="$OUT_BASE/EA_confirmatorio_epppower" HYPERION_ROOT="$ROOT" bash "$ROOT/scripts/pacca/hyp_fase4_EA_confirmatorio.sbatch" > "$OUT_BASE/EA_confirmatorio_epppower.log" 2>&1; then
    state "OK EA_confirmatorio_epppower"
else
    hold "FALLO EA_confirmatorio_epppower; ver EA_confirmatorio_epppower.log"
fi
dump > "$OUT_BASE/cpu_state_tras_EA.txt"

state "TODAS LAS ETAPAS TERMINARON"
