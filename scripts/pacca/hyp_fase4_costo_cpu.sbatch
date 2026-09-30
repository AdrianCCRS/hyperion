#!/bin/bash
# Fase 4, diagnostico del costo de observacion del agente de CPU (revision C6 del director;
# docs/planeacion/Plan_Fase4_Costo_CPU.md, declarado antes de medir).
# Parte 2 (primero, porque construye el binario): aplicacion de kernels vistos (alcance cpu, set known), 6 bloques
#   aleatorizados con base, sombra, sombra_10ms, sombra_idle1ms y sombra_pin, con DIAG_SNAPSHOT=1.
# Parte 1: nodo sin aplicacion (proceso `sleep` en 0-5 como objetivo), 60 s sin agente y 60 s con el agente en sombra,
#   3 repeticiones alternadas, con la misma lectura de RAPL y los mismos snapshots.
# Estado de CPU: el de la base del libro (performance, turbo off, 3.2 GHz fijos en los delegados); se restaura al final.
# Sin --time por politica operativa de pacca (cuenta compartida latorresn). Mientras mide, una shell adjunta
# (srun --jobid=<id> --overlap --pty bash) es solo para inspeccion: nada que use CPU o GPU.
#SBATCH --job-name=hyp_fase4_costo_cpu
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

ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"
GOV=performance; SEED="${RANDOM_SEED_COSTO:-20261003}"
MIN_KHZ=3200000
export OUT_BASE="${OUT_BASE:-$HOME/hyperion-results/final/fase4_costo_cpu_${SLURM_JOB_ID}}"
mkdir -p "$OUT_BASE"
C=/sys/devices/system/cpu; CPUS="0 1 2 3 4 5 16 17 18 19 20 21"; GPU=0; KHZ=3200000
STOP="$OUT_BASE/HOLDER_STOP"; rm -f "$STOP"
state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT_BASE/campaign_status.txt"; }
dump() { echo "no_turbo=$(cat $C/intel_pstate/no_turbo)"; for c in $CPUS; do d=$C/cpu$c/cpufreq
    echo "cpu$c gov=$(cat $d/scaling_governor) epp=$(cat $d/energy_performance_preference) min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq)"; done; }

dump > "$OUT_BASE/cpu_state_original.txt"
original_governor=$(cat $C/cpu0/cpufreq/scaling_governor)
original_epp=$(cat $C/cpu0/cpufreq/energy_performance_preference)
original_no_turbo=$(cat $C/intel_pstate/no_turbo)
restore() {
    sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true
    sudo -n /usr/local/bin/set_cpu_gov "$original_governor" "$original_epp" >/dev/null 2>&1 || true
    sudo -n /usr/local/bin/set_turbo_state "$original_no_turbo"
    while read -r line; do case "$line" in cpu*)
        c=${line%% *}; c=${c#cpu}; mn=$(echo "$line" | sed -n 's/.*min=\([0-9]*\).*/\1/p'); mx=$(echo "$line" | sed -n 's/.*max=\([0-9]*\).*/\1/p')
        d=$C/cpu$c/cpufreq; echo "$mx" > $d/scaling_max_freq; echo "$mn" > $d/scaling_min_freq; echo "$mx" > $d/scaling_max_freq;; esac
    done < "$OUT_BASE/cpu_state_original.txt"
    dump > "$OUT_BASE/cpu_state_restaurado.txt"
    if diff -q "$OUT_BASE/cpu_state_original.txt" "$OUT_BASE/cpu_state_restaurado.txt" >/dev/null; then
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

# ---------- 1. estado de CPU: base del libro (performance, job 7696) o powersave EPP default (job 7789) ----------
ok=0
for try in 1 2 3; do
    if [ "$GOV" = powersave ]; then sudo -n /usr/local/bin/set_cpu_gov powersave default >> "$OUT_BASE/campaign_status.txt" 2>&1
    else [ "$(cat $C/cpu0/cpufreq/scaling_governor)" = performance ] || sudo -n /usr/local/bin/set_cpu_gov performance "$original_epp" >> "$OUT_BASE/campaign_status.txt" 2>&1; fi
    sudo -n /usr/local/bin/set_turbo_state 1
    for c in $CPUS; do d=$C/cpu$c/cpufreq; echo $KHZ > $d/scaling_max_freq; echo $MIN_KHZ > $d/scaling_min_freq; done
    sleep 2
    govs=$(dump | grep -oP 'gov=\K\S+' | sort -u); epps=$(dump | grep -oP 'epp=\K\S+' | sort -u)
    mins=$(dump | grep -oP 'min=\K\S+' | sort -u); maxs=$(dump | grep -oP 'max=\K\S+' | sort -u)
    if [ "$govs" = "$GOV" ] && [ "$(echo "$epps" | wc -l)" -eq 1 ] && [ "$(cat $C/intel_pstate/no_turbo)" = 1 ] && [ "$mins" = $MIN_KHZ ] && [ "$maxs" = $KHZ ]; then ok=1; break; fi
    state "intento $try: gov=[$govs] epp=[$epps] no_turbo=$(cat $C/intel_pstate/no_turbo) min=[$mins] max=[$maxs]"
done
dump > "$OUT_BASE/cpu_state_medicion.txt"
[ "$ok" = 1 ] || hold "no se pudo fijar $GOV, turbo off y $MIN_KHZ-$KHZ kHz de forma uniforme"
state "nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT commit=$(git -C "$ROOT" log --oneline -1) estado: $GOV epp=$(dump | grep -oP 'epp=\K\S+' | sort -u) no_turbo=1 min=$MIN_KHZ max=$KHZ"
sudo -n nvidia-smi -i $GPU -rgc >/dev/null || hold "sudo -n nvidia-smi -rgc NO permitido"


# ---------- 2. matriz: aplicacion de kernels vistos, 6 bloques ----------
M="$OUT_BASE/costo_matriz"
state "INICIO costo_matriz (6 bloques, semilla $SEED)"
env OUT="$M" HYPERION_ROOT="$ROOT" ONLY_SCOPES=cpu ONLY_SETS=known ONLY_ARMS="base sombra sombra_10ms sombra_idle1ms sombra_pin" \
    REPS_GENERAL=6 CYCLES=2 GAP_S=1 RANDOM_SEED="$SEED" ORDER_MODE=balanced_blocks DIAG_SNAPSHOT=1 \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" > "$OUT_BASE/costo_matriz.log" 2>&1 || hold "la matriz termino con error; ver costo_matriz.log"
bad=$(awk -F, 'NR>1 && ($9!="0" || $12!="1" || ($10!="0" && $10!="NA")) {print $1}' "$M/results.csv" | tr '\n' ' ')
state "costo_matriz: $(($(wc -l < "$M/results.csv")-1)) celdas; con problemas: ${bad:-ninguna}"
dump > "$OUT_BASE/cpu_state_tras_matriz.txt"
diff -q "$OUT_BASE/cpu_state_medicion.txt" "$OUT_BASE/cpu_state_tras_matriz.txt" >/dev/null || hold "el estado de CPU cambio durante la matriz"

# ---------- 1. nodo sin aplicacion: agente solo ----------
I="$OUT_BASE/reposo"; mkdir -p "$I"; CPU_BIN="$M/build/cpu_build/cpu_loop_main"; R0=/sys/class/powercap/intel-rapl:0; R1=/sys/class/powercap/intel-rapl:1
RMAX=$(cat $R0/max_energy_range_uj)
[ -x "$CPU_BIN" ] || hold "no existe $CPU_BIN"
source <(sed -n '/^snap() {/,/^}/p' "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch")
echo "cell,arm,rep,wall_s,e_cpu_j,rc_cpu_daemon" > "$I/results.csv"
idle_cell() {  # $1 brazo (reposo | reposo_sombra) $2 repeticion
  local arm=$1 rep=$2 D="$I/$1_$2" cpu_d="" rc=NA; mkdir -p "$D"
  taskset -c 0-5 sleep 75 & local T=$!
  if [ "$arm" = reposo_sombra ]; then
    ( exec python3 "$ROOT/fase3_daemon/cpu_loop/launch_cpu_daemon.py" --policy-table "$M/policy_activo.yaml" --binary "$CPU_BIN" --arm sombra -- \
        --model "$ROOT/fase3_daemon/cpu_loop/xgboost_cpu.onnx" --perf-cpus 0,1,2,3,4,5 --collector-cpu 6 --consumer-cpu 7 --target-pid $T \
        --interval-ns 1000000 --gpu-active-floor-khz 3200000 --log-path "$D/cpu_decisions.jsonl" ) > "$D/cpu_daemon.out" 2>&1 & cpu_d=$!
  fi
  sleep 8
  snap "$D/snap_inicio.txt"
  local e0 f0 t0 e1 f1 t1; e0=$(cat $R0/energy_uj); f0=$(cat $R1/energy_uj); t0=$(date +%s.%N)
  sleep 60
  e1=$(cat $R0/energy_uj); f1=$(cat $R1/energy_uj); t1=$(date +%s.%N)
  snap "$D/snap_fin.txt"
  [ -n "$cpu_d" ] && { kill -TERM $cpu_d 2>/dev/null; wait $cpu_d; rc=$?; }
  kill $T 2>/dev/null; wait $T 2>/dev/null
  echo "$1_$2,$arm,$rep,$(python3 -c "print(round($t1-$t0,3))"),$(python3 -c "m=$RMAX; d=lambda a,b:(b-a)%m; print(round((d($e0,$e1)+d($f0,$f1))/1e6,2))"),$rc" | tee -a "$I/results.csv"
  sleep 5
}
state "INICIO reposo (3 repeticiones alternadas)"
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then idle_cell reposo $r; idle_cell reposo_sombra $r; else idle_cell reposo_sombra $r; idle_cell reposo $r; fi
done
dump > "$OUT_BASE/cpu_state_tras_reposo.txt"
state "OK: TODAS LAS ETAPAS TERMINARON, se libera el nodo"
