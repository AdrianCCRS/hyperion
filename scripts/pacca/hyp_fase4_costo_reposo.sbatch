#!/bin/bash
# Fase 4, diagnostico del costo de observacion del agente de CPU en reposo (revision C6; docs/planeacion/Plan_Fase4_Costo_CPU.md,
# segunda parte). Con el nodo sin aplicacion el agente solo elevo la potencia de CPU de 75.6 a 146.3 W (job 7824), asi que el
# reposo es el lugar donde se puede descomponer el costo. Cada brazo corre 60 s con un proceso `sleep` en los nucleos 0-5 como
# objetivo, y se mide la energia RAPL de ambos paquetes, la residencia en estados de reposo, las interrupciones entre nucleos
# y el tiempo de CPU de cada hilo del agente. Brazos:
#   reposo            sin agente
#   sombra_1ms        agente completo (colector 1 ms, consumidor 100 us)
#   sombra_10ms       colector a 10 ms
#   sombra_100ms      colector a 100 ms
#   sombra_noperf     agente completo pero el colector no lee contadores (solo despierta cada 1 ms)
#   despertador_1ms   sin agente: un proceso minimo en el nucleo 6 que duerme 1 ms en bucle
# 3 bloques aleatorizados (semilla RANDOM_SEED_REPOSO, 20261004). Funciona en paccaA100 (estado de CPU de la base del libro:
# performance, turbo off, 3.2 GHz, si hay permisos) y en nodos de la particion normal (sin permisos de frecuencia: se
# registra el estado tal como esta). Se envia con -p/-w/--gres por linea de comandos; el nombre del job debe empezar por hyp_.
# Sin --time por politica operativa de pacca (cuenta compartida latorresn). Si algo falla en paccaA100 el job se PAUSA (no suelta
# el nodo): liberar con touch $OUT_BASE/HOLDER_STOP o scancel. En otros nodos termina con error.
#SBATCH --job-name=hyp_fase4_costo_reposo
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err
set -uo pipefail

ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; NODE=$(hostname -s)
SEED="${RANDOM_SEED_REPOSO:-20261004}"; REPS="${REPS_REPOSO:-3}"; SECS="${SECS_REPOSO:-60}"
export OUT_BASE="${OUT_BASE:-$HOME/hyperion-results/final/fase4_costo_reposo_${NODE}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT_BASE/cells"
ORT="$HOME/.conda/envs/hyperion-cpu-onnx"; CMAKE="$ORT/bin/cmake"
module load cmake/4.3.4 gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
C=/sys/devices/system/cpu; CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
R0=/sys/class/powercap/intel-rapl:0; R1=/sys/class/powercap/intel-rapl:1; RMAX=$(cat $R0/max_energy_range_uj)
STOP="$OUT_BASE/HOLDER_STOP"; rm -f "$STOP"
state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT_BASE/campaign_status.txt"; }
dump() { echo "no_turbo=$(cat $C/intel_pstate/no_turbo)"; for c in $CPUS; do d=$C/cpu$c/cpufreq
    echo "cpu$c gov=$(cat $d/scaling_governor 2>/dev/null) epp=$(cat $d/energy_performance_preference 2>/dev/null) min=$(cat $d/scaling_min_freq 2>/dev/null) max=$(cat $d/scaling_max_freq 2>/dev/null)"; done; }
fail() {  # en paccaA100 pausa sin soltar el nodo; en otros nodos termina
    state "FALLO: $1"
    if [ "$NODE" = paccaA100 ]; then
        state "PAUSA: adjuntar srun --jobid=$SLURM_JOB_ID --overlap --pty bash | liberar: touch $STOP o scancel $SLURM_JOB_ID"
        while [[ ! -f "$STOP" ]]; do sleep 60; done
    fi
    exit 2
}

lscpu | grep -E "Model name|Socket|Thread" | tee -a "$OUT_BASE/campaign_status.txt"
dump > "$OUT_BASE/cpu_state_original.txt"
original_governor=$(cat $C/cpu0/cpufreq/scaling_governor); original_epp=$(cat $C/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo "")
original_no_turbo=$(cat $C/intel_pstate/no_turbo)
CAN_SET=0; [ -x /usr/local/bin/set_cpu_gov ] && [ -x /usr/local/bin/set_turbo_state ] && sudo -n /usr/local/bin/set_turbo_state "$original_no_turbo" >/dev/null 2>&1 && CAN_SET=1
restore() {
    [ "$CAN_SET" = 1 ] || return 0
    sudo -n /usr/local/bin/set_cpu_gov "$original_governor" "$original_epp" >/dev/null 2>&1 || true
    sudo -n /usr/local/bin/set_turbo_state "$original_no_turbo"
    while read -r line; do case "$line" in cpu*)
        c=${line%% *}; c=${c#cpu}; mn=$(echo "$line" | sed -n 's/.*min=\([0-9]*\).*/\1/p'); mx=$(echo "$line" | sed -n 's/.*max=\([0-9]*\).*/\1/p')
        d=$C/cpu$c/cpufreq; echo "$mx" > $d/scaling_max_freq; echo "$mn" > $d/scaling_min_freq; echo "$mx" > $d/scaling_max_freq;; esac
    done < "$OUT_BASE/cpu_state_original.txt"
    dump > "$OUT_BASE/cpu_state_restaurado.txt"
    diff -q "$OUT_BASE/cpu_state_original.txt" "$OUT_BASE/cpu_state_restaurado.txt" >/dev/null && state "estado de CPU restaurado al original" \
        || state "AVISO: el estado restaurado difiere del original"
}
trap restore EXIT

# ---------- 1. estado de CPU: el de la base del libro si hay permisos; si no, el que tenga el nodo ----------
if [ "$CAN_SET" = 1 ]; then
    ok=0
    for try in 1 2 3; do
        [ "$(cat $C/cpu0/cpufreq/scaling_governor)" = performance ] || sudo -n /usr/local/bin/set_cpu_gov performance "$original_epp" >/dev/null 2>&1
        sudo -n /usr/local/bin/set_turbo_state 1
        for c in $CPUS; do d=$C/cpu$c/cpufreq; echo 3200000 > $d/scaling_max_freq; echo 3200000 > $d/scaling_min_freq; done
        sleep 2
        govs=$(dump | grep -oP 'gov=\K\S+' | sort -u); mins=$(dump | grep -oP 'min=\K\S+' | sort -u); maxs=$(dump | grep -oP 'max=\K\S+' | sort -u)
        [ "$govs" = performance ] && [ "$(cat $C/intel_pstate/no_turbo)" = 1 ] && [ "$mins" = 3200000 ] && [ "$maxs" = 3200000 ] && { ok=1; break; }
    done
    [ "$ok" = 1 ] || fail "no se pudo fijar performance, turbo off y 3.2 GHz de forma uniforme"
    state "estado de la base del libro verificado (performance, turbo off, 3.2 GHz)"
else
    state "SIN permisos de frecuencia en $NODE: se mide con el estado del nodo (gov=$original_governor no_turbo=$original_no_turbo)"
fi
dump > "$OUT_BASE/cpu_state_medicion.txt"
state "nodo=$NODE job=${SLURM_JOB_ID:-manual} commit=$(git -C "$ROOT" log --oneline -1) semilla=$SEED reps=$REPS segundos=$SECS"

# ---------- 2. binario del agente de CPU, construido en este nodo ----------
BUILD="$OUT_BASE/build"; mkdir -p "$BUILD"
"$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD/cpu_build" -DONNXRUNTIME_ROOT="$ORT" >"$BUILD/cmake.log" 2>&1 \
  && "$CMAKE" --build "$BUILD/cpu_build" -j 8 --target cpu_loop_main >>"$BUILD/cmake.log" 2>&1
CPU_BIN="$BUILD/cpu_build/cpu_loop_main"
[ -x "$CPU_BIN" ] || fail "no se construyo cpu_loop_main en $NODE (ver $BUILD/cmake.log)"
"$CPU_BIN" --help 2>&1 | grep -q -- "--no-perf" || fail "el binario no tiene --no-perf: falta actualizar el repositorio en pacca"

source <(sed -n '/^snap() {/,/^}/p' "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch")
POLICY="$ROOT/fase3_daemon/policy_table.yaml"; MODEL="$ROOT/fase3_daemon/cpu_loop/xgboost_cpu.onnx"
agent() {  # $1 directorio $2 pid objetivo ; resto: flags extra de cpu_loop_main; deja el PID en $LAUNCHED
    local D=$1 T=$2; shift 2
    ( exec python3 "$ROOT/fase3_daemon/cpu_loop/launch_cpu_daemon.py" --policy-table "$POLICY" --binary "$CPU_BIN" --arm sombra -- \
        --model "$MODEL" --perf-cpus 0,1,2,3,4,5 --collector-cpu 6 --consumer-cpu 7 --target-pid "$T" \
        --gpu-active-floor-khz 3200000 --log-path "$D/cpu_decisions.jsonl" "$@" ) > "$D/cpu_daemon.out" 2>&1 &
    LAUNCHED=$!
}
waker() { ( exec taskset -c 6 python3 -c 'import time
while True: time.sleep(0.001)' ) > /dev/null 2>&1 & LAUNCHED=$!; }

# ---------- 3. brazos y comprobacion previa (10 s por brazo con agente; descarta los que no arranquen) ----------
ARMS_ALL="reposo sombra_1ms sombra_10ms sombra_100ms sombra_noperf despertador_1ms"
ARMS=""
for arm in $ARMS_ALL; do
    case "$arm" in reposo|despertador_1ms) ARMS="$ARMS $arm"; continue;; esac
    D="$OUT_BASE/smoke_$arm"; mkdir -p "$D"; taskset -c 0-5 sleep 20 & T=$!
    case "$arm" in sombra_1ms) F="--interval-ns 1000000";; sombra_10ms) F="--interval-ns 10000000";;
        sombra_100ms) F="--interval-ns 100000000";; sombra_noperf) F="--interval-ns 1000000 --no-perf";; esac
    agent "$D" $T $F; a=$LAUNCHED; sleep 6
    if kill -0 $a 2>/dev/null; then kill -TERM $a; wait $a; rc=$?; else rc=dead; fi
    kill $T 2>/dev/null; wait $T 2>/dev/null
    if [ "$rc" = 0 ]; then ARMS="$ARMS $arm"; state "comprobacion $arm: OK"; else state "comprobacion $arm: FALLA (rc=$rc), se omite; ver $D/cpu_daemon.out"; fi
done
state "brazos a medir:$ARMS"
case "$ARMS" in *sombra_1ms*) ;; *) fail "el agente completo no arranca en $NODE: sin el brazo de referencia no hay contraste";; esac

# ---------- 4. mediciones: bloques aleatorizados, 60 s por celda ----------
python3 - "$SEED" "$REPS" $ARMS > "$OUT_BASE/cells.txt" <<'PY'
import random, sys
rng = random.Random(int(sys.argv[1])); reps = int(sys.argv[2]); arms = sys.argv[3:]
for r in range(1, reps + 1):
    b = list(arms); rng.shuffle(b)
    for a in b: print(a, r)
PY
echo "cell,arm,rep,wall_s,e_cpu_j,rc" > "$OUT_BASE/results.csv"
while read -r arm rep; do
    id="${arm}_$rep"; D="$OUT_BASE/cells/$id"; mkdir -p "$D"
    taskset -c 0-5 sleep $((SECS + 20)) & T=$!
    pid=""; rc=NA
    case "$arm" in
        sombra_1ms) agent "$D" $T --interval-ns 1000000; pid=$LAUNCHED;;
        sombra_10ms) agent "$D" $T --interval-ns 10000000; pid=$LAUNCHED;;
        sombra_100ms) agent "$D" $T --interval-ns 100000000; pid=$LAUNCHED;;
        sombra_noperf) agent "$D" $T --interval-ns 1000000 --no-perf; pid=$LAUNCHED;;
        despertador_1ms) waker; pid=$LAUNCHED;;
    esac
    sleep 8; snap "$D/snap_inicio.txt"
    e0=$(cat $R0/energy_uj); f0=$(cat $R1/energy_uj); t0=$(date +%s.%N)
    sleep "$SECS"
    e1=$(cat $R0/energy_uj); f1=$(cat $R1/energy_uj); t1=$(date +%s.%N)
    snap "$D/snap_fin.txt"
    if [ -n "$pid" ]; then
        if [ "$arm" = despertador_1ms ]; then kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null; rc=0
        else kill -TERM $pid 2>/dev/null; wait $pid; rc=$?; fi
    fi
    kill $T 2>/dev/null; wait $T 2>/dev/null
    echo "$id,$arm,$rep,$(python3 -c "print(round($t1-$t0,3))"),$(python3 -c "m=$RMAX; d=lambda a,b:(b-a)%m; print(round((d($e0,$e1)+d($f0,$f1))/1e6,2))"),$rc" | tee -a "$OUT_BASE/results.csv"
    sleep 6
done < "$OUT_BASE/cells.txt"
dump > "$OUT_BASE/cpu_state_tras.txt"
diff -q "$OUT_BASE/cpu_state_medicion.txt" "$OUT_BASE/cpu_state_tras.txt" >/dev/null || state "AVISO: el estado de CPU cambio durante las mediciones"
state "OK: $(($(wc -l < "$OUT_BASE/results.csv")-1)) celdas; TODAS LAS ETAPAS TERMINARON, se libera el nodo"
