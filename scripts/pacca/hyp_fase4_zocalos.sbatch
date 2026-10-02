#!/bin/bash
# Fase 4, análisis de zócalos (aparte del libro): repite E-A por bloques, E-C y CloverLeaf bajo performance guardando la
# energía RAPL de cada paquete por separado (e_cpu0_j, e_cpu1_j) además de la suma (e_cpu_j). Pregunta: cuánto aporta el
# zócalo 1, si su aporte es igual en todos los brazos y cuánto cambian las razones al contar solo el zócalo 0.
# Etapas: estado de CPU de la base del libro (performance, turbo off, 3.2 GHz fijos en los delegados, como E-C) ->
# pre-vuelo (1 celda base de E-A) con compuerta de columnas -> E-A (5 bloques) -> E-C (6 bloques) -> CloverLeaf (5 bloques).
# Semillas nuevas, independientes de las del libro. Si una compuerta o etapa falla, el job se PAUSA sin soltar el nodo;
# liberar con: touch $OUT_BASE/HOLDER_STOP o scancel. Al terminar bien, restaura el estado de CPU y SUELTA el nodo.
# Sin --time por política operativa de pacca (cuenta compartida latorresn). Mientras mide, una shell adjunta
# (srun --jobid=<id> --overlap --pty bash) es solo para inspección: nada que use CPU o GPU.
#SBATCH --job-name=hyp_fase4_zocalos
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
export OUT_BASE="${OUT_BASE:-$HOME/hyperion-results/final/fase4_zocalos_${SLURM_JOB_ID}}"
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

# ---------- 1. estado de CPU de la base del libro (performance, turbo off, 3.2 GHz fijos; igual que E-C) ----------
ok=0
for try in 1 2 3; do
    [ "$(cat $C/cpu0/cpufreq/scaling_governor)" = performance ] || sudo -n /usr/local/bin/set_cpu_gov performance "$original_epp" >> "$OUT_BASE/campaign_status.txt" 2>&1
    sudo -n /usr/local/bin/set_turbo_state 1
    for c in $CPUS; do d=$C/cpu$c/cpufreq; echo $KHZ > $d/scaling_max_freq; echo $KHZ > $d/scaling_min_freq; done
    sleep 2
    govs=$(dump | grep -oP 'gov=\K\S+' | sort -u); mins=$(dump | grep -oP 'min=\K\S+' | sort -u); maxs=$(dump | grep -oP 'max=\K\S+' | sort -u)
    if [ "$govs" = performance ] && [ "$(cat $C/intel_pstate/no_turbo)" = 1 ] && [ "$mins" = $KHZ ] && [ "$maxs" = $KHZ ]; then ok=1; break; fi
    state "intento $try: gov=[$govs] no_turbo=$(cat $C/intel_pstate/no_turbo) min=[$mins] max=[$maxs]"
done
dump > "$OUT_BASE/cpu_state_medicion.txt"
[ "$ok" = 1 ] || hold "no se pudo fijar performance, turbo off y 3.2 GHz de forma uniforme"
state "nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT commit=$(git -C "$ROOT" log --oneline -1) estado: performance no_turbo=1 min=max=$KHZ"
sudo -n nvidia-smi -i $GPU -rgc >/dev/null || hold "sudo -n nvidia-smi -rgc NO permitido"
grep -q 'e_cpu0_j' "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" && grep -q 'e_cpu0_j' "$ROOT/scripts/pacca/hyp_cloverleaf_confirm.sbatch" \
    || hold "los scripts de $ROOT no escriben e_cpu0_j/e_cpu1_j (falta git pull)"

run_matrix() {  # $1 carpeta $2 conjunto $3 brazos $4 repeticiones $5 semilla
    env OUT="$OUT_BASE/$1" HYPERION_ROOT="$ROOT" ONLY_SCOPES=gpumem ONLY_SETS="$2" ONLY_ARMS="$3" \
        REPS_GENERAL="$4" CYCLES_GPUMEM=1 GAP_S=1 RANDOM_SEED="$5" ORDER_MODE=balanced_blocks \
        bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" > "$OUT_BASE/$1.log" 2>&1
}
check_cells() {  # $1 results.csv $2 celdas esperadas: celdas válidas y suma de zócalos igual a e_cpu_j
    python3 - "$1" "$2" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]))); n = int(sys.argv[2])
bad = [f"{r['cell']} app_rc={r['app_rc']} rc_gpu={r['rc_gpu_daemon']} state_ok={r['state_ok']}" for r in rows
       if r["app_rc"] != "0" or r["state_ok"] != "1" or r["rc_gpu_daemon"] not in ("0", "NA")]
try:
    bad += [f"{r['cell']} suma {float(r['e_cpu0_j'])+float(r['e_cpu1_j']):.2f} != {r['e_cpu_j']}" for r in rows
            if abs(float(r["e_cpu0_j"]) + float(r["e_cpu1_j"]) - float(r["e_cpu_j"])) > 0.05]
except (KeyError, ValueError) as e:
    bad.append(f"columnas por zócalo ausentes o inválidas: {e}")
if len(rows) != n: bad.append(f"{len(rows)} celdas, se esperaban {n}")
print(("FAIL " + "; ".join(bad)) if bad else f"OK {len(rows)} celdas")
PY
}

# ---------- 2. pre-vuelo: una celda base de E-A ----------
state "INICIO pre-vuelo (1 celda base de E-A)"
run_matrix preflight known base 1 20261002 || hold "el pre-vuelo terminó con error; ver preflight.log"
gate=$(check_cells "$OUT_BASE/preflight/results.csv" 1); state "pre-vuelo: $gate"
[[ "$gate" == OK* ]] || hold "compuerta del pre-vuelo no superada"

# ---------- 3. E-A por bloques (5 bloques: base, sombra, activo_gpu, fijo_gpu_f1) ----------
state "INICIO E-A (5 bloques)"
run_matrix EA known "base sombra activo_gpu fijo_gpu_f1" 5 20261002 || hold "E-A terminó con error; ver EA.log"
gate=$(check_cells "$OUT_BASE/EA/results.csv" 20); state "E-A: $gate"
[[ "$gate" == OK* ]] || state "AVISO: E-A con celdas no válidas; se continúa (F1 fijo de E-A falla el criterio de reloj en DGEMM por diseño)"

# ---------- 4. E-C (6 bloques: base, activo_gpu, fijo_gpu_f1), heartwall calibrado en el job 7815 ----------
state "INICIO E-C (6 bloques, HEARTWALL_FRAMES=10097)"
export HEARTWALL_FRAMES=10097 HEARTWALL_REPEATS=1
run_matrix EC sensitive "base activo_gpu fijo_gpu_f1" 6 20261003 || hold "E-C terminó con error; ver EC.log"
gate=$(check_cells "$OUT_BASE/EC/results.csv" 18); state "E-C: $gate"
[[ "$gate" == OK* ]] || state "AVISO: E-C con celdas no válidas; se continúa"

# ---------- 5. CloverLeaf (5 bloques: ref, sombra, activo) ----------
state "INICIO CloverLeaf (5 bloques)"
env OUT="$OUT_BASE/cloverleaf" HYPERION_ROOT="$ROOT" CLOVER_SEED=20261004 \
    bash "$ROOT/scripts/pacca/hyp_cloverleaf_confirm.sbatch" > "$OUT_BASE/cloverleaf.log" 2>&1 || hold "CloverLeaf terminó con error; ver cloverleaf.log"
state "CloverLeaf: $(($(wc -l < "$OUT_BASE/cloverleaf/results.csv")-1)) celdas"

dump > "$OUT_BASE/cpu_state_final.txt"
diff -q "$OUT_BASE/cpu_state_medicion.txt" "$OUT_BASE/cpu_state_final.txt" >/dev/null || state "AVISO: el estado de CPU cambió durante la campaña"
state "TODAS LAS ETAPAS TERMINARON, se libera el nodo"
