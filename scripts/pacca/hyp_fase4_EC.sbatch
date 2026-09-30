#!/bin/bash
# Fase 4, escenario E-C (docs/planeacion/Plan_Fase4_Escenario_EC.md): memoria + heartwall (cómputo sensible al reloj) + memoria,
# para distinguir la conmutación por fase del agente de un F1 fijo (revisión C1 del director).
# Etapas: estado de CPU de la base del libro (performance, turbo off, 3.2 GHz fijos en los delegados) -> sondeo de heartwall
# con compuerta automática -> pre-vuelo (1 celda por brazo) con compuerta técnica -> confirmatorio de 6 bloques
# (base, activo_gpu, fijo_gpu_f1). Si una compuerta falla, el job se PAUSA sin medir (no suelta el nodo) para diagnosticar;
# liberar con: touch $OUT_BASE/HOLDER_STOP o scancel. Al terminar bien, restaura el estado de CPU y SUELTA el nodo.
# Sin --time por política operativa de pacca (cuenta compartida latorresn). Mientras mide, una shell adjunta
# (srun --jobid=<id> --overlap --pty bash) es solo para inspección: nada que use CPU o GPU.
#SBATCH --job-name=hyp_fase4_EC
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
export OUT_BASE="${OUT_BASE:-$HOME/hyperion-results/final/fase4_EC_${SLURM_JOB_ID}}"
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

# ---------- 1. estado de CPU de la base del libro (job 7696) ----------
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
state "nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT commit=$(git -C "$ROOT" log --oneline -1) estado: performance, no_turbo=1, min=max=$KHZ"
sudo -n nvidia-smi -i $GPU -rgc >/dev/null || hold "sudo -n nvidia-smi -rgc NO permitido"

# ---------- 2. sondeo de heartwall con compuerta automática ----------
P="$OUT_BASE/sondeo"; mkdir -p "$P"; HW="$KERNEL_ROOT/bin/rodinia_heartwall"
[ -x "$HW" ] || hold "no existe $HW"
probe() {  # $1 etiqueta $2 cuadros $3 nativo|f1
    local tag=$1 frames=$2 clock=$3
    sudo -n nvidia-smi -i $GPU -rgc >/dev/null
    [ "$clock" = f1 ] && sudo -n nvidia-smi -i $GPU -lgc 1260,1260 >/dev/null
    sleep 3
    nvidia-smi -i $GPU --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits -lms 250 > "$P/$tag.clock.csv" 2>&1 & local smp=$!
    local t0 t1 rc; t0=$(date +%s.%N)
    ( cd "$KERNEL_ROOT" && exec taskset -c 0-5 "$HW" data/heartwall/test.avi "$frames" ) > "$P/$tag.out" 2>&1; rc=$?
    t1=$(date +%s.%N); kill $smp 2>/dev/null; wait $smp 2>/dev/null
    sudo -n nvidia-smi -i $GPU -rgc >/dev/null
    echo "$tag,$frames,$clock,$(python3 -c "print(round($t1-$t0,3))"),$rc" | tee -a "$P/sondeo.csv"
    sleep 2
}
echo "tag,frames,clock,wall_s,rc" > "$P/sondeo.csv"
probe nat2000 2000 nativo; probe nat6000a 6000 nativo; probe f1_6000a 6000 f1; probe nat6000b 6000 nativo; probe f1_6000b 6000 f1
decision=$(python3 - "$P" <<'PY'
import csv, math, statistics, sys
from pathlib import Path
P = Path(sys.argv[1]); rows = {r["tag"]: r for r in csv.DictReader(open(P / "sondeo.csv"))}
fails = [f"{t} rc={r['rc']}" for t, r in rows.items() if r["rc"] != "0"]
w = {t: float(r["wall_s"]) for t, r in rows.items()}
def samples(tag):
    out = []
    for line in open(P / f"{tag}.clock.csv"):
        try: u, c = (int(x) for x in line.split(",")); out.append((u, c))
        except ValueError: pass
    return out
nat = (w["nat6000a"] + w["nat6000b"]) / 2; f1 = (w["f1_6000a"] + w["f1_6000b"]) / 2; ratio = f1 / nat
nat_clk = [c for t in ("nat6000a", "nat6000b") for u, c in samples(t) if u >= 10]
f1_bad = [c for t in ("f1_6000a", "f1_6000b") for u, c in samples(t) if u >= 10 and not 1230 <= c <= 1290]
cover = []
for t in ("nat6000a", "nat6000b", "f1_6000a", "f1_6000b"):
    s = samples(t); idx = [i for i, (u, _) in enumerate(s) if u > 5]
    cover.append(sum(1 for u, _ in s[idx[0]:idx[-1] + 1] if u > 5) / (idx[-1] - idx[0] + 1) if idx else 0.0)
med_clk = statistics.median(nat_clk) if nat_clk else 0
if ratio < 1.05: fails.append(f"sensibilidad F1/nativo={ratio:.3f} < 1.05")
if med_clk < 1350: fails.append(f"reloj nativo mediano {med_clk} < 1350 (limitado por potencia)")
if f1_bad: fails.append(f"F1 fuera de 1230-1290 en {len(f1_bad)} muestras")
if min(cover) < 0.90: fails.append(f"actividad intermitente: cobertura minima {min(cover):.2f} < 0.90")
per_frame = (nat - w["nat2000"]) / 4000; fixed = max(w["nat2000"] - 2000 * per_frame, 0.0)
if per_frame <= 0:
    print(f"FAIL tiempo por cuadro no positivo (nat2000={w['nat2000']:.2f}s, nat6000={nat:.2f}s)"); print("0 0"); sys.exit(0)
target = 30.0; repeats = 1
while fixed + per_frame * 20000 < target / repeats and repeats < 4: repeats += 1
frames = min(20000, max(1000, math.ceil((target / repeats - fixed) / per_frame)))
summary = (f"nativo6000={nat:.2f}s F1_6000={f1:.2f}s sensibilidad={ratio:.3f} reloj_nativo_mediano={med_clk} "
           f"cobertura_min={min(cover):.2f} fijo={fixed:.2f}s por_cuadro={per_frame*1e3:.3f}ms -> frames={frames} repeats={repeats} "
           f"(heartwall estimado {repeats*(fixed+per_frame*frames):.1f}s)")
print(("FAIL " + "; ".join(fails) + " | " if fails else "OK ") + summary)
print(frames, repeats)
PY
)
state "sondeo: $(echo "$decision" | head -1)"
case "$(echo "$decision" | head -1)" in OK*) ;; *) hold "compuerta del sondeo no superada";; esac
read -r HEARTWALL_FRAMES HEARTWALL_REPEATS <<< "$(echo "$decision" | tail -1)"
export HEARTWALL_FRAMES HEARTWALL_REPEATS

# ---------- 3. pre-vuelo y 4. confirmatorio ----------
run_matrix() {  # $1 carpeta $2 repeticiones $3 semilla
    env OUT="$OUT_BASE/$1" HYPERION_ROOT="$ROOT" ONLY_SCOPES=gpumem ONLY_SETS=sensitive ONLY_ARMS="base activo_gpu fijo_gpu_f1" \
        REPS_GENERAL="$2" CYCLES_GPUMEM=1 GAP_S=1 RANDOM_SEED="$3" ORDER_MODE=balanced_blocks \
        bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" > "$OUT_BASE/$1.log" 2>&1
}
state "INICIO preflight (HEARTWALL_FRAMES=$HEARTWALL_FRAMES HEARTWALL_REPEATS=$HEARTWALL_REPEATS)"
run_matrix preflight 1 20260930 || hold "el pre-vuelo terminó con error; ver preflight.log"
gate=$(python3 - "$OUT_BASE/preflight/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [f"{r['cell']} app_rc={r['app_rc']} rc_gpu={r['rc_gpu_daemon']} state_ok={r['state_ok']}" for r in rows
       if r["app_rc"] != "0" or r["state_ok"] != "1" or r["rc_gpu_daemon"] not in ("0", "NA")]
print(("FAIL " + "; ".join(bad)) if bad or len(rows) != 3 else f"OK {len(rows)} celdas")
PY
)
state "pre-vuelo: $gate"
[[ "$gate" == OK* ]] || hold "compuerta del pre-vuelo no superada"
dump > "$OUT_BASE/cpu_state_tras_preflight.txt"
diff -q "$OUT_BASE/cpu_state_medicion.txt" "$OUT_BASE/cpu_state_tras_preflight.txt" >/dev/null || hold "el estado de CPU cambió durante el pre-vuelo"

state "INICIO EC_confirmatorio (6 bloques)"
run_matrix EC_confirmatorio 6 20261001 || hold "el confirmatorio terminó con error; ver EC_confirmatorio.log"
dump > "$OUT_BASE/cpu_state_tras_EC.txt"
diff -q "$OUT_BASE/cpu_state_medicion.txt" "$OUT_BASE/cpu_state_tras_EC.txt" >/dev/null || state "AVISO: el estado de CPU cambió durante el confirmatorio"
state "OK EC_confirmatorio: $(($(wc -l < "$OUT_BASE/EC_confirmatorio/results.csv")-1)) celdas; TODAS LAS ETAPAS TERMINARON, se libera el nodo"
