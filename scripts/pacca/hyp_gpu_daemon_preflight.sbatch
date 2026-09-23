#!/bin/bash
# Preflight REAL del daemon de GPU (Bloque D, Plan_Fase3_Daemon.md): fija el reloj
# de la GPU de verdad. Cuatro etapas:
#   1. H1 directo: para cada reloj, `nvidia-smi -lgc c,c` con carga real, y se mide
#      (a) clocks.sm observado y (b) el rendimiento de una carga compute y otra memory.
#      Si el candado funciona: observado == pedido y GFLOP/s escala con el reloj.
#   2. T_transicion_gpu: costo real de cambiar de reloj con el codigo del daemon
#      (gpu_freqctl.apply_gpu_frequency) bajo carga. Alimenta --min-dwell-ns.
#   3. Caos: run_daemon.py --arm activo --mode pid contra una carga que alterna fases
#      memory/compute; SIGTERM y SIGINT con el candado puesto; el reloj debe volver al
#      nativo. Se puntua tambien la clasificacion contra las fronteras reales.
#   4. Estado final: `-rgc` incondicional.
# Toca la GPU de verdad: correr solo con el nodo libre y ninguna campana activa.
# Sin --time. No intervenir a mano sobre el job en vuelo.
#SBATCH --job-name=hyp_gpu_daemon_preflight
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_daemon_preflight_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_daemon_preflight_%j.err

set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
WORK="$ROOT/fase3_daemon/gpu_loop/preflight_work"
mkdir -p "$WORK"; rm -f "$WORK"/*.jsonl "$WORK"/*.txt "$WORK"/*.json 2>/dev/null

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate"
NVCC=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/bin/nvcc
GPU=0

hostname; git -C "$ROOT" log --oneline -1
clk() { nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits; }
cleanup() {
  pkill -P $$ gpu_phase_target 2>/dev/null || true
  pkill -P $$ -f run_daemon.py 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== ESTADO INICIAL ==="
nvidia-smi -i $GPU --query-gpu=clocks.sm,clocks.max.sm,power.draw,utilization.gpu --format=csv
sudo -n nvidia-smi -i $GPU -rgc || { echo "sudo -n nvidia-smi -rgc NO permitido" >&2; exit 1; }
"$NVCC" -O2 -arch=sm_80 -o "$WORK/gpu_phase_target" "$ROOT/fase3_daemon/gpu_loop/tools/gpu_phase_target.cu" || exit 1
T="$WORK/gpu_phase_target"

echo; echo "=== 0. pytest de run_daemon (regresion del detect_environment sin cpuset) ==="
(cd "$ROOT" && python3 -m pytest -q fase3_daemon/tests/test_run_daemon.py fase3_daemon/tests/test_gpu_loop.py 2>&1 | tail -8) || { echo "pytest de run_daemon fallo" >&2; exit 1; }

echo; echo "=== 1. H1 DIRECTO: candado, reloj observado y rendimiento real ==="
: > "$WORK/h1.txt"
for c in 1410 1260 1110 810 510 210; do
  sudo -n nvidia-smi -i $GPU -lgc $c,$c >/dev/null 2>&1 || { echo "-lgc $c fallo" >&2; }
  for m in compute memory; do
    "$T" bench_$m 4 > "$WORK/b.out" 2>&1 & bp=$!
    sleep 2.5; obs=$(clk); wait $bp
    echo "$c $m $obs $(awk '{print $2}' "$WORK/b.out")" | tee -a "$WORK/h1.txt"
  done
done
sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1
python3 - "$WORK/h1.txt" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
ok = all(abs(int(r[2]) - int(r[0])) <= 15 for r in rows)
def series(m): return {int(r[0]): float(r[3]) for r in rows if r[1] == m}
comp, mem = series("compute"), series("memory")
print("clocks.sm observado == pedido (+-15 MHz) en los 6 relojes:", "SI" if ok else "NO")
print("compute GFLOP/s / GFLOP/s@1410 (esperado ~ reloj/1410):", {c: round(comp[c] / comp[1410], 2) for c in sorted(comp, reverse=True)})
print("memory  GB/s / GB/s@1410 (esperado ~1 hasta reloj bajo):", {c: round(mem[c] / mem[1410], 2) for c in sorted(mem, reverse=True)})
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || { echo "H1 NO RESUELTO: el candado no se refleja en clocks.sm; se aborta" >&2; exit 1; }

echo; echo "=== 2. T_transicion_gpu (codigo del daemon, bajo carga; con la espera de asentamiento por defecto y con 0.3 s) ==="
"$T" bench_compute 240 > /dev/null 2>&1 & LOAD=$!
sleep 3
python3 "$ROOT/fase3_daemon/gpu_loop/measure_clock_transition.py" --clocks 1410,1260 --cycles 6 --out "$WORK/transition_default.json" 2>&1 | grep -E "restore|p50|name|unobserved" 
python3 "$ROOT/fase3_daemon/gpu_loop/measure_clock_transition.py" --clocks 1410,1260 --cycles 6 --settle-s 0.3 --out "$WORK/transition_settle03.json" 2>&1 | grep -E "restore|p50|name|unobserved"
kill $LOAD 2>/dev/null; wait $LOAD 2>/dev/null
SETTLE=0.3
DWELL_NS=$(python3 -c "import json;print(int(10*json.load(open('$WORK/transition_settle03.json'))['apply_gpu_frequency']['p50_ms']*1e6))" 2>/dev/null || echo 20000000000)
echo "min_dwell_ns = 10 x p50(apply_gpu_frequency, settle=$SETTLE) = $DWELL_NS"

echo; echo "=== 3. CAOS: run_daemon.py --arm activo (carga con huecos ociosos entre fases de 20 s) ==="
start_round() {  # $1 = nombre; deja TGT y DP definidos
  "$T" alternate 20 400 2 > "$WORK/phases_$1.txt" 2>&1 & TGT=$!
  python3 "$ROOT/fase3_daemon/run_daemon.py" --policy-table "$ROOT/fase3_daemon/policy_table.yaml" \
    --arm activo --mode pid --pid $TGT --min-dwell-ns $DWELL_NS --gpu-settle-s $SETTLE --gpu-index $GPU \
    --delegated-cpus 0 --log-path "$WORK/daemon_$1.jsonl" -v > "$WORK/daemon_$1.out" 2>&1 & DP=$!
}
wait_lock() {  # espera hasta 30 s el candado de memory (1260 +-15)
  for i in $(seq 1 30); do
    sleep 1; c=$(clk)
    if [ "$c" -le 1275 ] && [ "$c" -ge 1245 ]; then echo "candado observado a t=${i}s: clocks.sm=$c"; return 0; fi
    kill -0 $DP 2>/dev/null || { echo "el daemon murio antes de fijar el reloj" >&2; tail -20 "$WORK/daemon_$1.out"; return 1; }
  done
  echo "el daemon NO fijo el reloj en 30 s" >&2; tail -20 "$WORK/daemon_$1.out"; return 1
}
stop_round() {  # $1 = TERM|INT ; verifica que atiende la senal y que el reloj vuelve al nativo
  kill -$1 $DP
  for i in $(seq 1 40); do kill -0 $DP 2>/dev/null || break; sleep 0.5; done
  if kill -0 $DP 2>/dev/null; then echo "!!! el daemon NO atendio SIG$1 en 20 s" >&2; kill -9 $DP; kill $TGT 2>/dev/null; return 1; fi
  sleep 3; local after; after=$(clk)
  if [ "$after" -gt 1300 ]; then echo "tras SIG$1: clocks.sm=$after -> reloj RESTAURADO al nativo bajo carga"; else echo "!!! tras SIG$1 clocks.sm=$after: NO restaurado" >&2; kill $TGT 2>/dev/null; return 1; fi
  kill $TGT 2>/dev/null; wait $TGT 2>/dev/null; return 0
}
echo "--- ronda A: SIGTERM con el candado estable (tras registrar la decision) ---"
start_round A
wait_lock A && { for i in $(seq 1 20); do [ -s "$WORK/daemon_A.jsonl" ] && break; sleep 0.5; done; sleep 1; stop_round TERM; } || RC=1
sleep 3
echo "--- ronda B: SIGINT justo al fijarse el candado (posible corte a mitad de actuacion) ---"
start_round B
wait_lock B && stop_round INT || RC=1
sleep 3
echo "--- ronda C: ciclo completo memory -> compute -> memory (60 s), linea de tiempo del reloj ---"
start_round C
for t in $(seq 1 60); do
  sleep 1
  ph=$(tail -1 "$WORK/phases_C.txt" | awk '{print $2}')
  [ $((t % 3)) -eq 0 ] && echo "  t=${t}s fase_real=$ph clocks.sm=$(clk) util=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
done
stop_round TERM || RC=1
echo "decisiones registradas (ronda C):"
python3 - "$WORK/phases_C.txt" "$WORK/daemon_C.jsonl" <<'PY'
import json, sys
phases = [(int(a), b) for a, b in (l.split() for l in open(sys.argv[1]) if len(l.split()) == 2 and l.split()[1] in "MC")]
def truth(ts):
    cur = None
    for t, c in phases:
        if t <= ts: cur = c
        else: break
    return cur
n = ok = 0
for line in open(sys.argv[2]):
    r = json.loads(line)
    if r.get("label") is None: continue
    n += 1; t = truth(r["ts_ns"]); ok += (r["label"] == "memory_bound") == (t == "M")
    print(f"  verdad={t} clase={r['label']} escrito={r.get('written')} objetivo_khz={r.get('target_freq_khz')} aplicado_khz={r.get('applied_freq_khz')} inferencia_ms={(r.get('inference_time_ns') or 0)/1e6:.1f} actuacion_ms={(r.get('actuation_time_ns') or 0)/1e6:.0f}")
print(f"clasificacion: {ok}/{n} coinciden con la fase real")
PY

echo; echo "=== ESTADO FINAL ==="
sudo -n nvidia-smi -i $GPU -rgc; nvidia-smi -i $GPU --query-gpu=clocks.sm,clocks.max.sm --format=csv
[ "${RC:-0}" = 0 ] && echo "=== FIN: preflight GPU OK ===" || { echo "=== FIN: preflight GPU CON FALLOS ===" >&2; exit 1; }
