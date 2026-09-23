#!/bin/bash
# Prueba de COEXISTENCIA de los dos daemons activos a la vez (Fase 3): la compuesta A de GPU (cutlass_dgemm + stream_triad,
# 3 ciclos) corre con el daemon de GPU (gpu_loop_main) y el de CPU (cpu_loop_main) en brazo activo simultaneamente, coordinados
# por el archivo de senal gpu_active. Verifica: ambos arrancan y corren toda la aplicacion sin bloquearse, la senal sube durante
# la actividad de GPU, la CPU respeta el piso (FLOOR_KHZ) mientras la senal esta activa, ambos salen con codigo 0 al recibir
# SIGTERM y el estado de frecuencia/turbo de la CPU queda identico al inicial. NO es un experimento de EDP (los experimentos
# de CPU y de GPU son separados, decision del usuario). Toca frecuencias de CPU, turbo y reloj de GPU: nodo libre. Sin --time.
#SBATCH --job-name=hyp_daemons_coex
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/daemons_coex_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/daemons_coex_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"
FLOOR_KHZ="${FLOOR_KHZ:-0}"
WORK="$ROOT/fase3_daemon/coexistence_work"; mkdir -p "$WORK"; rm -f "$WORK"/*.jsonl "$WORK"/*.out "$WORK"/*.txt "$WORK"/sig 2>/dev/null
ORT="$HOME/.conda/envs/hyperion-cpu-onnx"; CMAKE="$ORT/bin/cmake"; CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
C=/sys/devices/system/cpu; CPUS="0 1 2 3 4 5 16 17 18 19 20 21"; GPU=0
hostname; git -C "$ROOT" log --oneline -1; echo "piso de CPU con GPU activa: ${FLOOR_KHZ} kHz"

echo "=== compilando cpu_loop_main y gpu_loop_main ==="
"$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$WORK/cpu_build" -DONNXRUNTIME_ROOT="$ORT" >/dev/null && "$CMAKE" --build "$WORK/cpu_build" -j 8 --target cpu_loop_main 2>&1 | grep -E "error|Built target cpu_loop_main"
"$CMAKE" -S "$ROOT/fase3_daemon/gpu_loop_cpp" -B "$WORK/gpu_build" -DONNXRUNTIME_ROOT="$ORT" -DNVML_INCLUDE_DIR="$CUDA/include" \
  -DNVML_LIB="$CUDA/targets/x86_64-linux/lib/stubs/libnvidia-ml.so" >/dev/null && "$CMAKE" --build "$WORK/gpu_build" -j 8 --target gpu_loop_main 2>&1 | grep -E "error|Built target gpu_loop_main"
CPU_BIN="$WORK/cpu_build/cpu_loop_main"; GPU_BIN="$WORK/gpu_build/gpu_loop_main"
[ -x "$CPU_BIN" ] && [ -x "$GPU_BIN" ] || { echo "no se construyeron ambos binarios" >&2; exit 1; }

state() { echo "no_turbo=$(cat $C/intel_pstate/no_turbo)"; for c in $CPUS; do d=$C/cpu$c/cpufreq; echo "cpu$c min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq)"; done; }
state > "$WORK/state_inicial.txt"; ORIG_NO_TURBO=$(head -1 "$WORK/state_inicial.txt" | cut -d= -f2)
recover() {
  pkill -P $$ -f "cpu_loop_main|gpu_loop_main|composite_gpu_known" 2>/dev/null || true
  state > "$WORK/state_ahora.txt"
  if ! diff -q "$WORK/state_inicial.txt" "$WORK/state_ahora.txt" >/dev/null 2>&1; then
    echo "!!! estado de CPU distinto del inicial al salir; restaurando desde el snapshot" >&2
    while read -r line; do case "$line" in cpu*) c=${line%% *}; c=${c#cpu}; mn=$(echo "$line" | sed -n 's/.*min=\([0-9]*\).*/\1/p'); mx=$(echo "$line" | sed -n 's/.*max=\([0-9]*\).*/\1/p'); d=$C/cpu$c/cpufreq; echo "$mx" > $d/scaling_max_freq; echo "$mn" > $d/scaling_min_freq;; esac; done < "$WORK/state_inicial.txt"
    sudo -n /usr/local/bin/set_turbo_state "$ORIG_NO_TURBO" || true
  fi
  sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true
}
trap recover EXIT
sudo -n nvidia-smi -i $GPU -rgc || { echo "sudo -n nvidia-smi -rgc NO permitido" >&2; exit 1; }

# politica: GPU tal como esta versionada (memory 1260, compute no_actuar); CPU experimental F0 base / F1 en memory
python3 - "$ROOT/fase3_daemon/policy_table.yaml" "$WORK/policy_coex.yaml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
d["policy"]["cpu-compute_bound"] = {"action": "actuar_experimental", "chosen_level": "F0", "resolved_freq_khz": 3200000}
d["policy"]["cpu-memory_bound"] = {"action": "actuar_experimental", "chosen_level": "F1", "resolved_freq_khz": 2900000}
yaml.safe_dump(d, open(sys.argv[2], "w"))
PY
SIG="$WORK/sig"

echo "=== aplicacion compuesta A de GPU + los dos daemons activos ==="
( cd "$KERNEL_ROOT" && exec taskset -c 0-3 python3 "$ROOT/fase3_daemon/composite_apps/composite_gpu_known.py" --node-id pacca-a100 \
    --kernels-root "$KERNEL_ROOT" --boundaries-out "$WORK/phases.jsonl" > "$WORK/composite.out" 2>&1 ) & COMP=$!
( exec python3 "$ROOT/fase3_daemon/gpu_loop_cpp/launch_gpu_daemon.py" --policy-table "$WORK/policy_coex.yaml" --binary "$GPU_BIN" --arm activo -- \
    --model "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.onnx" --features "$ROOT/fase3_daemon/gpu_loop_cpp/gpu_rf_sin_reloj.features.txt" \
    --gpu-index $GPU --min-dwell-ns 3700000000 --settle-s 0.3 --target-pid $COMP --gpu-active-signal-path "$SIG" --log-path "$WORK/gpu_decisions.jsonl" ) > "$WORK/gpu_daemon.out" 2>&1 & GD=$!
( exec python3 "$ROOT/fase3_daemon/cpu_loop/launch_cpu_daemon.py" --policy-table "$WORK/policy_coex.yaml" --binary "$CPU_BIN" --arm activo -- \
    --model "$ROOT/fase3_daemon/cpu_loop/xgboost_cpu.onnx" --perf-cpus 0,1,2,3 --collector-cpu 6 --consumer-cpu 7 --target-pid $COMP \
    --gpu-active-signal-path "$SIG" --gpu-active-floor-khz $FLOOR_KHZ --log-path "$WORK/cpu_decisions.jsonl" ) > "$WORK/cpu_daemon.out" 2>&1 & CD=$!
sleep 3
kill -0 $GD 2>/dev/null && echo "daemon GPU vivo (pid $GD)" || { echo "el daemon de GPU murio al arrancar" >&2; tail -5 "$WORK/gpu_daemon.out"; }
kill -0 $CD 2>/dev/null && echo "daemon CPU vivo (pid $CD)" || { echo "el daemon de CPU murio al arrancar" >&2; tail -5 "$WORK/cpu_daemon.out"; }

# linea de tiempo cada 200 ms mientras corre la aplicacion
: > "$WORK/timeline.txt"
while kill -0 $COMP 2>/dev/null; do
  echo "$(date +%s.%N) sig=$(cat $SIG 2>/dev/null || echo -) cpu0_max=$(cat $C/cpu0/cpufreq/scaling_max_freq) cpu16_max=$(cat $C/cpu16/cpufreq/scaling_max_freq) no_turbo=$(cat $C/intel_pstate/no_turbo) gpu_clk=$(nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits)" >> "$WORK/timeline.txt"
  sleep 0.2
done
wait $COMP; echo "compuesta termino (codigo $?)"; cut -c1-140 "$WORK/composite.out"
sleep 2
echo "gpu daemon vivo tras terminar el target? $(kill -0 $GD 2>/dev/null && echo si || echo no)"
kill -TERM $CD 2>/dev/null; kill -TERM $GD 2>/dev/null
for i in $(seq 1 40); do kill -0 $CD 2>/dev/null || kill -0 $GD 2>/dev/null || break; sleep 0.5; done
wait $CD; RC_CPU=$?; wait $GD; RC_GPU=$?
echo "codigos de salida: cpu_loop_main=$RC_CPU gpu_loop_main=$RC_GPU"
grep -E "^(actuador|resumen|deteniendo)" "$WORK/cpu_daemon.out" "$WORK/gpu_daemon.out" | cut -c1-200
state > "$WORK/state_final.txt"
diff "$WORK/state_inicial.txt" "$WORK/state_final.txt" && echo "estado de CPU (turbo y rango) IDENTICO al inicial"
echo "reloj de GPU tras los daemons (reposo): $(nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits) MHz"

python3 - "$WORK/timeline.txt" "$FLOOR_KHZ" "$WORK/cpu_decisions.jsonl" "$WORK/gpu_decisions.jsonl" <<'PY'
import json, re, sys
rows = []
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) sig=(\S+) cpu0_max=(\d+) cpu16_max=(\d+) no_turbo=(\d) gpu_clk=(\d+)", l)
    if m: rows.append((float(m[1]), m[2], int(m[3]), int(m[4]), int(m[5]), int(m[6])))
floor = int(sys.argv[2])
on = [r for r in rows if r[1] == "1"]; off = [r for r in rows if r[1] == "0"]
print(f"\nmuestras: {len(rows)} (senal=1: {len(on)}, senal=0: {len(off)}, sin senal: {len(rows)-len(on)-len(off)})")
if on: print(f"con la GPU activa (senal=1): max de CPU minimo={min(r[2] for r in on)} kHz, maximo={max(r[2] for r in on)} kHz; reloj GPU minimo={min(r[5] for r in on)} MHz")
if off: print(f"con la GPU inactiva (senal=0): max de CPU minimo={min(r[2] for r in off)} kHz, maximo={max(r[2] for r in off)} kHz")
if on and floor: print(f"piso {floor} kHz respetado con la senal activa: {'SI' if min(r[2] for r in on) >= floor else 'NO'}")
print("turbo apagado durante toda la corrida:", all(r[4] == 1 for r in rows))
for name, path in (("CPU", sys.argv[3]), ("GPU", sys.argv[4])):
    try: recs = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError: recs = []
    lab = {}
    for r in recs: lab[str(r.get("label"))] = lab.get(str(r.get("label")), 0) + 1
    print(f"decisiones {name}: {len(recs)}  por etiqueta: {lab}  escrituras reales: {sum(1 for r in recs if r.get('written'))}  fallos de escritura: {sum(1 for r in recs if r.get('write_failed'))}")
PY
echo "=== FIN ==="
