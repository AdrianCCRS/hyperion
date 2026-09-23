#!/bin/bash
# Preflight REAL del actuador de frecuencia de CPU (Bloque C8): escribe
# frecuencia y turbo de verdad en paccaA100. Tres cosas:
#   1. Estado ANTES: cpufreq de los nucleos 0-5 y sus hermanos (16-21),
#      no_turbo y relojes de GPU -- confirma que la campana cancelada a mitad
#      de corrida (job 7590) no dejo el nodo contaminado.
#   2. Sonda real: latencia de enter/set_khz/restore (turbo incluido) y tiempo
#      de asentamiento de scaling_cur_freq bajo carga en los nucleos 0-5.
#   3. Caos: cpu_loop_main --arm activo real, cortado con SIGTERM y con SIGINT;
#      el estado tras cada corte debe ser IDENTICO al inicial. (SIGKILL no se
#      prueba: no es atrapable, es un limite declarado del actuador.)
# Toca el nodo de verdad: correr solo con el nodo libre y NINGUNA medicion
# activa (ver AGENTS.md). Sin --time. Un trap final restaura el estado
# original desde el snapshot de este script si algo quedo distinto.
#SBATCH --job-name=hyp_cpu_actuator_preflight
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_actuator_preflight_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_actuator_preflight_%j.err

set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
ONNXRUNTIME_ROOT="$HOME/.conda/envs/hyperion-cpu-onnx"
CMAKE="$ONNXRUNTIME_ROOT/bin/cmake"
BUILD_DIR="$ROOT/fase3_daemon/cpu_loop/build_preflight"
CPUFREQ=/sys/devices/system/cpu
CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
WORK="$BUILD_DIR/preflight_work"

hostname; git -C "$ROOT" log --oneline -1

state() {  # una linea por atributo, sin cur_freq (es volatil)
  echo "no_turbo=$(cat $CPUFREQ/intel_pstate/no_turbo)"
  for c in $CPUS; do
    d=$CPUFREQ/cpu$c/cpufreq
    echo "cpu$c min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq) gov=$(cat $d/scaling_governor)"
  done
}

mkdir -p "$WORK"
state > "$WORK/state_inicial.txt"
echo "=== ESTADO INICIAL (tras la cancelacion del job 7590) ==="
cat "$WORK/state_inicial.txt"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.draw --format=csv || true

ORIG_NO_TURBO=$(grep '^no_turbo=' "$WORK/state_inicial.txt" | cut -d= -f2)
recover() {  # red de seguridad: si algo quedo distinto del inicial, se deja como estaba
  state > "$WORK/state_ahora.txt" || true
  if ! diff -q "$WORK/state_inicial.txt" "$WORK/state_ahora.txt" >/dev/null 2>&1; then
    echo "!!! estado distinto del inicial al salir; restaurando desde el snapshot del script" >&2
    while read -r line; do
      case "$line" in
        cpu*) c=${line%% *}; c=${c#cpu}; mn=$(echo "$line" | sed -n 's/.*min=\([0-9]*\).*/\1/p'); mx=$(echo "$line" | sed -n 's/.*max=\([0-9]*\).*/\1/p')
              d=$CPUFREQ/cpu$c/cpufreq; echo "$mn" > $d/scaling_min_freq || true; echo "$mx" > $d/scaling_max_freq || true; echo "$mn" > $d/scaling_min_freq || true ;;
      esac
    done < "$WORK/state_inicial.txt"
    sudo -n /usr/local/bin/set_turbo_state "$ORIG_NO_TURBO" || true
  fi
  pkill -P $$ yes 2>/dev/null || true
}
trap recover EXIT

sudo -n -l /usr/local/bin/set_turbo_state >/dev/null 2>&1 && echo "sudo -n set_turbo_state: permitido" || { echo "sudo -n NO permite set_turbo_state" >&2; exit 1; }

rm -rf "$BUILD_DIR/cmake_out"
"$CMAKE" -S "$ROOT/fase3_daemon/cpu_loop" -B "$BUILD_DIR/cmake_out" -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" >/dev/null
"$CMAKE" --build "$BUILD_DIR/cmake_out" -j 8 --target cpu_freq_actuator_probe cpu_loop_main 2>&1 | tail -3
BIN="$BUILD_DIR/cmake_out"

echo "=== SONDA REAL: latencia y asentamiento (turbo incluido, carga en 0-5) ==="
for c in 0 1 2 3 4 5; do taskset -c $c yes > /dev/null & done
sleep 1
"$BIN/cpu_freq_actuator_probe" --root "$CPUFREQ" --cpus 0,1,2,3,4,5 --khz-seq 3200000,2900000 \
  --manage-turbo 1 --iterations 100 --settle-cpu 0
pkill -P $$ yes || true
sleep 1
state > "$WORK/state_tras_sonda.txt"
diff "$WORK/state_inicial.txt" "$WORK/state_tras_sonda.txt" && echo "tras la sonda: estado IDENTICO al inicial"

echo "=== CAOS: cpu_loop_main --arm activo real, cortado por senal ==="
cat > "$WORK/policy_c8.yaml" <<Y
policy:
  cpu-compute_bound: {action: actuar_experimental, chosen_level: F0, resolved_freq_khz: 3200000}
  cpu-memory_bound: {action: actuar_experimental, chosen_level: F1, resolved_freq_khz: 2900000}
Y
run_chaos() {  # $1 = SIGTERM|SIGINT
  local sig=$1 out="$WORK/chaos_$1.out"
  echo "--- corte con $sig ---"
  ( cd "$ROOT/fase3_daemon/cpu_loop" && LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH:-}" \
    python3 launch_cpu_daemon.py --policy-table "$WORK/policy_c8.yaml" --binary "$BIN/cpu_loop_main" --arm activo -- \
      --perf-cpus 0,1,2,3 --collector-cpu 6 --consumer-cpu 7 --log-path "$WORK/chaos_$1.jsonl" -v ) > "$out" 2>&1 &
  local pid=$!
  sleep 6
  echo "en pleno vuelo: no_turbo=$(cat $CPUFREQ/intel_pstate/no_turbo) cpu0 min=$(cat $CPUFREQ/cpu0/cpufreq/scaling_min_freq) max=$(cat $CPUFREQ/cpu0/cpufreq/scaling_max_freq) hermano cpu16 max=$(cat $CPUFREQ/cpu16/cpufreq/scaling_max_freq)"
  [ "$(cat $CPUFREQ/intel_pstate/no_turbo)" = "1" ] || { echo "el turbo NO estaba apagado en pleno vuelo" >&2; exit 1; }
  pkill -"${sig#SIG}" -P $pid cpu_loop_main 2>/dev/null || kill -"${sig#SIG}" $pid
  wait $pid || true
  tail -4 "$out"
  state > "$WORK/state_tras_$1.txt"
  diff "$WORK/state_inicial.txt" "$WORK/state_tras_$1.txt" && echo "tras $sig: estado IDENTICO al inicial"
  grep -c '"written":true' "$WORK/chaos_$1.jsonl" | sed 's/^/decisiones con written=true: /'
}
run_chaos SIGTERM
run_chaos SIGINT
echo "=== FIN: preflight del actuador OK ==="
