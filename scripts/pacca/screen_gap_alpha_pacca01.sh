#!/bin/bash
# Triage EXPLORATORIO de alpha para GAP Benchmark Suite, en pacca01.
#
# ESTO NO ES UN NUMERO CITABLE. Es la etapa de "pasa/no pasa" antes de
# gastar tiempo de paccaA100 (ocupado por trabajos ajenos): responde solo
# "¿el tiempo de este kernel escala con el reloj EN ABSOLUTO?", no si cruza
# el umbral 0.226 -- ese umbral se derivo del modelo de potencia de
# paccaA100 y no es transferible a pacca01 (26 nucleos/socket contra 8,
# freq maxima y contencion de memoria distintas). Ver la nota de
# Estrategia_CPU_Fase2.md sobre por que pacca01 sirve para esto y no para
# mas.
#
# CUALQUIER candidato que sobreviva este triage SE REMIDE EN paccaA100
# antes de entrar al catalogo o de citarse en ningun documento.
#
# KERNELS: bfs y pr (PageRank) primero -- los dos mas usados en la
# literatura de GAP, acceso irregular dependiente del dato (recorrido de
# lista de adyacencia), el hueco que ni STREAM (regular, ancho de banda)
# ni ptrchase (latencia pura sin estructura de algoritmo) cubren.
#
# GRAFO: Kronecker sintetico (-g 22, 2^22 ~ 4.2M vertices, Graph500), se
# genera en el momento -- sin descargar los 275 GB de "make bench-graphs".
# -n 3: 3 iteraciones internas por corrida, para que el binario no
# domine su propio tiempo de arranque/generacion de grafo.
#
# Mismo patron de pineo de frecuencia (hermanos SMT incluidos, ARC-162/163)
# que screen_rajaperf_cpu_alpha_v2.sh -- pacca01 tambien usa intel_pstate
# y tiene SMT activo (verificado: cpu0 hermano de cpu52).
set -o pipefail

GAP_BIN_DIR="/home/latorresn/hyperion-kernels/libexec/gapbs"
DELEGATED_CPUS=(0 1 2 3 4 5)
LEVELS=(F0:3400000 F1:2600000 F2:1800000 F3:1000000 F4:800000)
GRAPH_SCALE=22
ITERATIONS=3
KERNELS=(bfs pr)

smt_siblings_for() {
  local cpu="$1"
  local path="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
  [[ -r "$path" ]] && tr ',' '\n' < "$path" | tr '-' '\n'
}

declare -a FREQ_CPUS
_expand_with_smt_siblings() {
  local -A seen=()
  FREQ_CPUS=()
  for cpu in "${DELEGATED_CPUS[@]}"; do
    [[ -z "${seen[$cpu]:-}" ]] && { FREQ_CPUS+=("$cpu"); seen[$cpu]=1; }
    while read -r sib; do
      [[ -z "$sib" ]] && continue
      [[ -z "${seen[$sib]:-}" ]] && { FREQ_CPUS+=("$sib"); seen[$sib]=1; }
    done < <(smt_siblings_for "$cpu")
  done
}
_expand_with_smt_siblings
echo "CPUs de frecuencia (delegados + hermanos SMT): ${FREQ_CPUS[*]}" >&2

set_freq() {
  local target_khz="$1"
  for cpu in "${FREQ_CPUS[@]}"; do
    local base="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    local current_min
    current_min="$(cat "${base}/scaling_min_freq")"
    if [[ "$target_khz" -lt "$current_min" ]]; then
      echo "$target_khz" > "${base}/scaling_min_freq"
      echo "$target_khz" > "${base}/scaling_max_freq"
    else
      echo "$target_khz" > "${base}/scaling_max_freq"
      echo "$target_khz" > "${base}/scaling_min_freq"
    fi
  done
  sleep 2
}

restore_freq() {
  for cpu in "${FREQ_CPUS[@]}"; do
    local base="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    echo "$(cat "${base}/cpuinfo_max_freq")" > "${base}/scaling_max_freq"
    echo "$(cat "${base}/cpuinfo_min_freq")" > "${base}/scaling_min_freq"
  done
}
trap restore_freq EXIT

sample_freq_bg() {
  local out="$1"
  : > "$out"
  while :; do
    for cpu in "${DELEGATED_CPUS[@]}"; do
      cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq" 2>/dev/null
    done
    sleep 0.02
  done >> "$out"
}

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "kernel,level,khz_target,elapsed_s,freq_mean_khz,freq_within_5pct,n_freq_samples,rc"

for lv in "${LEVELS[@]}"; do
  level_id="${lv%%:*}"
  khz="${lv##*:}"
  set_freq "$khz"

  for kernel in "${KERNELS[@]}"; do
    bin="$GAP_BIN_DIR/$kernel"
    if [[ ! -x "$bin" ]]; then
      echo "${kernel},${level_id},${khz},,,NO,0,BINARIO_AUSENTE"
      continue
    fi

    "$bin" -g "$GRAPH_SCALE" -n 1 >/dev/null 2>&1  # warmup, descartado

    freq_log="$(mktemp -p /home/latorresn hyperion_gap_freq_XXXXXX)"
    sample_freq_bg "$freq_log" &
    sampler_pid=$!

    t0=$(date +%s%N)
    timeout 300 "$bin" -g "$GRAPH_SCALE" -n "$ITERATIONS" >/dev/null 2>&1
    rc=$?
    t1=$(date +%s%N)

    kill "$sampler_pid" 2>/dev/null
    wait "$sampler_pid" 2>/dev/null

    python3 - "$kernel" "$level_id" "$khz" "$t0" "$t1" "$freq_log" "$rc" <<'PY'
import sys
kernel, level_id, khz, t0, t1, freq_path, rc = sys.argv[1:8]
elapsed_s = (int(t1) - int(t0)) / 1e9
freqs = []
try:
    with open(freq_path) as handle:
        for line in handle:
            line = line.strip()
            if line.isdigit():
                freqs.append(int(line))
except OSError:
    pass
target = int(khz)
mean = sum(freqs) / len(freqs) if freqs else None
within = "yes" if mean and abs(mean - target) <= 0.05 * target else "NO"
mean_txt = "" if mean is None else f"{mean:.0f}"
print(f"{kernel},{level_id},{target},{elapsed_s:.6f},{mean_txt},{within},{len(freqs)},{rc}")
PY
    rm -f -- "$freq_log"
  done
done

echo "GAP_SCREEN_PACCA01_DONE" >&2
