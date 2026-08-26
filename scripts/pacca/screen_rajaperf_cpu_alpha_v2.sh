#!/bin/bash
# Tamizaje de alpha en CPU, VERSION 2 -- con conjunto de trabajo que
# EXCEDE la cache de ultimo nivel.
#
# POR QUE HAY UNA v2, Y POR QUE LA v1 NO ES CONCLUYENTE (2026-08-25).
# `screen_rajaperf_cpu_alpha.sh` (job 6483) corrio los 7 candidatos de
# Polybench SIN fijar el tamaño del problema, es decir al tamaño por
# defecto de RAJAPerf. Medido con `--dryrun`, `Polybench_JACOBI_1D` a ese
# tamaño mueve ~32 MB por repeticion. La L3 de este nodo son ~39 MB.
#
#   32 MB < 39 MB  =>  EL CONJUNTO DE TRABAJO CABIA EN CACHE.
#
# Los candidatos nunca tocaron DRAM de forma significativa: estaban
# midiendose contra L3, cuya latencia y ancho de banda escalan MUCHO mejor
# con el reloj del nucleo que los de memoria principal. Por eso los 7
# salieron aparentemente compute-bound (alpha 0.331-0.852) y se concluyo
# "0 de 7 sobrevivientes". Esa conclusion NO esta sostenida por el dato:
# mide una propiedad del tamaño elegido, no de los kernels.
#
# Es la misma clase de error ya encontrada y corregida en el eje GPU, donde
# el tamaño por defecto dejaba la medicion dominada por el arranque de
# contexto CUDA y `--sizefact 100` cambio r2 de 0.31-0.53 a >0.97.
#
# `--memory-touched` EN VEZ DE `--sizefact`. RAJAPerf permite fijar
# directamente cuantos bytes toca cada kernel por repeticion, y ajusta el
# tamaño de problema de cada uno para cumplirlo. Es mas principiado que un
# multiplicador: `--sizefact 10` sobre un kernel 3D no escala igual que
# sobre uno 1D, asi que un solo multiplicador dejaria a unos dentro y a
# otros fuera de cache. Con `--memory-touched` TODOS quedan al mismo
# multiplo de la LLC, que es la condicion que se quiere controlar.
#
# TARGET_BYTES = 10x la LLC. No es un numero magico: por debajo de ~2-3x
# el prefetcher y la propia LLC siguen absorbiendo buena parte del
# trafico, y el kernel parece mas compute-bound de lo que es. 10x deja el
# conjunto de trabajo inequivocamente en DRAM.
#
# TAMIZA TODOS LOS KERNELS, no 7 elegidos a mano. La v1 eligio 7
# candidatos por conocimiento algoritmico clasico. Con el binario ya
# compilado, correr los ~79 cuesta lo mismo por kernel y elimina el sesgo
# de seleccion -- mismo criterio que ya se aplico al tamizaje de GPU.
#
# Uso: sbatch run_screen_rajaperf_cpu_alpha_v2.sbatch
set -o pipefail
# ARC-126: sin "-u" a proposito.

BINARY="/home/latorresn/hyperion-kernels/libexec/raja-perf-v2025.12.1"
EXPECTED_SHA256="7f5251ac4c8f4bfd854441b7873f120080affec7bc77abfce1cc0fb9ec165ebb"

DELEGATED_CPUS=(0 1 2 3 4 5)
LEVELS=(F0:3200000 F1:2600000 F2:2000000 F3:1400000 F4:800000)

# Tamaño objetivo del conjunto de trabajo, leido de la LLC REAL del nodo en
# vivo (nunca hardcodeado: pacca01 y paccaA100 son CPUs distintos).
#
# Se lee de sysfs y no de `lscpu -B`: esa opcion no existe en la version de
# util-linux de este cluster (verificado 2026-08-25, "invalid option -- 'B'").
# sysfs expone el tamaño con sufijo de unidad ("39936K"), que hay que
# convertir a bytes explicitamente.
read_llc_bytes() {
  local raw index
  for index in 3 2; do  # index3 = L3; index2 = L2 como respaldo
    local path="/sys/devices/system/cpu/cpu0/cache/index${index}/size"
    [[ -r "$path" ]] || continue
    raw="$(tr -d '\n' < "$path")"
    case "$raw" in
      *K) echo $(( ${raw%K} * 1024 )); return 0 ;;
      *M) echo $(( ${raw%M} * 1024 * 1024 )); return 0 ;;
      *[0-9]) echo "$raw"; return 0 ;;
    esac
  done
  return 1
}
LLC_BYTES="$(read_llc_bytes)"
if [[ -z "$LLC_BYTES" || "$LLC_BYTES" -le 0 ]]; then
  echo "ERROR: no se pudo leer el tamaño de la LLC desde sysfs" >&2
  exit 1
fi
TARGET_BYTES=$(( LLC_BYTES * 10 ))
echo "L3 medida: ${LLC_BYTES} B; --memory-touched objetivo: ${TARGET_BYTES} B (10x LLC)" >&2

actual_sha256="$(sha256sum "$BINARY" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: checksum de raja-perf.exe no coincide ($actual_sha256)" >&2
  exit 1
fi

# --- Frecuencia: hermanos SMT incluidos (ARC-162/163) --------------------
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

RAPL_PKG=/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
RAPL_DRAM=/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj

read_energy_uj() {
  local pkg dram
  pkg="$(cat "$RAPL_PKG" 2>/dev/null)"
  dram="$(cat "$RAPL_DRAM" 2>/dev/null)"
  echo "$(( ${pkg:-0} + ${dram:-0} ))"
}

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

# Lista completa de kernels desde el propio binario -- sin seleccion previa.
mapfile -t CANDIDATES < <(
  "$BINARY" --print-kernels 2>/dev/null \
    | grep -E '^(Basic|Lcals|Polybench|Stream|Apps|Algorithm|Comm)_[A-Za-z0-9_]+$'
)
echo "kernels a tamizar: ${#CANDIDATES[@]}" >&2

echo "kernel,level,khz_target,elapsed_s,energy_j,freq_mean_khz,freq_within_5pct,n_freq_samples,rc"

for lv in "${LEVELS[@]}"; do
  level_id="${lv%%:*}"
  khz="${lv##*:}"
  set_freq "$khz"

  for kernel in "${CANDIDATES[@]}"; do
    run_dir="$(mktemp -d -p /home/latorresn hyperion_screen_v2_XXXXXX)"
    freq_log="$(mktemp -p /home/latorresn hyperion_screen_v2_freq_XXXXXX)"

    sample_freq_bg "$freq_log" &
    sampler_pid=$!

    e0="$(read_energy_uj)"
    t0=$(date +%s%N)
    ( cd "$run_dir" && timeout 300 "$BINARY" --warmup-disable \
        -k "$kernel" -v Base_OpenMP \
        --memory-touched "$TARGET_BYTES" >/dev/null 2>&1 )
    rc=$?
    t1=$(date +%s%N)
    e1="$(read_energy_uj)"

    kill "$sampler_pid" 2>/dev/null
    wait "$sampler_pid" 2>/dev/null
    rm -rf -- "$run_dir"

    python3 - "$kernel" "$level_id" "$khz" "$t0" "$t1" "$e0" "$e1" "$freq_log" "$rc" <<'PY'
import sys
kernel, level_id, khz, t0, t1, e0, e1, freq_path, rc = sys.argv[1:10]
elapsed_s = (int(t1) - int(t0)) / 1e9
# RAPL es un contador que envuelve; una diferencia negativa significa que
# dio la vuelta durante la corrida y ese valor no es utilizable.
delta_uj = int(e1) - int(e0)
energy_j = delta_uj / 1e6 if delta_uj >= 0 else None

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
energy_txt = "" if energy_j is None else f"{energy_j:.4f}"
mean_txt = "" if mean is None else f"{mean:.0f}"
print(f"{kernel},{level_id},{target},{elapsed_s:.6f},{energy_txt},"
      f"{mean_txt},{within},{len(freqs)},{rc}", flush=True)
PY

    rm -f -- "$freq_log"
  done
done

echo "CPU_SCREEN_V2_DONE" >&2
