#!/bin/bash
# Tamizaje de alpha para LULESH y HPCG, directo en paccaA100 -- segundo y
# tercer paso del pivote de catalogo motivado por C8
# (Estrategia_CPU_Fase2.md §7.bis / §6.septies), tras el negativo de GAP
# (job 6601, alpha 0.69-0.74).
#
# TAMAÑOS, elegidos para exceder claramente la LLC real (12 MB, medida en
# vivo en otros scripts -- aqui fijos porque ambos binarios toman el
# tamaño por linea de comandos, no hay --memory-touched):
#   LULESH -s 120 -i 300 : malla 120^3, ~30 arreglos de double por
#     dominio -> muy por encima de 12 MB (verificado: 83s a F0/3200MHz).
#   HPCG --nx=80 --ny=80 --nz=80 --rt=5 : dominio local 80^3 filas,
#     ~13.8M no-ceros -> ~166 MB de matriz dispersa, muy por encima de la
#     LLC (verificado: ~20s a F0/3200MHz con --rt=5).
#
# MISMO metodo de frecuencia que screen_gap_alpha_paccaA100.sh: escritura
# directa de scaling_min/max_freq (P1), expansion de hermanos SMT
# (ARC-162/163), energia RAPL pkg+dram.
#
# Uso: sbatch run_screen_lulesh_hpcg_alpha_paccaA100.sbatch
set -o pipefail
# ARC-126: sin "-u" a proposito.

LULESH_BIN="/home/latorresn/hyperion-kernels/libexec/lulesh/lulesh2.0"
HPCG_BIN="/home/latorresn/hyperion-kernels/libexec/hpcg/xhpcg"
DELEGATED_CPUS=(0 1 2 3 4 5)
LEVELS=(F0:3200000 F1:2600000 F2:2000000 F3:1400000 F4:800000)

for bin in "$LULESH_BIN" "$HPCG_BIN"; do
  if [[ ! -x "$bin" ]]; then
    echo "ERROR: binario ausente o no ejecutable: $bin" >&2
    exit 1
  fi
done

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

run_kernel() {
  local kernel="$1"
  case "$kernel" in
    lulesh) timeout 600 "$LULESH_BIN" -s 120 -i 300 -q >/dev/null 2>&1 ;;
    hpcg)
      local run_dir
      run_dir="$(mktemp -d -p /home/latorresn hyperion_hpcg_run_XXXXXX)"
      ( cd "$run_dir" && timeout 600 "$HPCG_BIN" --nx=80 --ny=80 --nz=80 --rt=5 >/dev/null 2>&1 )
      local rc=$?
      rm -rf -- "$run_dir"
      return "$rc"
      ;;
  esac
}

echo "kernel,level,khz_target,elapsed_s,energy_j,freq_mean_khz,freq_within_5pct,n_freq_samples,rc"

for lv in "${LEVELS[@]}"; do
  level_id="${lv%%:*}"
  khz="${lv##*:}"
  set_freq "$khz"

  for kernel in lulesh hpcg; do
    run_kernel "$kernel"  # warmup, descartado

    freq_log="$(mktemp -p /home/latorresn hyperion_lulesh_hpcg_freq_XXXXXX)"
    sample_freq_bg "$freq_log" &
    sampler_pid=$!

    e0="$(read_energy_uj)"
    t0=$(date +%s%N)
    run_kernel "$kernel"
    rc=$?
    t1=$(date +%s%N)
    e1="$(read_energy_uj)"

    kill "$sampler_pid" 2>/dev/null
    wait "$sampler_pid" 2>/dev/null

    python3 - "$kernel" "$level_id" "$khz" "$t0" "$t1" "$e0" "$e1" "$freq_log" "$rc" <<'PY'
import sys
kernel, level_id, khz, t0, t1, e0, e1, freq_path, rc = sys.argv[1:10]
elapsed_s = (int(t1) - int(t0)) / 1e9
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

echo "LULESH_HPCG_SCREEN_PACCAA100_DONE" >&2
