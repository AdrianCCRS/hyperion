#!/bin/bash
# Efecto de la frecuencia de la CPU sobre la duracion de cargas de GPU (Fase 3, piso de CPU mientras la GPU esta activa):
# corre gpu_cutlass_simt_dgemm_n4096 (compute) y gpu_rajaperf_stream_triad (memory) con la CPU (nucleos 0-5 y hermanos
# 16-21, el proceso fijado a 0-5) en REF (nativo, turbo activo) y en niveles fijos F0/F1/F4/F8 (turbo apagado), 3 repeticiones
# por celda, orden aleatorizado. Objetivo: saber si bajar la CPU a F1 (2.9 GHz) alarga la carga de GPU (F1-XDEV-006 midio
# +63-66% con la CPU al minimo, pero no F1). Toca frecuencias de CPU y turbo: nodo libre; restaura el estado original
# desde un snapshot (trap). NUNCA local. Sin --time. No intervenir a mano.
#SBATCH --job-name=hyp_cpu_freq_gpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_freq_gpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_freq_gpu_%j.err
set -uo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"
WORK="$ROOT/fase3_daemon/composite_apps/cpu_freq_gpu_work"; mkdir -p "$WORK"; rm -f "$WORK"/*.txt 2>/dev/null
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
C=/sys/devices/system/cpu; CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
hostname; git -C "$ROOT" log --oneline -1
# snapshot del estado original (turbo y rango por CPU) y restauracion garantizada
ORIG_NO_TURBO=$(cat $C/intel_pstate/no_turbo)
declare -A OMIN OMAX
for c in $CPUS; do OMIN[$c]=$(cat $C/cpu$c/cpufreq/scaling_min_freq); OMAX[$c]=$(cat $C/cpu$c/cpufreq/scaling_max_freq); done
restore() {
  for c in $CPUS; do echo "${OMAX[$c]}" > $C/cpu$c/cpufreq/scaling_max_freq 2>/dev/null; echo "${OMIN[$c]}" > $C/cpu$c/cpufreq/scaling_min_freq 2>/dev/null; done
  sudo -n /usr/local/bin/set_turbo_state "$ORIG_NO_TURBO" 2>/dev/null
}
trap restore EXIT
set_level() {  # $1 = REF | kHz fijo ; REF = rango original y turbo original
  if [ "$1" = REF ]; then restore; return; fi
  sudo -n /usr/local/bin/set_turbo_state 1 || return 1   # 1 = turbo APAGADO
  for c in $CPUS; do
    if [ "$(cat $C/cpu$c/cpufreq/scaling_min_freq)" -gt "$1" ]; then echo "$1" > $C/cpu$c/cpufreq/scaling_min_freq; echo "$1" > $C/cpu$c/cpufreq/scaling_max_freq
    else echo "$1" > $C/cpu$c/cpufreq/scaling_max_freq; echo "$1" > $C/cpu$c/cpufreq/scaling_min_freq; fi
  done
}
cells=(); for lvl in REF 3200000 2900000 2000000 800000; do for k in gpu_cutlass_simt_dgemm_n4096 gpu_rajaperf_stream_triad; do for r in 1 2 3; do cells+=("$lvl $k $r"); done; done; done
mapfile -t order < <(printf '%s\n' "${cells[@]}" | shuf --random-source=<(yes 20260923))
echo "celdas: ${#order[@]} (orden aleatorizado, semilla 20260923)"
: > "$WORK/results.txt"
for cell in "${order[@]}"; do
  read -r lvl k r <<<"$cell"
  set_level "$lvl" || { echo "no se pudo fijar $lvl" >&2; continue; }
  sleep 2
  out=$( ( cd "$KERNEL_ROOT" && taskset -c 0-5 python3 "$ROOT/fase3_daemon/composite_apps/composite_known.py" --node-id pacca-a100 \
      --kernels-root "$KERNEL_ROOT" --sequence $k --cycles 1 --boundaries-out "$WORK/tmp.jsonl" 2>&1 ) | grep -E "^\[ciclo" )
  dur=$(echo "$out" | grep -oE "en [0-9.]+s" | grep -oE "[0-9.]+")
  cur=$(cat $C/cpu0/cpufreq/scaling_cur_freq)
  echo "$lvl $k $r $dur $cur" | tee -a "$WORK/results.txt"
done
restore
python3 - "$WORK/results.txt" <<'PY'
import sys, statistics as st, collections
rows = [l.split() for l in open(sys.argv[1]) if len(l.split()) == 5 and l.split()[3] not in ("", "None")]
d = collections.defaultdict(list)
for lvl, k, r, dur, cur in rows: d[(k, lvl)].append(float(dur))
print("\n=== duracion mediana por kernel y nivel de CPU (s), y razon frente a REF ===")
for k in sorted({k for k, _ in d}):
    ref = st.median(d[(k, "REF")]) if d.get((k, "REF")) else float("nan")
    for lvl in ("REF", "3200000", "2900000", "2000000", "800000"):
        v = d.get((k, lvl))
        if v: print(f"  {k:32s} {lvl:>8s}  mediana={st.median(v):7.2f}s  n={len(v)}  vs REF x{st.median(v)/ref:.3f}  (min {min(v):.2f} max {max(v):.2f})")
PY
echo "estado final: no_turbo=$(cat $C/intel_pstate/no_turbo) (original $ORIG_NO_TURBO) cpu0 min/max=$(cat $C/cpu0/cpufreq/scaling_min_freq)/$(cat $C/cpu0/cpufreq/scaling_max_freq)"
