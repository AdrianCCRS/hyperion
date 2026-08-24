#!/bin/bash
# Tamizaje de alpha para candidatos memory-bound de RAJAPerf en CPU.
#
# Mismo propósito que gpu_alpha_screening (Anexo K.7): antes de invertir en
# catalogar + correr una campaña completa (que en CPU además exige uncore,
# bloqueado por CAP_PERFMON/ARC-184), medir barato si el kernel tiene
# margen de DVFS real.
#
# POR QUÉ ESTO NO PASA POR EL ORQUESTADOR. run-campaign exige uncore para
# cualquier kernel device=cpu en `kernels:` (E12/E13, ARC-191). Este script
# no usa perf/uncore en absoluto -- solo tiempo de pared + RAPL (pkg+dram),
# igual que measure_gpu_idle_power*.sh y el mismo dato que ya usa
# cpu_policy_headroom.py. Bypasea el orquestador por diseño, no por atajo:
# no hay preflight que saltarse porque no se está generando windows.csv ni
# ninguna etiqueta que dependa de PMU.
#
# BINARIO. Un solo ejecutable (`raja-perf.exe`, aquí como
# ~/hyperion-kernels/libexec/raja-perf-v2025.12.1) corre cualquier kernel
# vía `-k <NOMBRE> -v Base_OpenMP` -- ya compilado y con checksum
# verificado (ver scripts/pacca/build_rajaperf_polybench_3mm_omp.sh).
# Agregar un kernel a este tamizaje es una línea nueva en CANDIDATES, cero
# compilación nueva.
#
# CANDIDATOS. Elegidos por conocimiento algorítmico clásico (stencils y
# productos matriz-vector son memory-bound: O(n^2) trabajo sobre O(n^2)
# datos, sin reuso), NO por medición -- el tamizaje es justamente para
# confirmarlo o refutarlo, igual que se hizo con dwt2d/myocyte/backprop en
# GPU (donde myocyte, el más memory-bound POR OI DECLARADA, resultó
# invalidar el ajuste de alpha por el I/O de 290 MB -- ver Anexo L.1). Ese
# riesgo NO se asume ausente aquí: cada corrida reporta `output_bytes`,
# para que la prueba C3 lo verifique con dato en vez de con supuesto.
# Se excluyen los ya representados por el catálogo actual: 2MM/3MM/GEMM
# (compute-bound denso, como dgemm_n2048) y ADI/FLOYD_WARSHALL (mixtos,
# de clasificación menos clara).
#
# Uso: sbatch screen_rajaperf_cpu_alpha.sh
#SBATCH --job-name=hyp_cpu_rajaperf_screen
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/cpu_rajaperf_screen_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/cpu_rajaperf_screen_%j.err
set -o pipefail
# Sin "-u": mismo criterio que with_cpu_turbo_disabled.sh (ARC-153), el
# entorno de módulos del nodo referencia variables no definidas.

BINARY="/home/latorresn/hyperion-kernels/libexec/raja-perf-v2025.12.1"
EXPECTED_SHA256="7f5251ac4c8f4bfd854441b7873f120080affec7bc77abfce1cc0fb9ec165ebb"
DELEGATED_CPUS=(0 1 2 3 4 5)
RAPL_PKG=/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
RAPL_DRAM=/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj

# id:khz -- mismos 5 niveles fijos que la campaña de CPU ya validada
# (pacca_cpu_final_attempt03_20260820_arc174).
LEVELS=(F0:3200000 F1:2600000 F2:2000000 F3:1400000 F4:800000)

CANDIDATES=(
  Polybench_JACOBI_1D
  Polybench_JACOBI_2D
  Polybench_HEAT_3D
  Polybench_FDTD_2D
  Polybench_ATAX
  Polybench_GESUMMV
  Polybench_MVT
)

actual_sha256="$(sha256sum "$BINARY" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: checksum de raja-perf.exe no coincide ($actual_sha256)" >&2
  exit 1
fi

set_freq() {
  local target_khz="$1"
  for cpu in "${DELEGATED_CPUS[@]}"; do
    local base="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    # Mismo orden protegido que freqctl.py::_write_range_safe (ARC-94):
    # nunca escribir max < min vigente en un paso intermedio.
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
  sleep 1
}

restore_freq() {
  for cpu in "${DELEGATED_CPUS[@]}"; do
    local base="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    local min max
    min="$(cat "${base}/cpuinfo_min_freq")"
    max="$(cat "${base}/cpuinfo_max_freq")"
    echo "$max" > "${base}/scaling_max_freq"
    echo "$min" > "${base}/scaling_min_freq"
  done
}
trap restore_freq EXIT

# Muestreador de scaling_cur_freq EN SEGUNDO PLANO, activo solo mientras el
# kernel corre. Es la corrección al riesgo 6 / prueba C2 de
# docs/general/Estrategia_CPU_Fase2.md: ARC-160/164 documentó que bajo
# intel_pstate+HWP con EPP=performance el decaimiento hacia un techo más
# bajo tarda SEGUNDOS, y que scaling_cur_freq leído en reposo NO refleja el
# pineo (se calcula del ratio APERF/MPERF, que en un núcleo inactivo no
# puede mostrar el candado alto). Por eso hay que muestrear BAJO CARGA.
sample_freq_bg() {
  local out="$1"
  : > "$out"
  while :; do
    for cpu in "${DELEGATED_CPUS[@]}"; do
      cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq" 2>/dev/null
    done
    sleep 0.2
  done >> "$out"
}

run_kernel() {
  local kernel="$1"
  local run_dir
  run_dir="$(mktemp -d -p /home/latorresn hyperion_rajaperf_screen_XXXXXX)"
  ( cd "$run_dir" && "$BINARY" --warmup-disable -k "$kernel" -v Base_OpenMP >/dev/null 2>&1 )
  local rc=$?
  # Tamaño de lo que el kernel dejó escrito -- insumo de la prueba C3
  # (contaminación por I/O; la lección de rodinia_myocyte, que escribía
  # 290 MB e invalidaba su propio alpha, Anexo L.1).
  KERNEL_OUTPUT_BYTES="$(du -sb "$run_dir" 2>/dev/null | awk '{print $1}')"
  rm -rf -- "$run_dir"
  return $rc
}

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "kernel,level,khz_target,elapsed_s,energy_j,freq_mean_khz,freq_min_khz,freq_max_khz,freq_within_5pct,n_freq_samples,governor,output_bytes"

for kernel in "${CANDIDATES[@]}"; do
  for lv in "${LEVELS[@]}"; do
    id="${lv%%:*}"
    khz="${lv##*:}"
    set_freq "$khz"

    # WARMUP descartado: deja a HWP asentarse BAJO CARGA en el nivel nuevo
    # antes de cronometrar. Sin esto, el transitorio de bajada (segundos,
    # ARC-160) contamina el arranque de la corrida medida y sesga alpha.
    run_kernel "$kernel" || true

    # Gobernador activo -- prueba C5 (no asumir que REF == performance).
    governor="$(cat "/sys/devices/system/cpu/cpu${DELEGATED_CPUS[0]}/cpufreq/scaling_governor" 2>/dev/null)"

    freq_log="$(mktemp -p /home/latorresn hyperion_freqlog_XXXXXX)"
    sample_freq_bg "$freq_log" &
    sampler_pid=$!

    pkg_start="$(cat "$RAPL_PKG")"
    dram_start="$(cat "$RAPL_DRAM")"
    t_start=$(date +%s%N)

    run_kernel "$kernel"
    exit_code=$?

    t_end=$(date +%s%N)
    pkg_end="$(cat "$RAPL_PKG")"
    dram_end="$(cat "$RAPL_DRAM")"

    kill "$sampler_pid" 2>/dev/null
    wait "$sampler_pid" 2>/dev/null

    if [[ $exit_code -ne 0 ]]; then
      echo "AVISO: $kernel a $id salió con codigo $exit_code, fila omitida" >&2
      rm -f -- "$freq_log"
      continue
    fi

    freq_stats="$(python3 -c "
import sys
target = ${khz}
vals = [float(x) for x in open('${freq_log}').read().split() if x.strip()]
if not vals:
    print('nan,nan,nan,0,0')
else:
    mean = sum(vals)/len(vals)
    within = 'yes' if abs(mean - target) <= 0.05*target else 'NO'
    print(f'{mean:.0f},{min(vals):.0f},{max(vals):.0f},{within},{len(vals)}')
")"
    rm -f -- "$freq_log"

    elapsed_s=$(python3 -c "print(($t_end-$t_start)/1e9)")
    energy_j=$(python3 -c "print((($pkg_end-$pkg_start)+($dram_end-$dram_start))/1e6)")
    echo "${kernel},${id},${khz},${elapsed_s},${energy_j},${freq_stats},${governor},${KERNEL_OUTPUT_BYTES}"
  done
done

echo "CPU_RAJAPERF_SCREEN_DONE" >&2
