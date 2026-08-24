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
# invalidar el ajuste de alpha por el I/O de 290 MB -- ver Anexo L.1;
# aquí no hay ese riesgo porque RAJAPerf no escribe archivos de salida).
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
#SBATCH --time=00:45:00
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

echo "kernel,level,khz_target,t_start_ns,t_end_ns,elapsed_s,pkg_start_uj,pkg_end_uj,dram_start_uj,dram_end_uj,energy_j"

for kernel in "${CANDIDATES[@]}"; do
  for lv in "${LEVELS[@]}"; do
    id="${lv%%:*}"
    khz="${lv##*:}"
    set_freq "$khz"

    run_dir="$(mktemp -d -p /home/latorresn hyperion_rajaperf_screen_XXXXXX)"
    cd "$run_dir" || exit 1

    pkg_start="$(cat "$RAPL_PKG")"
    dram_start="$(cat "$RAPL_DRAM")"
    t_start=$(date +%s%N)

    export OMP_NUM_THREADS=6
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    "$BINARY" --warmup-disable -k "$kernel" -v Base_OpenMP >/dev/null 2>&1
    exit_code=$?

    t_end=$(date +%s%N)
    pkg_end="$(cat "$RAPL_PKG")"
    dram_end="$(cat "$RAPL_DRAM")"

    cd /home/latorresn || exit 1
    rm -rf -- "$run_dir"

    if [[ $exit_code -ne 0 ]]; then
      echo "AVISO: $kernel a $id salió con codigo $exit_code, fila omitida" >&2
      continue
    fi

    elapsed_s=$(python3 -c "print(($t_end-$t_start)/1e9)")
    energy_j=$(python3 -c "print((($pkg_end-$pkg_start)+($dram_end-$dram_start))/1e6)")
    echo "${kernel},${id},${khz},${t_start},${t_end},${elapsed_s},${pkg_start},${pkg_end},${dram_start},${dram_end},${energy_j}"
  done
done

echo "CPU_RAJAPERF_SCREEN_DONE" >&2
