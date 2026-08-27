#!/bin/bash
# Tamizaje de alpha en GPU, VERSION 2 -- sobre los 43 candidatos que la
# clasificacion directa de cuello de botella (classify_gpu_bottleneck.py,
# job 6571, DRAM% vs SM% medido con ncu) senalo como MEMORY_BOUND, en vez
# de los 6 elegidos a mano en v1 (screen_rajaperf_gpu_alpha.sh).
#
# MISMO METODO que v1 (nvidia-smi -lgc + muestreo de potencia), mismo
# --sizefact 100 ya corregido (el intento original a tamano por defecto,
# job 6514, dio r2=0.31-0.53 -- inservible; ver v1 para el diagnostico
# completo). No se repite aqui esa leccion, se hereda.
#
# QUE MIDE ESTO QUE v1 NO MEDIA. v1 elegia candidatos por conocimiento
# algoritmico (dos familias de acceso: ancho de banda puro, reduccion/scan)
# y encontro que 4 de 6 pasaban el umbral. v2 no elige por conocimiento --
# tamiza TODOS los que la medicion directa de DRAM%/SM% ya senalo como
# memory-bound, para no repetir el sesgo de seleccion que el propio v1 de
# CPU tuvo que corregir (Estrategia_CPU_Fase2.md riesgo 2).
#
# Uso: sbatch run_screen_rajaperf_gpu_alpha_v2.sbatch
#SBATCH --job-name=hyp_gpu_rajaperf_screen_v2
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_rajaperf_screen_v2_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_rajaperf_screen_v2_%j.err
set -o pipefail
# Sin "-u" a proposito -- rompe "module load" en este cluster (ARC-126).

BINARY="/home/latorresn/hyperion-kernels/libexec/raja-perf-cuda-v2025.12.1"
EXPECTED_SHA256="4803d205ec61a129f6901508779bd73a415f49604538bac70bfb877bbafbf067"

# Los 43 candidatos MEMORY_BOUND de classify_gpu_bottleneck.py (job 6571),
# ordenados por DRAM% descendente tal como salieron del clasificador.
CANDIDATES=(
  Lcals_TRIDIAG_ELIM
  Apps_PRESSURE
  Stream_ADD
  Stream_TRIAD
  Basic_DAXPY
  Lcals_HYDRO_1D
  Lcals_EOS
  Lcals_PLANCKIAN
  Basic_COPY8
  Lcals_GEN_LIN_RECUR
  Basic_NESTED_INIT
  Apps_ENERGY
  Stream_COPY
  Stream_MUL
  Algorithm_MEMCPY
  Lcals_FIRST_DIFF
  Lcals_FIRST_SUM
  Basic_INIT_VIEW1D
  Algorithm_MEMSET
  Basic_INIT_VIEW1D_OFFSET
  Polybench_JACOBI_1D
  Basic_IF_QUAD
  Polybench_FDTD_2D
  Lcals_INT_PREDICT
  Basic_INIT3
  Basic_MULADDSUB
  Polybench_FLOYD_WARSHALL
  Polybench_JACOBI_2D
  Lcals_HYDRO_2D
  Apps_ZONAL_ACCUMULATION_3D
  Basic_ARRAY_OF_PTRS
  Apps_DEL_DOT_VEC_2D
  Apps_MATVEC_3D_STENCIL
  Basic_INDEXLIST_3LOOP
  Polybench_HEAT_3D
  Comm_HALO_PACKING
  Algorithm_SCAN
  Apps_VOL3D
  Algorithm_REDUCE_SUM
  Basic_INDEXLIST
  Apps_EDGE3D
  Apps_FEMSWEEP
  Polybench_GEMVER
)

SETTLE_SECONDS=3
SAMPLE_INTERVAL=0.05
SIZEFACT=100

actual_sha256="$(sha256sum "$BINARY" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: checksum de raja-perf-cuda no coincide ($actual_sha256)" >&2
  exit 1
fi

MAX_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -n | tail -1)
MIN_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -n | head -1)

LEVELS=(F0:1.0 F1:0.75 F2:0.5 F3:0.25 F4:0.0)

restore_clock() {
  sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1 || true
}
trap restore_clock EXIT

sample_power_bg() {
  local out="$1"
  : > "$out"
  while :; do
    nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits
    sleep "$SAMPLE_INTERVAL"
  done >> "$out"
}

run_kernel() {
  local kernel="$1"
  "$BINARY" -k "$kernel" -v Base_CUDA --sizefact "$SIZEFACT" >/dev/null 2>&1
  return $?
}

echo "kernel,level,mhz_target,elapsed_s,energy_j,power_mean_mw,clock_observed_mhz,clock_within_5pct,n_power_samples"

for kernel in "${CANDIDATES[@]}"; do
  for lv in "${LEVELS[@]}"; do
    id="${lv%%:*}"
    frac="${lv##*:}"
    target_mhz=$(python3 -c "print(round(${MIN_MHZ} + ${frac} * (${MAX_MHZ} - ${MIN_MHZ})))")

    sudo nvidia-smi -i 0 -lgc "${target_mhz},${target_mhz}" >/dev/null
    sleep "$SETTLE_SECONDS"

    run_kernel "$kernel" || true  # warmup descartado

    power_log="$(mktemp -p /home/latorresn hyperion_gpu_powerlog_v2_XXXXXX)"
    sample_power_bg "$power_log" &
    sampler_pid=$!

    t_start=$(date +%s%N)
    run_kernel "$kernel"
    exit_code=$?
    t_end=$(date +%s%N)

    kill "$sampler_pid" 2>/dev/null
    wait "$sampler_pid" 2>/dev/null

    observed_mhz=$(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits)

    if [[ $exit_code -ne 0 ]]; then
      echo "AVISO: $kernel a $id salio con codigo $exit_code, fila omitida" >&2
      rm -f -- "$power_log"
      continue
    fi

    elapsed_s=$(python3 -c "print(($t_end-$t_start)/1e9)")

    row="$(python3 -c "
target = ${target_mhz}
observed = ${observed_mhz}
elapsed_s = ${elapsed_s}
vals = [float(x) for x in open('${power_log}').read().split() if x.strip()]
if not vals:
    energy_j, mean_mw, within, n = float('nan'), float('nan'), 'NO', 0
else:
    mean_mw = sum(vals) / len(vals)
    within = 'yes' if abs(observed - target) <= 0.05 * target else 'NO'
    energy_j = mean_mw / 1000.0 * elapsed_s
    n = len(vals)
print(f'{energy_j:.3f},{mean_mw:.1f},{observed},{within},{n}')
")"
    rm -f -- "$power_log"

    echo "${kernel},${id},${target_mhz},${elapsed_s},${row}"
  done
done

echo "GPU_RAJAPERF_SCREEN_V2_DONE" >&2
