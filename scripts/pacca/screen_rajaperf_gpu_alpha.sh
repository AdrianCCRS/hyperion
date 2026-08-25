#!/bin/bash
# Tamizaje de alpha para candidatos de diversidad GPU de RAJAPerf-CUDA.
#
# Mismo proposito que screen_rajaperf_cpu_alpha.sh, del otro lado del
# nodo: medir barato, ANTES de catalogar + correr una campana completa,
# si un kernel candidato responde de verdad al reloj de GPU. El catalogo
# GPU esta atascado en 7 familias reales (Estrategia_GPU_Fase2.md riesgo
# 6/nuevo hallazgo de diversidad de dwt2d, 2026-08-25) -- este script es
# el "impulso" que nunca se ejecuto, ahora con el binario CUDA ya
# compilado y verificado (build_rajaperf_cuda.sh, corrido en paccaA100,
# no en pacca01 -- ver esa leccion en el propio script).
#
# CANDIDATOS. Dos familias de acceso distintas, no solo "mas de lo mismo":
#   - Stream_COPY, Stream_TRIAD: ancho de banda puro, mismo linaje que
#     gpu_stream_bw (calibracion ya en el catalogo, alpha=0.071 medido) --
#     sirven de control positivo esperado, no solo de candidato.
#   - Basic_REDUCE3_INT, Basic_INDEXLIST_3LOOP: reduccion y
#     compactacion/scan, patrones de acceso que el catalogo actual (denso
#     GEMM, DWT, N-body tipo lavamd/myocyte) no cubre en absoluto.
#   - Polybench_JACOBI_2D, Polybench_HEAT_3D: stencils -- en CPU (tamizaje
#     6483) NINGUNO paso el umbral (alpha 0.599-0.852), pero la fisica de
#     GPU es otra (rango dinamico de potencia 2.54x vs 1.40x, sec.
#     rango-dinamico del libro) -- no se asume el mismo resultado sin medir.
#
# NO PASA POR EL ORQUESTADOR, mismo motivo que el tamizaje CPU: no genera
# windows.csv ni etiqueta que dependa de nada mas que tiempo de pared +
# potencia NVML, asi que no hay preflight que saltarse.
#
# Mismo patron de potencia que measure_gpu_idle_power_v2.sh (SETTLE_SECONDS
# tras -lgc, muestreo nivelado de nvidia-smi --query-gpu=power.draw).
#
# Uso: sbatch screen_rajaperf_gpu_alpha.sh
#SBATCH --job-name=hyp_gpu_rajaperf_screen
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_rajaperf_screen_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_rajaperf_screen_%j.err
set -o pipefail
# Sin "-u" a proposito -- rompe "module load" en este cluster (ARC-126).

BINARY="/home/latorresn/hyperion-kernels/libexec/raja-perf-cuda-v2025.12.1"
EXPECTED_SHA256="4803d205ec61a129f6901508779bd73a415f49604538bac70bfb877bbafbf067"

CANDIDATES=(
  Stream_COPY
  Stream_TRIAD
  Basic_REDUCE3_INT
  Basic_INDEXLIST_3LOOP
  Polybench_JACOBI_2D
  Polybench_HEAT_3D
)

SETTLE_SECONDS=3
SAMPLE_INTERVAL=0.05

actual_sha256="$(sha256sum "$BINARY" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: checksum de raja-perf-cuda no coincide ($actual_sha256)" >&2
  exit 1
fi

MAX_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -n | tail -1)
MIN_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -n | head -1)

# id:fraccion -- mismos 5 niveles fijos estandar de GPU (REF se maneja
# aparte, sin -lgc). fraccion=none marca REF.
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

# SIZEFACT: primer intento (job 6514, tamano por defecto) dio r^2=0.31-0.53
# en los 6 candidatos -- inservible. Diagnostico (2026-08-25): a tamano por
# defecto el binario completo tarda ~441ms, de los cuales ~378ms son
# arranque de contexto CUDA fijo, no computo del kernel (medido variando
# --sizefact vs --repfact por separado). Mismo problema de fondo que dwt2d
# (Estrategia_GPU_Fase2.md riesgo 6): la ventana medida queda dominada por
# overhead de proceso, no por el kernel. Con --sizefact 100 el computo pasa
# a ser ~94% del tiempo total (378ms fijo + ~6.25s de computo), suficiente
# para que la frecuencia de GPU se refleje limpio en el tiempo.
SIZEFACT=100

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

    # Warmup descartado: deja al reloj asentarse bajo carga real antes de
    # cronometrar (mismo principio que el tamizaje CPU, ARC-160/164).
    run_kernel "$kernel" || true

    power_log="$(mktemp -p /home/latorresn hyperion_gpu_powerlog_XXXXXX)"
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

echo "GPU_RAJAPERF_SCREEN_DONE" >&2
