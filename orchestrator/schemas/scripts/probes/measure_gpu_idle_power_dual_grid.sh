#!/bin/bash
# Línea de reposo NVML para la rejilla GPU vigente del selector dual:
# REF + F0..F6 con fracciones 1, .833, .667, .5, .333, .167 y 0.
# Conserva el protocolo ya validado en measure_gpu_idle_power_v2.sh:
# 60 s por nivel y 3 s de asentamiento descartados. No define márgenes de
# actividad; solo mide la línea de reposo que validate_windows() necesita.
#
# Uso: sbatch measure_gpu_idle_power_dual_grid.sh
#SBATCH --job-name=hyp_gpu_idle_dual
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_idle_power_dual_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_idle_power_dual_%j.err
set -e -o pipefail

SETTLE_SECONDS=3
SAMPLE_SECONDS=60
SAMPLE_INTERVAL=0.2
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"

restore_gpu_clock() {
    sudo -n nvidia-smi -i "$GPU_INDEX" -rgc >/dev/null
}
trap restore_gpu_clock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mapfile -t AVAILABLE_MHZ < <(
    nvidia-smi -i "$GPU_INDEX" --query-supported-clocks=graphics \
      --format=csv,noheader,nounits | sort -n -u
)
if [[ "${#AVAILABLE_MHZ[@]}" -eq 0 ]]; then
    echo "G02: nvidia-smi no reportó relojes graphics disponibles" >&2
    exit 65
fi
MIN_MHZ="${AVAILABLE_MHZ[0]}"
MAX_MHZ="${AVAILABLE_MHZ[${#AVAILABLE_MHZ[@]}-1]}"

nearest_available() {
    local raw="$1"
    printf '%s\n' "${AVAILABLE_MHZ[@]}" | python3 -c '
import sys
target = float(sys.argv[1])
values = [int(line) for line in sys.stdin if line.strip()]
print(min(values, key=lambda value: abs(value - target)))
' "$raw"
}

echo "gpu_index,level_id,target_mhz,observed_mhz,mean_power_mw,median_power_mw,p95_power_mw,min_power_mw,max_power_mw,n_samples"

for LEVEL in REF:none F0:1.0 F1:0.833 F2:0.667 F3:0.5 F4:0.333 F5:0.167 F6:0.0; do
    id="${LEVEL%%:*}"
    frac="${LEVEL##*:}"

    if [[ "$frac" == "none" ]]; then
        sudo -n nvidia-smi -i "$GPU_INDEX" -rgc >/dev/null
        target="native"
    else
        raw=$(python3 -c "print(round(${MIN_MHZ} + ${frac} * (${MAX_MHZ} - ${MIN_MHZ})))")
        target="$(nearest_available "$raw")"
        sudo -n nvidia-smi -i "$GPU_INDEX" -lgc "${target},${target}" >/dev/null
    fi

    sleep "$SETTLE_SECONDS"
    samples=()
    n=$(python3 -c "print(int(${SAMPLE_SECONDS}/${SAMPLE_INTERVAL}))")
    for _ in $(seq 1 "$n"); do
        samples+=("$(nvidia-smi -i "$GPU_INDEX" --query-gpu=power.draw --format=csv,noheader,nounits)")
        sleep "$SAMPLE_INTERVAL"
    done

    observed=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks.sm --format=csv,noheader,nounits)
    stats=$(printf '%s\n' "${samples[@]}" | python3 -c '
import sys
values = sorted(float(value) * 1000.0 for value in sys.stdin)
n = len(values)
mean = sum(values) / n
median = values[n // 2]
p95 = values[min(n - 1, int(n * 0.95))]
print(f"{mean:.1f},{median:.1f},{p95:.1f},{values[0]:.1f},{values[-1]:.1f}")
')
    echo "${GPU_INDEX},${id},${target},${observed},${stats},${n}"
done

echo "GPU_IDLE_DUAL_GRID_DONE"
