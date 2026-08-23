#!/bin/bash
# Sonda de reposo GPU v2 (ARC-194) -- corrige dos huecos de la v1
# (ARC-185/G.4): 1) la v1 no excluía el transitorio de asentamiento justo
# tras `nvidia-smi -lgc`, y F1 mostró un pico de +16 W sin explicar sobre
# solo 20 s de muestra; 2) la v1 nunca midió REF (gobernador nativo, sin
# reloj fijado), así que ese nivel no tiene línea de reposo ni margen.
#
# Cambios: 60 s por nivel (antes 20 s), primeros 3 s descartados tras fijar
# el reloj (mismo principio que `frequency_settle` ya aplica en CPU), y
# nivel REF agregado (sin `-lgc`, se deja el reloj sin fijar y se muestrea
# tal cual el gobernador lo deje).
#
# Uso: sbatch measure_gpu_idle_power_v2.sh
#SBATCH --job-name=hyp_gpu_idle_v2
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_idle_power_v2_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_idle_power_v2_%j.err
set -e -o pipefail

SETTLE_SECONDS=3
SAMPLE_SECONDS=60
SAMPLE_INTERVAL=0.2

echo "gpu_index,level_id,target_mhz,observed_mhz,mean_power_mw,median_power_mw,p95_power_mw,min_power_mw,max_power_mw,n_samples"

MAX_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -n | tail -1)
MIN_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=graphics --format=csv,noheader,nounits | sort -n | head -1)

for LEVEL in REF:none F0:1.0 F1:0.75 F2:0.5 F3:0.25 F4:0.0; do
    id="${LEVEL%%:*}"
    frac="${LEVEL##*:}"

    if [[ "$frac" == "none" ]]; then
        sudo nvidia-smi -i 0 -rgc >/dev/null
        target="native"
    else
        target=$(python3 -c "print(round(${MIN_MHZ} + ${frac} * (${MAX_MHZ} - ${MIN_MHZ})))")
        sudo nvidia-smi -i 0 -lgc "${target},${target}" >/dev/null
    fi

    # Transitorio de asentamiento descartado, sin muestrear.
    sleep "$SETTLE_SECONDS"

    samples=()
    n=$(python3 -c "print(int(${SAMPLE_SECONDS}/${SAMPLE_INTERVAL}))")
    for _ in $(seq 1 "$n"); do
        p=$(nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits)
        samples+=("$p")
        sleep "$SAMPLE_INTERVAL"
    done

    observed=$(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits)

    stats=$(printf '%s\n' "${samples[@]}" | python3 -c "
import sys
vals = sorted(float(x) * 1000.0 for x in sys.stdin)  # W -> mW
n = len(vals)
mean = sum(vals) / n
median = vals[n // 2]
p95 = vals[int(n * 0.95)]
print(f'{mean:.1f},{median:.1f},{p95:.1f},{vals[0]:.1f},{vals[-1]:.1f}')
")
    echo "0,${id},${target},${observed},${stats},${n}"
done

sudo nvidia-smi -i 0 -rgc >/dev/null
echo "GPU_IDLE_PROBE_V2_DONE"
