#!/bin/bash
# Mide la potencia de reposo de la GPU en cada nivel de reloj SM del
# barrido, para el criterio de actividad invariante a la frecuencia
# (T2.3, ARC-185 -- reemplaza el piso de utilizacion de
# orchestrator/validation.py::_GPU_UTIL_NOISE_FLOOR_PCT).
#
# POR QUE. F3 del plan maestro: el piso de utilizacion del 5 % crece de
# forma monotona al bajar el reloj para un kernel realmente ocioso
# (rodinia_lud: 0.0, 0.0, 1.1, 2.9, 3.4, 3.5 % en REF/F0/F1/F2/F3/F4), asi
# que aceptaba mas corridas ociosas cuanto mas bajo el nivel -- sesgo
# dependiente de la frecuencia por construccion. La potencia SI es una
# magnitud fisica real (NVML, no una fraccion de tiempo con ruido de
# muestreo), asi que "potencia sobre el reposo" es invariante: un kernel
# ocioso deberia dar ~0 W de exceso en CUALQUIER nivel de reloj.
#
# No hace falta compilar nada: nvidia-smi solo, sin carga, en cada nivel.
#
# Uso: sbatch measure_gpu_idle_power.sh <turbo/no-op N/A para GPU>
#SBATCH --job-name=hyp_gpu_idle
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_idle_power_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_idle_power_%j.err
set -e -o pipefail

SAMPLE_SECONDS=20
SAMPLE_INTERVAL=0.2

# nvidia-smi reporta power.draw en WATIOS; el dataset (nvmlDeviceGetPowerUsage
# via telemetry/src/nvml_reader.cpp) guarda gpu_power_mw en MILIWATIOS. Se
# convierte aqui mismo para que el resultado sea directamente comparable con
# la columna del dataset, sin un factor de conversion oculto en otro sitio.
echo "gpu_index,level_id,target_mhz,observed_mhz_first,observed_mhz_last,mean_power_mw,min_power_mw,max_power_mw,n_samples"

# Niveles porcentuales del rango disponible, mismos puntos que
# gpu_frequency_levels en los manifiestos (F0=100%, F1=75%, F2=50%,
# F3=25%, F4=0%). Se leen los relojes SM disponibles en vivo -- no se
# asume ningun valor fijo de A100, el hardware real decide.
mapfile -t AVAILABLE_MHZ < <(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits)
MAX_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=sm --format=csv,noheader,nounits | sort -n | tail -1)
MIN_MHZ=$(nvidia-smi -i 0 --query-supported-clocks=sm --format=csv,noheader,nounits | sort -n | head -1)

for LEVEL in F0:1.0 F1:0.75 F2:0.5 F3:0.25 F4:0.0; do
    id="${LEVEL%%:*}"
    frac="${LEVEL##*:}"
    target=$(python3 -c "print(round(${MIN_MHZ} + ${frac} * (${MAX_MHZ} - ${MIN_MHZ})))")

    sudo nvidia-smi -i 0 -lgc "${target},${target}" >/dev/null

    observed_first=$(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits)

    samples=()
    n=$(python3 -c "print(int(${SAMPLE_SECONDS}/${SAMPLE_INTERVAL}))")
    for _ in $(seq 1 "$n"); do
        p=$(nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits)
        samples+=("$p")
        sleep "$SAMPLE_INTERVAL"
    done

    observed_last=$(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits)

    stats=$(printf '%s\n' "${samples[@]}" | python3 -c "
import sys
vals = [float(x) * 1000.0 for x in sys.stdin]  # W -> mW
print(f'{sum(vals)/len(vals):.1f},{min(vals):.1f},{max(vals):.1f},{len(vals)}')
")
    echo "0,${id},${target},${observed_first},${observed_last},${stats}"
done

sudo nvidia-smi -i 0 -rgc >/dev/null
echo "GPU_IDLE_PROBE_DONE"
