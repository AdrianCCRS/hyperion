#!/bin/bash
# Prueba V8 (docs/general/Estrategia_GPU_Fase2.md): ¿existe el reloj de
# memoria como SEGUNDO mando de DVFS en esta A100?
#
# POR QUÉ IMPORTA. Todo el trabajo de GPU hasta hoy escaló solo el reloj de
# SM (`nvidia-smi -lgc`). El Anexo K.4 mostró que eso explica por qué el
# margen vive en kernels limitados por ancho de banda: el reloj de memoria
# queda intacto, así que un kernel memory-bound apenas se frena. Varios
# trabajos citados (Fan2020, Guerreiro2019) escalan AMBOS relojes, y es la
# hipótesis pendiente para dwt2d/stream_bw, que no respondieron al de SM.
#
# Esta sonda NO mide rendimiento: solo responde tres preguntas de
# disponibilidad, que es lo que decide si esa línea de trabajo existe:
#   1. ¿Cuántos relojes de memoria distintos soporta el dispositivo?
#   2. ¿`-lmc` (lock memory clock) es aceptado con los permisos actuales?
#   3. ¿El reloj fijado se SOSTIENE, o el driver lo revierte?
#
# Si solo hay un reloj de memoria soportado (caso común en A100, cuya HBM2
# suele exponer un único punto operativo), la respuesta es negativa y esa
# línea se cierra con evidencia en vez de quedar como pendiente eterno.
#
# Uso: sbatch probe_gpu_memory_clock.sh
#SBATCH --job-name=hyp_gpu_memclk
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_memclock_probe_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_memclock_probe_%j.err
set -o pipefail

echo "===== 1. RELOJES DE MEMORIA SOPORTADOS ====="
mem_clocks="$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits | sort -un)"
echo "$mem_clocks"
n_mem="$(echo "$mem_clocks" | grep -c .)"
echo "n_relojes_memoria=${n_mem}"
echo

echo "===== 2. RELOJES DE NUCLEO SOPORTADOS (referencia) ====="
nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits | sort -un | tr '\n' ' '
echo
echo "n_relojes_nucleo=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits | sort -un | wc -l)"
echo

if [[ "$n_mem" -le 1 ]]; then
    echo "===== VEREDICTO ====="
    echo "V8 = NEGATIVO: el dispositivo expone ${n_mem} reloj(es) de memoria."
    echo "No hay segundo mando de DVFS disponible. La linea de trabajo de"
    echo "escalado de memoria (riesgo 2 de Estrategia_GPU_Fase2.md) se cierra"
    echo "con evidencia, no queda pendiente."
    echo "GPU_MEMCLOCK_PROBE_DONE"
    exit 0
fi

echo "===== 3. PRUEBA DE ESCRITURA Y PERSISTENCIA DE -lmc ====="
# Se prueba el reloj de memoria mas bajo distinto del maximo.
target_mem="$(echo "$mem_clocks" | head -1)"
max_mem="$(echo "$mem_clocks" | tail -1)"
echo "objetivo=${target_mem} MHz (maximo soportado=${max_mem} MHz)"

before="$(nvidia-smi -i 0 --query-gpu=clocks.mem --format=csv,noheader,nounits)"
echo "clocks.mem antes=${before}"

if sudo nvidia-smi -i 0 -lmc "${target_mem},${target_mem}" 2>&1; then
    echo "-lmc ACEPTADO"
    sleep 3
    after="$(nvidia-smi -i 0 --query-gpu=clocks.mem --format=csv,noheader,nounits)"
    echo "clocks.mem despues=${after}"
    if [[ "$after" == "$target_mem" ]]; then
        echo "V8 = POSITIVO: el reloj de memoria es fijable y se sostiene."
    else
        echo "V8 = PARCIAL: -lmc fue aceptado pero el reloj observado (${after})"
        echo "no coincide con el objetivo (${target_mem}) -- el driver lo revirtio."
    fi
    sudo nvidia-smi -i 0 -rmc >/dev/null 2>&1 || true
    echo "reloj de memoria restaurado"
else
    echo "V8 = NEGATIVO: -lmc RECHAZADO (permisos o no soportado)."
    echo "Nota: -lgc (reloj de nucleo) SI funciona con los permisos actuales"
    echo "(P4 concedido, ARC-137), asi que esto seria una restriccion"
    echo "especifica del mando de memoria, no un problema general de permisos."
fi

echo "GPU_MEMCLOCK_PROBE_DONE"
