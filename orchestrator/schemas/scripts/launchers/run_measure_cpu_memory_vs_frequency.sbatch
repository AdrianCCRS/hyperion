#!/bin/bash
#SBATCH --job-name=hyp_cpu_memfreq
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/cpu_memfreq_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/cpu_memfreq_%j.err
#
# ¿La memoria se frena cuando baja el reloj del nucleo? Ver
# scripts/pacca/measure_cpu_memory_vs_frequency.sh -- responde la pregunta
# sobre la que descansa toda la parte CPU del trabajo.
#
# EN paccaA100, no pacca01: toda la caracterizacion de CPU del proyecto se
# hizo en este nodo y los CPUs de ambas particiones son DISTINTOS (Xeon
# Gold 5315Y vs 5320). Una medicion de frecuencia de uncore en el otro
# nodo no seria comparable con nada de lo ya medido.
#
# --exclusive SIN --gres=gpu: es un experimento de CPU, no necesita la GPU,
# pero si necesita el nodo entero -- los contadores uncore son de AMBITO
# DE SOCKET y cualquier otro proceso contaminaria el trafico DRAM medido.
#
# ENVUELTO EN with_cpu_turbo_disabled.sh, obligatorio: fija niveles de
# frecuencia concretos (F0-F4) y con turbo activo el reloj real no es el
# pedido, lo que invalidaria tanto el eje x como P_pico (incidente real del
# 2026-08-19).
set -eo pipefail
# ARC-153: sin "-u" (rompe module load). set -e SI (leccion del job 6527).

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
bash ~/hyperion/scripts/pacca/measure_cpu_memory_vs_frequency.sh
'
echo CPU_MEMFREQ_SCRIPT_DONE
