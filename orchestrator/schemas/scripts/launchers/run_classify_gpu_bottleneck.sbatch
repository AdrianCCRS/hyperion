#!/bin/bash
#SBATCH --job-name=hyp_gpu_bottleneck
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_bottleneck_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_bottleneck_%j.err
#
# Clasifica el cuello de botella real (DRAM% vs SM%, medido con ncu) de los
# 79 kernels de RAJAPerf-CUDA. Ver scripts/pacca/classify_gpu_bottleneck.py
# para por que esto reemplaza al criterio de "conocimiento algoritmico" con
# el que se eligieron los 6 primeros candidatos.
#
# NO fija el reloj a proposito: la clasificacion del cuello de botella no
# lo necesita (DRAM% y SM% son fracciones del pico sostenible, no tiempos),
# asi que no aplica la regla de with_cpu_turbo_disabled.sh -- esa vale para
# corridas con nivel de frecuencia FIJO, donde el turbo invalida P_pico.
set -eo pipefail
# ARC-153: sin "-u" (rompe module load). set -e SI: leccion del job 6527,
# donde un crash de Python se reporto como exito.

source ~/hyperion-venv/bin/activate
python3 /home/latorresn/hyperion/scripts/pacca/classify_gpu_bottleneck.py
echo BOTTLENECK_SCRIPT_DONE
