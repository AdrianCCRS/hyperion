#!/bin/bash
#SBATCH --job-name=hyp_util_reales
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/analisis/util_reales_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analisis/util_reales_%j.err
#
# Analisis de solo lectura sobre datos YA medidos: no ejecuta kernels, no
# toca frecuencias, no necesita GPU. Va a la particion `normal` justamente
# para NO competir por paccaA100 (donde corre la cadena de cribado) y para
# no usar el nodo de login.
#
# Sin --time: se detiene a mano con scancel si hiciera falta.
# Solo libreria estandar de Python, sin venv, para no depender de que
# pacca01 tenga las mismas dependencias que paccaA100.

set -eo pipefail

mkdir -p /home/latorresn/hyperion-results/analisis

echo "nodo: $(hostname)"
echo "inicio: $(date -Is)"
echo

python3 /home/latorresn/hyperion/scripts/pacca/analysis/utilizacion_gpu_kernels_reales.py

echo
echo "fin: $(date -Is)"
echo ANALISIS_TAMANOS_DONE
