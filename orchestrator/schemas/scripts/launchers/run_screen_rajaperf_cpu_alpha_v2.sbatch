#!/bin/bash
#SBATCH --job-name=hyp_cpu_screen_v2
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/cpu_screen_v2_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/cpu_screen_v2_%j.err
#
# Tamizaje de alpha en CPU con conjunto de trabajo 10x la LLC. Ver
# scripts/pacca/screen_rajaperf_cpu_alpha_v2.sh -- rehace el job 6483,
# cuya conclusion ("0 de 7 sobrevivientes") midio el tamaño del problema y
# no los kernels: a tamaño por defecto el conjunto de trabajo (~32 MB)
# cabia en la L3 de 39 MB de este nodo.
#
# EN paccaA100: toda la caracterizacion de CPU del proyecto se hizo aqui y
# los CPUs de las dos particiones son DISTINTOS (Xeon Gold 5315Y vs 5320),
# asi que un alpha medido en el otro nodo no seria comparable con nada de
# lo ya medido ni con el umbral 0.226, que se derivo del modelo de
# potencia de ESTE procesador.
#
# --exclusive: la medida de energia es RAPL de ambito de SOCKET; cualquier
# otro proceso en el nodo entra en el contador.
#
# --mem=64G: el conjunto de trabajo objetivo es 10x la LLC (~400 MB) por
# kernel, con holgura para las estructuras auxiliares de RAJAPerf.
# Explicito porque DefMemPerNode=UNLIMITED haria pedir el nodo entero y
# bloquear todo backfill.
#
# ENVUELTO EN with_cpu_turbo_disabled.sh, obligatorio: fija niveles
# concretos (F0-F4) y con turbo activo el reloj real no es el pedido.
set -eo pipefail
# ARC-153: sin "-u" (rompe module load). set -e SI (leccion del job 6527).

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
bash ~/hyperion/scripts/pacca/screen_rajaperf_cpu_alpha_v2.sh
'
echo CPU_SCREEN_V2_SCRIPT_DONE
