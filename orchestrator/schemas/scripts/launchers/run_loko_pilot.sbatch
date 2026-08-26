#!/bin/bash
#SBATCH --job-name=hyp_loko_pilot
#SBATCH --partition=normal
#SBATCH --nodelist=pacca01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/loko_pilot_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/loko_pilot_%j.err
#
# Primer entrenamiento/evaluacion real de la arquitectura por par bajo
# LOKO (riesgo 4 de Estrategia_GPU_Fase2.md). Ver
# classifier/analysis/loko_pilot.py.
#
# PARTICION `normal`/pacca01 A PROPOSITO, no paccaA100: es analisis puro
# (pandas + sklearn sobre CSVs ya en disco), no necesita GPU ni el nodo
# exclusivo, y paccaA100 esta ocupado por la campaña final de GPU (job
# 6529) durante ~10 h. La divergencia de CPU/paquetes entre pacca01 y
# paccaA100 (ver scripts/pacca/build_rajaperf_cuda.sh) solo afecta a
# COMPILAR binarios; ejecutar Python puro sobre datos ya medidos no la
# toca.
#
# --mem=32G explicito: DefMemPerNode=UNLIMITED en este cluster hace que un
# job sin --mem pida el nodo entero y bloquee todo backfill. La agregacion
# es en streaming (una corrida a la vez), asi que 32G sobra -- un intento
# previo que cargaba los 10.25 M de filas del eje CPU de una vez murio por
# OOM en el portatil del usuario, y ese es justamente el modo de fallo que
# el streaming evita.
set -eo pipefail
# ARC-153: sin "-u" a proposito (rompe module load). set -e SI, para que un
# fallo de Python no se reporte como exito (leccion del job 6527).

source ~/hyperion-venv/bin/activate
cd ~/hyperion

echo "############## EJE GPU (rejilla fina, job 6471) ##############"
python3 classifier/analysis/loko_pilot.py --axis gpu \
  --out-json /home/latorresn/hyperion-results/analysis/loko_pilot_gpu.json

echo
echo "############## EJE CPU (arc174 corregido) ##############"
python3 classifier/analysis/loko_pilot.py --axis cpu \
  --out-json /home/latorresn/hyperion-results/analysis/loko_pilot_cpu.json

echo LOKO_PILOT_DONE
