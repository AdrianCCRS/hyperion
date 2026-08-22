#!/bin/bash
#SBATCH --job-name=hyp_gates
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gates_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gates_%j.err
#
# Compuertas C1/C2/C3 (ARC-178). Ver classifier/analysis/gates_c1_c2_c3.py.
#
# PARTICIÓN `normal`, NO paccaA100, y es deliberado. Esto es aritmética
# sobre windows.csv: los números ya se midieron en la A100 y están escritos
# en el archivo, así que el procesador del nodo de análisis no cambia el
# resultado -- con semilla fija el modelo ajustado es el mismo en cualquier
# x86. Mandarlo a la A100 solo le quitaría el nodo de medición a la cola
# compartida sin ganar nada.
#
# La excepción, que este job NO hace: la latencia de inferencia sí es una
# afirmación sobre el hardware de despliegue y tiene que medirse en
# paccaA100 cuando haya un modelo que valga la pena desplegar.
#
# Sin --exclusive: no se mide energía ni frecuencia aquí, así que otro
# inquilino en el nodo no contamina nada. 32 CPUs de los 104 dejan sitio.
set -e -o pipefail

mkdir -p /home/latorresn/hyperion-results/analysis
export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"
source ~/hyperion-venv/bin/activate

python3 /home/latorresn/hyperion/classifier/analysis/gates_c1_c2_c3.py \
  --campaign-dir /home/latorresn/hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174 \
  --out /home/latorresn/hyperion-results/analysis/gates_c1_c2_c3.json

echo GATES_DONE
