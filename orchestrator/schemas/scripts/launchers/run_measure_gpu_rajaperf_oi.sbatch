#!/bin/bash
#SBATCH --job-name=hyp_gpu_rajaperf_oi
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gpu_rajaperf_oi_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gpu_rajaperf_oi_%j.err
#
# Mide operational_intensity_flops_per_byte real (ncu) para los 6
# candidatos de RAJAPerf-CUDA que sobrevivieron el tamizaje (job 6517).
# Reemplaza al intento fallido con srun+timeout (job 6525, murio sin
# canal de E/S porque el proceso local que lo lanzo se mato a los 90s
# mientras esperaba recursos en cola) -- sbatch con --output explicito
# no depende de que el proceso que lo somete siga vivo.
set -eo pipefail
# ARC-153 aplica a "module load" (rompe con -u), no a set -e: sin esto,
# la corrida 6527 crasheo con TypeError a mitad de los 6 kernels y el
# job igual reporto exit 0 (echo final corrio de todas formas) -- el
# fallo paso desapercibido hasta revisar el .out a mano.
source ~/hyperion-venv/bin/activate
python3 /home/latorresn/hyperion/scripts/pacca/measure_gpu_rajaperf_oi.py
echo OI_MEASUREMENT_DONE
