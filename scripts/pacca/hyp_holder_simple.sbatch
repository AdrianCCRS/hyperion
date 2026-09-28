#!/bin/bash
#SBATCH --job-name=hyp_holder_cfg
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/holder_cfg_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/holder_cfg_%j.err
#
# Holder minimo: toma paccaA100 en exclusiva y no hace nada, para conservar
# el puesto en la cola (FIFO puro) mientras se hacen configuraciones a mano.
# No toca turbo ni GPU. SIN --time a proposito (control manual).
#
# Adjuntar shell:  srun --jobid=<id> --overlap --pty bash
# Soltar el nodo:  touch ~/hyperion-results/HOLDER_CFG_STOP   (o scancel <id>)

STOP_FILE="$HOME/hyperion-results/HOLDER_CFG_STOP"
rm -f "$STOP_FILE"
echo "[$(date -Is)] holder activo en $(hostname), job $SLURM_JOB_ID"
while [[ ! -f "$STOP_FILE" ]]; do sleep 30; done
echo "[$(date -Is)] STOP recibido, liberando"
