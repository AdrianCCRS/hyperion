#!/bin/bash
#SBATCH --job-name=hyp_reprocess_rep_fix
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/reprocess_arc174_repfix_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/reprocess_arc174_repfix_%j.err
#
# Corrige un bug real de reprocess_frequency_quality_v2.py (2026-08-24):
# leia metadata.get("repetition", 1) en vez de metadata.get("repetition_index",
# 1) -- la clave "repetition" nunca existió en metadata.json (la real es
# "repetition_index", metadata_schema.RUN_METADATA_REQUIRED_KEYS), así que
# el .get() SIEMPRE caía al default 1. Las 540 corridas de
# pacca_cpu_final_attempt03_20260820_arc174 quedaron con la columna
# "repetition" de windows.csv constante en 1, sin importar la repetición
# real (1-10). No afecta al runner en vivo (campaign.py pasa
# repetition_index directo desde memoria, nunca releído de metadata.json),
# así que ninguna campaña en cola (6431, 6412, 6471-6477) estaba en riesgo
# -- ver tests/orchestrator/test_reprocess_frequency_quality_v2.py.
#
# PARTICION NORMAL, NO GPU. Este reprocesamiento es puro post-procesamiento
# Python (lee samples.csv/metadata.json, escribe windows.csv) -- no toca la
# GPU ni necesita paccaA100. Corre en la particion "normal" (pacca01/03/05/
# 07/09, confirmados idle 2026-08-24) para NO esperar a que 6470 libere el
# nodo GPU, que bloquea backfill por TimeLimit=UNLIMITED (ver memoria
# slurm-cola-pacca-unlimited-bloquea-backfill).
#
# SALIDA A UN DERIVED-DIR NUEVO, no sobrescribe el arc174 existente (mismo
# principio que la herramienta ya respeta: nunca pisa windows.csv original).
# Verificar el resultado antes de decidir si reemplaza al arc174 vigente.
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este clúster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

bash -c '
source ~/hyperion-venv/bin/activate
cd ~/hyperion
python3 -m orchestrator.schemas.tools.reprocess_frequency_quality_v2 \
  --manifest orchestrator/schemas/campaigns/campaign_pacca_dvfs.yaml \
  --campaign-dir /home/latorresn/hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820 \
  --derived-dir /home/latorresn/hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174_repfix \
  --workers 4
'

echo REPROCESS_REPFIX_DONE
