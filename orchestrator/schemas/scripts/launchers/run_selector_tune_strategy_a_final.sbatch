#!/bin/bash
#SBATCH --job-name=hyp_tune_a_final
#SBATCH --partition=normal
#SBATCH --nodelist=pacca01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/tune_strategy_a_final_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/tune_strategy_a_final_%j.err
#
# Optuna multiobjetivo + validacion anidada leave-one-operation-out sobre
# la Estrategia A del dataset dual FINAL (68/68 config_id, CPU+GPU
# completos, build 20260830). label_health.json confirma
# verdict=comparison_valid para A sobre este dataset (margen mediano
# 11.31%, 5 acciones con masa >=5%) -- primera vez que la comparacion de
# familias es interpretable sobre el catalogo completo, no solo el eje de
# CPU aislado.
#
# PARTICION normal/pacca01 a proposito, no paccaA100: es analisis puro
# (pandas + sklearn + optuna sobre CSVs ya en disco), no necesita GPU.
# --mem=32G explicito: DefMemPerNode=UNLIMITED bloquea backfill sin esto.
set -eo pipefail

source ~/hyperion-venv/bin/activate
cd ~/hyperion

python3 -m classifier.selector tune \
  --dataset ~/hyperion-results/analysis/selector_final_20260830/strategy_a_candidates.csv \
  --output-dir ~/hyperion-results/analysis/selector_final_20260830/tune_strategy_a \
  --trials 100 \
  --seed 20260828

echo TUNE_STRATEGY_A_DONE
