#!/bin/bash
#SBATCH --job-name=hyp_gap_triage_pacca01
#SBATCH --partition=normal
#SBATCH --nodelist=pacca01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gap_triage_pacca01_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gap_triage_pacca01_%j.err
#
# Triage EXPLORATORIO de GAP Benchmark Suite (bfs, pr) en pacca01, mientras
# paccaA100 esta ocupado. Ver scripts/pacca/screen_gap_alpha_pacca01.sh --
# NINGUN numero de aqui es citable sin remedirse en paccaA100.
#
# A PROPOSITO EN pacca01, no paccaA100: es exactamente el caso que
# Estrategia_CPU_Fase2.md documenta como aceptable -- triage cualitativo
# (¿el tiempo responde al reloj EN ABSOLUTO?), nunca un numero final.
set -eo pipefail
# ARC-153: sin "-u" (rompe module load). set -e SI (leccion del job 6527).

bash ~/hyperion/scripts/pacca/build_gap_benchmark.sh
bash ~/hyperion/scripts/pacca/screen_gap_alpha_pacca01.sh
echo GAP_TRIAGE_SCRIPT_DONE
