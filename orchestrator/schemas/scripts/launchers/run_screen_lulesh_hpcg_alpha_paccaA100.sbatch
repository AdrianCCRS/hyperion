#!/bin/bash
#SBATCH --job-name=hyp_lulesh_hpcg_screen
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/lulesh_hpcg_screen_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/lulesh_hpcg_screen_%j.err
#
# Tamizaje de alpha para LULESH y HPCG, directo en paccaA100. Segundo y
# tercer candidato del pivote de catalogo motivado por C8 (tras el
# negativo de GAP, job 6601). Ver
# scripts/pacca/screen_lulesh_hpcg_alpha_paccaA100.sh.
#
# --exclusive: RAPL es de ambito de socket.
# --mem=8G: LULESH -s120 (~166 MB) y HPCG --nx=80 (~166 MB de matriz
#   dispersa) son modestos frente a DefMemPerNode=UNLIMITED, pero
#   explicito de todas formas para no bloquear backfill.
#
# ENVUELTO EN with_cpu_turbo_disabled.sh, obligatorio.
set -eo pipefail
# ARC-153/126: sin "-u".

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
bash ~/hyperion/scripts/pacca/screen_lulesh_hpcg_alpha_paccaA100.sh
'
echo LULESH_HPCG_SCREEN_SCRIPT_DONE
