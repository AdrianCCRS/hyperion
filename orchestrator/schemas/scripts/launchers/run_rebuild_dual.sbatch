#!/bin/bash
#SBATCH --job-name=hyp_dual_rebuild
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_rebuild_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_rebuild_%j.err
#
# Recompila los 12 binarios duales y resincroniza la cadena de checksums
# (binario -> wrapper -> catalogo). Ver scripts/pacca/rebuild_and_resync_dual.sh
#
# CORRE EN paccaA100 A PROPOSITO: pacca01 y paccaA100 tienen CPUs distintos
# (Gold 5320 vs 5315Y) y paquetes de sistema distintos -- compilar en uno
# para correr en el otro ya rompio antes. No es --exclusive porque solo
# compila; no mide nada, asi que no necesita el nodo limpio.
#
# Las campanas se encolan con --dependency=afterok sobre este job: si la
# recompilacion o la verificacion de checksums falla, NINGUNA campana
# arranca, que es el modo de fallo correcto.
set -o pipefail

bash ~/hyperion/scripts/pacca/rebuild_and_resync_dual.sh
echo REBUILD_SCRIPT_DONE
