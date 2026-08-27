#!/bin/bash
#SBATCH --job-name=hyp_gap_screen_paccaA100
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/gap_screen_paccaA100_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/gap_screen_paccaA100_%j.err
#
# Tamizaje de alpha de GAP Benchmark (bfs, pr) DIRECTO en paccaA100 --
# el triage de pacca01 (job 6583) no dio señal util por falta de permiso
# de escritura de frecuencia en ese nodo. Ver
# scripts/pacca/screen_gap_alpha_paccaA100.sh.
#
# Primer paso del pivote de catalogo motivado por C8
# (Estrategia_CPU_Fase2.md §7.bis): el clasificador de fase por ventana
# funciona donde hay mezcla real -- el siguiente paso es ampliar el
# catalogo con kernels que la produzcan. GAP (bfs/pr) es el primer
# candidato: binarios ya compilados, cero costo de compilacion.
#
# --exclusive: RAPL es de ambito de socket.
# --mem=32G: grafo Kronecker 2^22 (~570 MB) + estructuras auxiliares,
#   explicito porque DefMemPerNode=UNLIMITED pediria el nodo entero.
#
# ENVUELTO EN with_cpu_turbo_disabled.sh, obligatorio: fija niveles
# concretos (F0-F4) y con turbo activo el reloj real no es el pedido.
set -eo pipefail
# ARC-153: sin "-u" (rompe module load).

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
bash ~/hyperion/scripts/pacca/screen_gap_alpha_paccaA100.sh
'
echo GAP_SCREEN_PACCAA100_SCRIPT_DONE
