#!/bin/bash
# Compila hpccg_cpu (CG sobre Laplaciano 3D de 7 puntos,
# kernels/hpccg/hpccg_cg_cpu_bench.c): familia NUEVA, no dual_*, no
# lbm_cpu. Ver Seguimiento_Cambios_Plan_Director.md (benchmarks del paper
# Littman/Deakin SC25, "HPCCG (MiniFE)"). Implementacion propia del metodo
# CG estandar, no vendorizada de Mantevo/HPCCG.
#
# ARC-126: `set -e -o pipefail` SIN `-u` -- Lmod referencia variables no
# definidas (LD_PRELOAD) y `-u` aborta el script al cargar cualquier modulo.
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
DEST="${DEST:-$HOME/hyperion-kernels/libexec/dual}"

module load gnu12/12.4.0

mkdir -p "$DEST"
cd "$REPO"

echo "== hpccg_cpu (CG sobre Laplaciano 3D, OpenMP) =="
gcc -O3 -march=native -fopenmp \
    kernels/hpccg/hpccg_cg_cpu_bench.c \
    -o "$DEST/hpccg_cpu" \
    -lm

echo
echo "== binario en $DEST =="
ls -la "$DEST/hpccg_cpu"
echo
echo "== checksum del binario real (NO es el que va en el catalogo -- ese es el del wrapper bin/hpccg_cpu) =="
sha256sum "$DEST/hpccg_cpu"
echo
echo BUILD_HPCCG_DONE
