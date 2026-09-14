#!/bin/bash
# Compila lbm_cpu (LBM D2Q9-BGK, kernels/lbm/lbm_d2q9_cpu_bench.c): familia
# NUEVA, no dual_*, agregada para densificar la zona cercana al ridge del
# catalogo CPU (ver Seguimiento_Cambios_Plan_Director.md, entrada sobre
# benchmarks del paper Littman/Deakin SC25). Implementacion propia del
# metodo LBM estandar, no vendorizada de UoB-HPC/advanced-hpc-lbm (ese repo
# no tiene licencia -- 404 en /license de GitHub, verificado 2026-09-14).
#
# ARC-126: `set -e -o pipefail` SIN `-u` -- Lmod referencia variables no
# definidas (LD_PRELOAD) y `-u` aborta el script al cargar cualquier modulo.
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
DEST="${DEST:-$HOME/hyperion-kernels/libexec/dual}"

module load gnu12/12.4.0

mkdir -p "$DEST"
cd "$REPO"

echo "== lbm_cpu (LBM D2Q9-BGK, OpenMP) =="
gcc -O3 -march=native -fopenmp \
    kernels/lbm/lbm_d2q9_cpu_bench.c \
    -o "$DEST/lbm_cpu" \
    -lm

echo
echo "== binario en $DEST =="
ls -la "$DEST/lbm_cpu"
echo
echo "== checksum (para binary_checksum del catalogo) =="
sha256sum "$DEST/lbm_cpu"
echo
echo BUILD_LBM_DONE
