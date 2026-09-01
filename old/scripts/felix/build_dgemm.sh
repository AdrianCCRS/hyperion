#!/bin/bash
# Compila kernels/dgemm/dgemm_bench.c contra OpenBLAS del sistema
# (confirmado disponible en felix sin modulo, ver ARC-38) y corre un smoke
# test a varios tamanos de matriz.
set -euo pipefail

module load gnu14 2>&1 || true
gcc --version | head -1

FLAGS=$(pkg-config --cflags openblas)
LIBS=$(pkg-config --libs openblas)
echo "FLAGS=$FLAGS LIBS=$LIBS"

ROOT=~/hyperion-kernels
BIN="$ROOT/bin"
mkdir -p "$BIN"

gcc -O3 -Wall -Wextra $FLAGS -o "$BIN/dgemm_bench" ~/hyperion/kernels/dgemm/dgemm_bench.c $LIBS -lm
echo "dgemm_bench compilado"

sha256sum "$BIN/dgemm_bench" | tee -a "$ROOT/checksums.sha256"

echo "=== smoke: tamanos pequenos ==="
for n in 512 1024 2048; do
  echo "--- N=$n ---"
  OPENBLAS_NUM_THREADS=6 "$BIN/dgemm_bench" --size "$n" --iterations 3
done
