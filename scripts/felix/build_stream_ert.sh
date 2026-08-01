#!/bin/bash
# Compila STREAM (kernels/stream/stream.c) y el probe de FLOPs pico
# (kernels/ert/ert_probe.c, ver ARC-31 para por que no es ERT completo).
# Correr desde dentro de un checkout de ~/hyperion actualizado.
set -euo pipefail

module load gnu14 2>&1 || true

ROOT=~/hyperion-kernels
BIN="$ROOT/bin"
mkdir -p "$BIN"

cd ~/hyperion
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=64000000 -o "$BIN/stream_c" kernels/stream/stream.c
echo "stream_c compilado"

gcc -O3 -fopenmp -o "$BIN/ert_probe" kernels/ert/ert_probe.c -lm
echo "ert_probe compilado"

cd "$BIN"
sha256sum stream_c ert_probe | tee -a "$ROOT/checksums.sha256"

echo "=== smoke ==="
echo "-- stream_c --"
OMP_NUM_THREADS=8 ./stream_c | tail -15
echo "-- ert_probe --"
OMP_NUM_THREADS=8 ./ert_probe

echo "DONE"
