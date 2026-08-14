#!/bin/bash
# Compila STREAM (kernels/stream/stream.c) y el probe de FLOPs pico
# (kernels/ert/ert_probe.c, ver ARC-31 para por que no es ERT completo).
# Correr desde dentro de un checkout de ~/hyperion actualizado.
set -eo pipefail
# ARC-126: sin "-u" a proposito -- ver build_npb.sh para el detalle. Lmod
# rompe "module load" en silencio bajo "set -u" (LD_PRELOAD sin definir
# en su init); aqui no se notaba porque gcc del sistema ya alcanzaba para
# compilar C, pero es el mismo bug de fondo.

# ARC-125: "gnu14" no existe en este cluster -- el modulo real es
# "gnu12/12.4.0" (confirmado con `module avail gnu`).
module load gnu12 2>&1 || true

ROOT=~/hyperion-kernels
BIN="$ROOT/bin"
mkdir -p "$BIN"

cd ~/hyperion
# ARC-125: -march=native agregado despues de que Intel Advisor confirmara
# (Vector ISA: SSE2, ancho de vector 2/4) que sin esta flag GCC nunca
# generaba AVX-512 -- ert_probe media un P_pico ~8x por debajo del real
# (71.8 GFLOP/s medido vs ~577 GFLOP/s que un DGEMM real con AVX-512 sí
# alcanza en los mismos 6 nucleos). stream_c no mostraba el mismo sesgo
# (limitado por ancho de banda de DRAM, no por ejecucion vectorial), pero
# se corrige igual por consistencia -- el harness de telemetria
# (telemetry/CMakeLists.txt) ya usa -march=native, esta era la
# inconsistencia real entre el instrumento y los kernels que mide.
# stream_c: solo -march=native. Se probo agregar tambien
# -mprefer-vector-width=512 (igual que ert_probe abajo) pero midio Triad
# mas bajo y con mucha mas varianza entre corridas (49.3-69.2 GB/s vs
# 58.8-59.5 GB/s estable sin esa flag) -- consistente con el downclocking
# de reloj que AVX-512 puede causar, sin ningun beneficio para un kernel
# limitado por ancho de banda de DRAM, no por ejecucion vectorial. No se
# aplica aqui a proposito.
gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=64000000 -o "$BIN/stream_c" kernels/stream/stream.c
echo "stream_c compilado"

# ert_probe: -mprefer-vector-width=512 SI se aplica -- verificado con
# Advisor que -march=native solo dejaba a GCC eligiendo AVX2 (ancho 4) en
# vez de AVX-512 (ancho 8), heuristica por defecto de GCC en objetivos de
# servidor para evitar el mismo throttling de arriba. Aqui SI conviene
# pedirlo explicito: ert_probe mide P_pico de computo, que sí se beneficia
# del ancho de vector maximo (417.8 GFLOP/s con la flag vs 308.5 sin ella,
# a 6 nucleos).
gcc -O3 -march=native -mprefer-vector-width=512 -fopenmp -o "$BIN/ert_probe" kernels/ert/ert_probe.c -lm
echo "ert_probe compilado"

cd "$BIN"
sha256sum stream_c ert_probe | tee -a "$ROOT/checksums.sha256"

echo "=== smoke ==="
echo "-- stream_c --"
OMP_NUM_THREADS=8 ./stream_c | tail -15
echo "-- ert_probe --"
OMP_NUM_THREADS=8 ./ert_probe

echo "DONE"
