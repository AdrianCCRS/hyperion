#!/bin/bash
# Construye GAP Benchmark Suite (Beamer et al., arXiv:1508.03619) para
# triage EXPLORATORIO en pacca01 -- ver la nota de la Estrategia CPU
# sobre por que pacca01 sirve para triage cualitativo pero no para
# numeros finales (L3/nucleo identica a paccaA100 -- 1.50 MB -- pero
# modelo de potencia y contencion de memoria distintos).
#
# POR QUE GAP. Seis kernels de grafos (BFS, PageRank, componentes
# conexas, camino mas corto, betweenness centrality, triangulos) con
# acceso IRREGULAR y dependiente del dato -- el hueco que ni STREAM
# (ancho de banda, acceso regular) ni ptrchase (latencia pura, sin
# estructura de algoritmo real) cubren. Suite academica estandar, no un
# experimento aislado nuestro.
#
# BUILD LIGERO A PROPOSITO: esto es triage, no un kernel de catalogo
# final. Sin checksum pineado del binario (a diferencia de
# build_rajaperf_cuda.sh) porque nada de lo que salga de aqui entra al
# catalogo sin remedirse en paccaA100 primero -- ahi si se pinearia.
#
# Uso: bash build_gap_benchmark.sh [dir_salida]
set -euo pipefail

output_root="${1:-/home/latorresn/hyperion-kernels}"
src_dir="$output_root/src/gapbs"

module load gnu12/12.4.0 2>&1 || true
export CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++

if [[ ! -d "$src_dir" ]]; then
  git clone --quiet https://github.com/sbeamer/gapbs.git "$src_dir"
fi

# SIN commit pineado a proposito: es triage exploratorio, no un kernel de
# catalogo (build_rajaperf_cuda.sh si pinea, porque ese SI entra al
# catalogo). El commit real clonado se imprime abajo -- si algun resultado
# de aqui se termina citando, registrar ESE commit entonces, no antes.
actual_commit="$(git -C "$src_dir" rev-parse HEAD)"

cd "$src_dir"
make clean >/dev/null 2>&1 || true
make -j4

mkdir -p "$output_root/libexec/gapbs"
for bin in bfs pr cc sssp bc tc; do
  if [[ -x "$src_dir/$bin" ]]; then
    install -m 0755 "$src_dir/$bin" "$output_root/libexec/gapbs/$bin"
  else
    echo "AVISO: $bin no se construyo" >&2
  fi
done

echo "commit real: $actual_commit"
echo "binarios en: $output_root/libexec/gapbs/"
ls -la "$output_root/libexec/gapbs/"
