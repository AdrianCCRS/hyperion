#!/bin/bash
# Construye HPCG (High Performance Conjugate Gradient, Sandia/UTK) --
# candidato del pivote de catalogo motivado por C8
# (Estrategia_CPU_Fase2.md §7.bis / §6.septies): el ciclo multigrid
# alterna SpMV, suavizado y restriccion/prolongacion, cada etapa con
# intensidad distinta -- candidato razonable a mezcla de fase, un
# escalon por debajo de LULESH porque su alternancia es mas regular
# (mismo ciclo repetido) que multifasica.
#
# CORRE EN paccaA100, NUNCA en pacca01/pacca-normal.
#
# BUILD: sin MPI (-DHPCG_NO_MPI, un solo proceso), con OpenMP -- config
# `Make.GCC_OMP` oficial del repo, que ya trae -DHPCG_NO_MPI por
# defecto (MPdir/MPinc/MPlib vacios). Solo se ajusta CXX a la ruta del
# compilador del cluster.
#
# Uso: bash build_hpcg.sh [dir_salida]
set -euo pipefail

output_root="${1:-/home/latorresn/hyperion-kernels}"
src_dir="$output_root/src/hpcg"

module load gnu12/12.4.0 2>&1 || true
CXX_PATH=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++

if [[ ! -d "$src_dir" ]]; then
  git clone --quiet https://github.com/hpcg-benchmark/hpcg.git "$src_dir"
fi

actual_commit="$(git -C "$src_dir" rev-parse HEAD)"

cd "$src_dir"
cp setup/Make.GCC_OMP setup/Make.hyperion_omp
# CXX del cluster, y se quita -ftree-vectorizer-verbose (removido en
# gcc >= 9, rompe el build con gcc12).
sed -i \
  -e "s|^CXX .*= .*|CXX          = ${CXX_PATH}|" \
  -e "s| -ftree-vectorizer-verbose=0||" \
  setup/Make.hyperion_omp

rm -rf build_hyperion
mkdir -p build_hyperion
cd build_hyperion
../configure hyperion_omp
make -j4

mkdir -p "$output_root/libexec/hpcg"
bin_found=""
for candidate in bin/xhpcg; do
  if [[ -x "$candidate" ]]; then
    bin_found="$candidate"
    break
  fi
done
if [[ -z "$bin_found" ]]; then
  echo "ERROR: no se encontro bin/xhpcg tras el build" >&2
  exit 1
fi
install -m 0755 "$bin_found" "$output_root/libexec/hpcg/xhpcg"

echo "commit real: $actual_commit"
echo "binario en: $output_root/libexec/hpcg/xhpcg"
