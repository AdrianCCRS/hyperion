#!/bin/bash
# Construye LULESH (LLNL, Livermore Unstructured Lagrangian Explicit
# Shock Hydrodynamics) -- candidato del pivote de catalogo motivado por
# C8 (Estrategia_CPU_Fase2.md §7.bis / §6.septies): fases fisicas
# explicitas por timestep (calculo de tensiones, compute-bound,
# alternado con actualizacion de malla no estructurada, memoria
# dependiente del dato) POR DISEÑO del algoritmo, no por artefacto de
# tamaño -- el candidato mas directo a mezcla real de fase tras GAP
# (que salio negativo en alpha, job 6601).
#
# CORRE EN paccaA100, NUNCA en pacca01/pacca-normal (misma leccion de
# build_rajaperf_cuda.sh: paquetes de sistema divergentes entre nodos).
#
# BUILD: CMake, OpenMP habilitado, MPI deshabilitado (un solo proceso,
# los 6 nucleos delegados via OMP_NUM_THREADS, igual que el resto del
# catalogo CPU).
#
# Uso: bash build_lulesh.sh [dir_salida]
set -eo pipefail
# ARC-126: sin "-u" a proposito (rompe module load / Lmod).

output_root="${1:-/home/latorresn/hyperion-kernels}"
src_dir="$output_root/src/lulesh"

# cmake en paccaA100 necesita libjsoncpp.so.19, presente en pacca01 pero
# no en paccaA100 (paquetes de sistema divergentes -- misma lección que
# build_rajaperf_cuda.sh). Se usa la copia ya cacheada.
export LD_LIBRARY_PATH="$HOME/yacacerest/libs_pacca01:${LD_LIBRARY_PATH:-}"
module load gnu12/12.4.0 cmake/4.3.4 2>&1 || true
export CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++
export CC=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc

if [[ ! -d "$src_dir" ]]; then
  git clone --quiet https://github.com/LLNL/LULESH.git "$src_dir"
fi

# SIN commit pineado a proposito, igual que build_gap_benchmark.sh: es
# tamizaje exploratorio, no un kernel de catalogo final todavia. El
# commit real se imprime abajo -- si algo de aqui se cita, registrar
# ESE commit entonces.
actual_commit="$(git -C "$src_dir" rev-parse HEAD)"

build_dir="$src_dir/build"
rm -rf "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5: el CMakeLists.txt de LULESH declara
# un cmake_minimum_required viejo, incompatible por defecto con cmake
# 4.3.4 del cluster.
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=Off -DWITH_OPENMP=On \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j4

mkdir -p "$output_root/libexec/lulesh"
bin_found=""
for candidate in "$build_dir/lulesh2.0" "$build_dir"/lulesh*; do
  if [[ -x "$candidate" && -f "$candidate" ]]; then
    bin_found="$candidate"
    break
  fi
done
if [[ -z "$bin_found" ]]; then
  echo "ERROR: no se encontro el binario lulesh2.0 tras el build" >&2
  exit 1
fi
install -m 0755 "$bin_found" "$output_root/libexec/lulesh/lulesh2.0"

echo "commit real: $actual_commit"
echo "binario en: $output_root/libexec/lulesh/lulesh2.0"
"$output_root/libexec/lulesh/lulesh2.0" -h 2>&1 | head -20 || true
