#!/bin/bash
# Construye CHOLMOD (SuiteSparse, Tim Davis) -- 4to candidato del pivote
# de catalogo motivado por C8 (Estrategia_CPU_Fase2.md §7.bis/§7.ter),
# tras GAP/LULESH/HPCG. Motivo distinto a los tres anteriores: la
# factorizacion supernodal de CHOLMOD agrupa columnas en "supernodos" y
# los factoriza con BLAS3 DENSO (LAPACK), mientras el resto (ensamblaje,
# permutacion) sigue siendo disperso -- el mismo patron denso-embebido-
# en-disperso que hace que `rajaperf_polybench_3mm_omp` sea el unico
# kernel del catalogo con mezcla real a frecuencia nativa (§7.ter).
#
# SOLO CHOLMOD y sus dependencias directas (SuiteSparse_config, AMD,
# CAMD, COLAMD, CCOLAMD) via SUITESPARSE_ENABLE_PROJECTS -- no el resto
# de SuiteSparse (GraphBLAS, UMFPACK, SPEX, que no hacen falta y traen
# dependencias extra como GMP/MPFR).
#
# CORRE EN paccaA100, NUNCA en pacca01/pacca-normal.
#
# BLAS/LAPACK: openblas/0.3.21, ya cargado como modulo por otras
# campañas de este proyecto -- documentacion oficial de CHOLMOD
# recomienda MKL sobre OpenBLAS ("puede degradar rendimiento en casos
# raros"), pero para tamizaje exploratorio no es bloqueante.
#
# Uso: bash build_cholmod.sh [dir_salida]
set -eo pipefail
# ARC-126: sin "-u" a proposito.

output_root="${1:-/home/latorresn/hyperion-kernels}"
src_dir="$output_root/src/suitesparse"

# cmake en paccaA100 necesita libjsoncpp.so.19 (misma leccion de
# build_rajaperf_cuda.sh / build_lulesh.sh).
export LD_LIBRARY_PATH="$HOME/yacacerest/libs_pacca01:${LD_LIBRARY_PATH:-}"
module load gnu12/12.4.0 cmake/4.3.4 openblas/0.3.21 2>&1 || true
export CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++
export CC=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc
export FC=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gfortran

if [[ ! -d "$src_dir" ]]; then
  git clone --quiet --branch stable --depth 1 \
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git "$src_dir"
fi

actual_commit="$(git -C "$src_dir" rev-parse HEAD)"

build_dir="$src_dir/build"
rm -rf "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"
cmake -DCMAKE_INSTALL_PREFIX="$src_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DSUITESPARSE_ENABLE_PROJECTS=cholmod \
      -DSUITESPARSE_DEMOS=ON \
      -DSUITESPARSE_USE_FORTRAN=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DBLA_VENDOR=OpenBLAS \
      ..
cmake --build . -j4
cmake --install .

mkdir -p "$output_root/libexec/cholmod"
bin_found="$(find "$build_dir" "$src_dir" -name 'cholmod_dl_demo' -type f -executable 2>/dev/null | head -1)"
if [[ -z "$bin_found" ]]; then
  echo "ERROR: no se encontro cholmod_dl_demo tras el build" >&2
  exit 1
fi
install -m 0755 "$bin_found" "$output_root/libexec/cholmod/cholmod_dl_demo"

# Las .so de CHOLMOD/AMD/COLAMD/SuiteSparse_config quedan en src/lib -- el
# wrapper del kernel necesita esa ruta en LD_LIBRARY_PATH para correr.
echo "commit real: $actual_commit"
echo "binario en: $output_root/libexec/cholmod/cholmod_dl_demo"
echo "libs en: $src_dir/lib"
