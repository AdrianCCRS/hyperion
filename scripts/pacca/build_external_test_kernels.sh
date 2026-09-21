#!/bin/bash
# Compila XSBench y RSBench (proxy apps de Argonne, OpenMP) para la prueba externa sellada del
# clasificador CPU. Ejecutar EN paccaA100 (nodo donde se mide): un binario compilado en otro nodo
# no es valido. Clona a commits fijos, compila con gcc y deja los binarios en
# ~/hyperion-kernels/libexec/{xsbench,rsbench}. No toca binarios ni checksums existentes.
#
# `set -e -o pipefail` SIN `-u`: Lmod referencia variables no definidas (ver build_hpccg_kernel.sh).
set -e -o pipefail

XS_COMMIT="${XS_COMMIT:-ba08e5221af6106252b866e50ea123c69d31a4e2}"
RS_COMMIT="${RS_COMMIT:-34b644787ea9af4fb188e1253da72e09bbed9989}"
ROOT="${ROOT:-$HOME/hyperion-kernels}"
SRC="$ROOT/external_src"
module load gnu12/12.4.0

mkdir -p "$SRC" "$ROOT/libexec/xsbench" "$ROOT/libexec/rsbench"

fetch() {  # nombre url commit
  if [[ ! -d "$SRC/$1/.git" ]]; then git clone -q "$2" "$SRC/$1"; fi
  git -C "$SRC/$1" fetch -q origin
  git -C "$SRC/$1" checkout -q "$3"
  echo "$1 commit: $(git -C "$SRC/$1" rev-parse HEAD)"
}

fetch XSBench https://github.com/ANL-CESAR/XSBench.git "$XS_COMMIT"
fetch RSBench https://github.com/ANL-CESAR/RSBench.git "$RS_COMMIT"

echo "== precision aritmetica declarada en el codigo (el instrumento CPU solo cuenta FLOPs double) =="
for d in XSBench RSBench; do
  echo "-- $d: apariciones de double / float en openmp-threading"
  printf 'double: %s  float: %s\n' \
    "$(grep -hw 'double' "$SRC/$d"/openmp-threading/*.[ch] | wc -l)" \
    "$(grep -hw 'float' "$SRC/$d"/openmp-threading/*.[ch] | wc -l)"
done

build() {  # nombre binario_salida
  make -C "$SRC/$1/openmp-threading" clean >/dev/null 2>&1 || true
  make -C "$SRC/$1/openmp-threading" -j 8
  exe="$(find "$SRC/$1/openmp-threading" -maxdepth 1 -type f -perm -u+x ! -name '*.o' ! -name Makefile | head -1)"
  cp "$exe" "$2"
  echo "$1 binario: $2"
  sha256sum "$2"
}

build XSBench "$ROOT/libexec/xsbench/XSBench"
build RSBench "$ROOT/libexec/rsbench/rsbench"

echo BUILD_EXTERNAL_DONE
