#!/bin/bash
# Compila NPB3.4-OMP (ep, mg, cg, is, ft, lu) para las clases pedidas.
#
# Uso: ./build_npb.sh [CLASE ...]     (por defecto: B, la clase de dataset
#                                       real elegida en ARC-32)
#   ./build_npb.sh S            -> solo smoke test
#   ./build_npb.sh S W A B      -> las cuatro, para repetir la medicion de
#                                   tiempos real que decidio la clase (F3.3)
#
# Requiere ~/hyperion-kernels/src/NPB3.4.4.tar.gz ya presente (procedencia
# y sha256 documentados en orchestrator/schemas/kernels/catalog.yaml).
set -eo pipefail
# ARC-126: sin "-u" a proposito. El init de Lmod en este cluster
# (/opt/ohpc/admin/lmod/lmod/init/bash) referencia LD_PRELOAD sin definir;
# bajo "set -u" eso hace que "module load" falle en silencio (el "|| true"
# de abajo se traga el error) y gcc/gfortran quedan resueltos al binario
# de sistema (8.5.0, sin gfortran) en vez del modulo gnu12. Mismo bug ya
# diagnosticado con los scripts de Advisor.
module load gnu12 2>&1 || true
gcc --version | head -1
gfortran --version | head -1

ROOT=~/hyperion-kernels
SRC="$ROOT/src"
BIN="$ROOT/bin"
mkdir -p "$SRC" "$BIN"

cd "$SRC"
if [ ! -d NPB3.4.4 ]; then
  tar xzf NPB3.4.4.tar.gz
fi
cd NPB3.4.4/NPB3.4-OMP
cp -f config/make.def.template config/make.def
cp -f config/suite.def.template config/suite.def
# ARC-125: el template de NPB trae FFLAGS/CFLAGS = "-O3 -fopenmp", sin
# -march=native -- mismo problema encontrado y corregido en ert_probe/
# stream_c (confirmado con Advisor). No sesga operational_intensity (el
# contador de hardware mide los FLOPs realmente ejecutados, sea cual sea
# el ancho de vector), pero los kernels corrían más lento de lo real.
# Deliberadamente SIN -mprefer-vector-width=512 aqui: el catalogo NPB
# mezcla kernels memory-bound (mg, cg) y compute-bound (bt) -- forzar
# ancho 512 en todos arriesga el mismo downclocking que ya penalizo a
# stream_c (memory-bound) sin beneficio, sin poder aplicarlo kernel por
# kernel dentro de un unico make.def compartido.
sed -i 's/^\(FFLAGS[[:space:]]*=[[:space:]]*\)-O3 -fopenmp/\1-O3 -march=native -fopenmp/' config/make.def
sed -i 's/^\(CFLAGS[[:space:]]*=[[:space:]]*\)-O3 -fopenmp/\1-O3 -march=native -fopenmp/' config/make.def
grep -E "^FFLAGS|^CFLAGS" config/make.def

# ARC-57: ep/is se retiraron del catalogo real (no hacen punto flotante
# real, ver Registro_Cambios) y se reemplazaron por bt/sp -- este script
# quedo desactualizado desde entonces, corregido aqui de paso.
for cls in "${CLASSES[@]}"; do
  for kernel in bt mg cg sp ft lu; do
    echo "--- building $kernel.$cls ---"
    make CLASS="$cls" "$kernel" 2>&1 | tail -10
  done
done

cp -f bin/*.x "$BIN"/

echo "=== checksums ==="
cd "$BIN"
sha256sum *.x | tee -a "$ROOT/checksums.sha256"

echo "=== timing ==="
for cls in "${CLASSES[@]}"; do
  for k in bt mg cg sp ft lu; do
    f="${k}.${cls}.x"
    if [ -f "$f" ]; then
      echo "=== $f ==="
      OMP_NUM_THREADS=8 /usr/bin/time -f "WALLTIME_SECONDS %e" ./"$f" 2>&1 | grep -E "Mop/s total|Time in seconds|WALLTIME_SECONDS|Verification"
    fi
  done
done

echo "DONE"
