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
set -euo pipefail

CLASSES=("${@:-B}")

module load gnu14 2>&1 || true
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

for cls in "${CLASSES[@]}"; do
  for kernel in ep mg cg is ft lu; do
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
  for k in ep mg cg is ft lu; do
    f="${k}.${cls}.x"
    if [ -f "$f" ]; then
      echo "=== $f ==="
      OMP_NUM_THREADS=8 /usr/bin/time -f "WALLTIME_SECONDS %e" ./"$f" 2>&1 | grep -E "Mop/s total|Time in seconds|WALLTIME_SECONDS|Verification"
    fi
  done
done

echo "DONE"
