#!/usr/bin/env bash
# Valida PLAN.md Fase 0. Requiere el nodo real con vtune/2023 cargable.
set -uo pipefail

fail() { echo "FALLO: $1" >&2; exit 1; }

module purge
module load vtune/2023 || fail "no se pudo cargar el modulo vtune/2023"

command -v vtune >/dev/null || fail "vtune no esta en PATH tras cargar el modulo"
vtune --version || fail "vtune --version fallo"

BIN="${1:-./bin/ep.C.x}"
[ -x "$BIN" ] || fail "binario de prueba no encontrado o sin permiso de ejecucion: $BIN"

TMPDIR=$(mktemp -d)

vtune -collect hotspots -knob sampling-mode=hw -r "$TMPDIR/hs" -- "$BIN" \
  || fail "Hotspots HW fallo - revisar si aparece 'cannot recognize the processor'"

vtune -collect hpc-performance -r "$TMPDIR/hpc" -- "$BIN" \
  || fail "HPC Performance Characterization fallo"

SUMMARY=$(vtune -report summary -r "$TMPDIR/hpc")
echo "$SUMMARY" > "$TMPDIR/summary.txt"

echo "$SUMMARY" | grep -qE "Memory Bound:\s*[0-9]" \
  || fail "Memory Bound no aparece con un valor numerico - ver context/04, puede requerir ajustar el nombre de campo"

echo "$SUMMARY" | grep -qE "DP GFLOPS:\s*[0-9]" \
  || fail "DP GFLOPS no aparece con un valor numerico"

echo "PASO: modulo carga, EBS funcional, Memory Bound y DP GFLOPS poblados."
echo "Reporte completo guardado en: $TMPDIR/summary.txt (revisar nombres de campo reales)"
exit 0
