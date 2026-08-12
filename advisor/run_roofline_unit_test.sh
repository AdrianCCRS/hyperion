#!/bin/bash
# Prueba unitaria de Intel Advisor Roofline (2023.0.0) sobre UN kernel, en
# paccaA100. Ver docs/advisor/estudio_intel_advisor_roofline.md para la
# metodologia completa -- este script solo automatiza la secuencia de
# comandos ya documentada ahi (seccion 2), para un solo binario.
#
# Uso:
#   bash run_roofline_unit_test.sh <ruta_al_binario> [args...]
#
# Ejemplo real usado en la primera corrida:
#   bash run_roofline_unit_test.sh $HOME/vtune_selfcheck/NPB3.4-OMP/bin/ep.C.x
#
# Regla dura, igual que en raperezp/: no mata ni interfiere con nada de
# otros usuarios. Pensado para correr dentro de un srun corto (prueba
# unitaria), no un sbatch de campana completa -- eso se agrega despues si
# esta prueba confirma que vale la pena escalar.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Uso: $0 <ruta_al_binario> [args del binario...]" >&2
  exit 1
fi

BIN="$1"; shift
BIN_NAME="$(basename "$BIN")"
OUT_ROOT="${ADVISOR_OUT_ROOT:-$HOME/raperezp/manual_tests}"
PROJECT_DIR="$OUT_ROOT/${BIN_NAME}_roofline"

module purge
module load devtools/intel/oneapi/2023
module load advisor/2023.0.0

echo "=== advisor --version ==="
advisor --version

export OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close

mkdir -p "$OUT_ROOT"
rm -rf "$PROJECT_DIR"

echo "=== Survey (bajo overhead: tiempo por loop/funcion) ==="
taskset -c 0-5 advisor --collect=survey \
  --project-dir="$PROJECT_DIR" \
  -- "$BIN" "$@"

echo "=== Trip Counts + FLOP + simulacion de cache (alto overhead, corrida aparte) ==="
taskset -c 0-5 advisor --collect=tripcounts -flop --enable-cache-simulation \
  --project-dir="$PROJECT_DIR" \
  -- "$BIN" "$@"

echo "=== Reporte HTML autocontenido (se puede ver sin GUI) ==="
advisor --report=roofline \
  --report-output="$PROJECT_DIR/roofline_report.html" \
  --project-dir="$PROJECT_DIR"

echo "=== Snapshot portable para abrir con la GUI en un equipo local ==="
advisor --snapshot --project-dir="$PROJECT_DIR" --pack \
  --cache-sources --cache-binaries \
  -- "$PROJECT_DIR/${BIN_NAME}_snapshot"

echo "=== FIN ==="
echo "Project dir:     $PROJECT_DIR"
echo "Reporte HTML:    $PROJECT_DIR/roofline_report.html"
echo "Snapshot portable (.advixeexpz o carpeta, ver salida de arriba): $PROJECT_DIR/${BIN_NAME}_snapshot*"
