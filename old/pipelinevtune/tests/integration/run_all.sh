#!/usr/bin/env bash
# Corre toda la suite de integracion en orden. Se detiene en el primer fallo.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  "00_test_module_and_smoke.sh"
  "01_test_preflight.sh"
  "02_test_baseline_ep.sh"
  "03_test_anchors.sh"
  "04_test_single_kernel_full.sh"
)

for s in "${SCRIPTS[@]}"; do
  echo "=== corriendo $s ==="
  bash "$SCRIPT_DIR/$s"
  if [ $? -ne 0 ]; then
    echo ""
    echo "DETENIDO en $s — corregir antes de continuar con la campana completa."
    exit 1
  fi
  echo ""
done

echo "TODAS LAS PRUEBAS DE INTEGRACION PASARON. Listo para lanzar por sbatch."
exit 0
