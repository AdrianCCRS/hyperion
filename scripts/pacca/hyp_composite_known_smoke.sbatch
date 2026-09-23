#!/bin/bash
# Prueba de humo de la Aplicacion A (Bloque C, item C2, Plan_Fase3_Daemon.md
# SS0.1): corre fase3_daemon/composite_apps/composite_known.py de verdad en
# paccaA100 -- dgemm_n2048 (compute_bound) + npb_cg (memory_bound), ambos
# ya en el catalogo con checksum verificado, 2 ciclos. Verifica que las
# fronteras de fase reales queden registradas y que ambos kernels reporten
# exito segun su success_check. NUNCA local, ver memoria
# feedback-never-run-compute-locally -- son binarios HPC reales.
#SBATCH --job-name=hyp_composite_known_smoke
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/composite_known_smoke_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/composite_known_smoke_%j.err

set -euo pipefail
ROOT="$HOME/hyperion"
KERNELS_ROOT="$HOME/hyperion-kernels"
BOUNDARIES_OUT="$ROOT/fase3_daemon/composite_apps/composite_known_smoke_%j.jsonl"
BOUNDARIES_OUT="${BOUNDARIES_OUT/\%j/$SLURM_JOB_ID}"

# Mismo entorno que las campanas reales (hyp_external_cpu.sbatch) -- sin
# esto, python3 cae al Python 3.10.20 del sistema (sin hashlib.file_digest,
# ver el fix en common/hpc/catalog.py) y dgemm_bench no encuentra
# libopenblas.so.0 en tiempo de ejecucion (jobs 7568-7570, 2026-09-23).
module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate"

hostname
python3 --version
echo "=== tests unitarios (sin binarios reales) ==="
python3 -m pytest "$ROOT/fase3_daemon/composite_apps/tests/test_composite_known.py" -q

echo "=== smoke real: dgemm_n2048 + npb_cg, 2 ciclos, paccaA100 ==="
python3 "$ROOT/fase3_daemon/composite_apps/composite_known.py" \
  --node-id pacca-a100 \
  --kernels-root "$KERNELS_ROOT" \
  --boundaries-out "$BOUNDARIES_OUT" \
  --cycles 2

echo "=== registro de fronteras de fase ==="
cat "$BOUNDARIES_OUT"
n_lines=$(wc -l < "$BOUNDARIES_OUT")
if [ "$n_lines" -ne 4 ]; then
  echo "se esperaban 4 lineas (2 kernels x 2 ciclos), se obtuvieron $n_lines" >&2
  exit 1
fi
n_success=$(grep -o '"success": true' "$BOUNDARIES_OUT" | wc -l)
if [ "$n_success" -ne 4 ]; then
  echo "se esperaban 4 fases exitosas, se obtuvieron $n_success -- ver $BOUNDARIES_OUT" >&2
  exit 1
fi
echo "smoke de Aplicacion A: OK (4/4 fases exitosas, registro en $BOUNDARIES_OUT)"
