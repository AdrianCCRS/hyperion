#!/bin/bash
# Smoke real de la campana CUPTI (gpu_cupti_20260923.yaml): 1 kernel
# (rodinia_gaussian), REF/REF, 1 repeticion -- confirma de punta a punta
# que --cupti-activity-lib produce datos validos antes de encolar las 570
# corridas de la campana completa. NUNCA local.
#SBATCH --job-name=hyp_gpu_cupti_smoke
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_cupti_smoke_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_cupti_smoke_%j.err

set -uo pipefail
ROOT="$HOME/hyperion"

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate"

hostname
[[ -f "$ROOT/common/telemetry/build-cupti/libhyperion_cupti_activity.so" ]] \
  || { echo "FALTA la biblioteca CUPTI Activity" >&2; exit 2; }

echo "=== smoke: 1 kernel, REF/REF, 1 rep, gpu.activity_trace.enabled=true ==="
python3 "$ROOT/fase1_telemetria/run_campaign.py" run-campaign \
  --manifest "$ROOT/scripts/pacca/final_campaign/gpu_cupti_smoke_20260923.yaml" \
  --node-id pacca-a100 \
  --reference-kernel-ref npb_mg
STATUS=$?

CAMPAIGN_DIR="/home/latorresn/hyperion-results/final/campaigns/gpu_cupti_smoke_20260923"
if [ -f "$CAMPAIGN_DIR/campaign_metadata.json" ]; then
  echo "=== reporte ==="
  python3 "$ROOT/fase1_telemetria/run_campaign.py" report --campaign-dir "$CAMPAIGN_DIR" 2>&1 || true
else
  echo "=== sin campaign_metadata.json, run-campaign no llego a completar (status=$STATUS) ==="
fi

echo "run-campaign status=$STATUS"
exit "$STATUS"
