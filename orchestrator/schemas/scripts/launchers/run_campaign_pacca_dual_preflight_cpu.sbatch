#!/bin/bash
#SBATCH --job-name=hyp_dual_preflight_cpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_preflight_cpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_preflight_cpu_%j.err
#
# Pre-vuelo CPU-solo del selector CPU/GPU. Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_preflight_cpu.yaml
#
# 6 kernels x 4 niveles x 10 rep = 240 corridas.
set -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_preflight_cpu.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 \
  --campaign-timeout-seconds 9000
'
echo CAMPAIGN_SCRIPT_DONE
