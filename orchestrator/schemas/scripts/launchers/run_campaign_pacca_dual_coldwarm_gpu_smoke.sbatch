#!/bin/bash
#SBATCH --job-name=hyp_dual_cw_gpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_cw_gpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_cw_gpu_%j.err
set -e -o pipefail
export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_coldwarm_gpu_smoke.yaml \
  --node-id pacca-a100 --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_gpu_N512 \
  --campaign-timeout-seconds 6600
'
echo CAMPAIGN_SCRIPT_DONE
