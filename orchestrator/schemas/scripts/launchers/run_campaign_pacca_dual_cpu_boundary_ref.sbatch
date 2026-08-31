#!/bin/bash
#SBATCH --job-name=hyp_boundary_cpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/boundary_cpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/boundary_cpu_%j.err
#
# REF-only, 13 config_id x 6 repeticiones. No enviar antes de publicar la
# evaluacion de 6763/6764: esta es una extension posterior, no datos para
# ajustar retrospectivamente el resultado confirmatorio congelado.
set -e -o pipefail
export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"
~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
sudo -n nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" -rgc
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_boundary_ref_20260830.yaml \
  --node-id pacca-a100 --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 --campaign-timeout-seconds 5400
'
