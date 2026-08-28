#!/bin/bash
#SBATCH --job-name=hyp_dual_cpu_full
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_cpu_full_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_cpu_full_%j.err
#
# Campana real del selector CPU/GPU -- eje CPU. Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_full.yaml
#
# 68 config_id x 8 niveles x 3 rep = 1632 corridas. Overhead real medido en
# el pre-vuelo (job 6680, 2026-08-27): ~5.5 s/corrida extremo-a-extremo ->
# proyeccion ~2.5 h, cabe en una sola sesion (margen generoso a 4h). Si por
# algo se corta antes de terminar, relanzar este mismo script reanuda solo
# (CAM-11 detecta las corridas ya aceptadas en el output_dir y las salta).
set -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_full.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 \
  --campaign-timeout-seconds 13200
'
echo CAMPAIGN_SCRIPT_DONE
