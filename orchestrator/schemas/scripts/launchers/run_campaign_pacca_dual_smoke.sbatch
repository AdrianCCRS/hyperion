#!/bin/bash
#SBATCH --job-name=hyp_dual_smoke
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_smoke_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_smoke_%j.err
#
# Campaña de humo del catalogo dual del selector CPU/GPU. Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_smoke.yaml
#
# 8 kernels (4 operaciones x {cpu,gpu}) x rejilla reducida REF/F0/F3/F6 en
# ambos ejes x 3 rep = 48 (cpu) + 192 (gpu, cartesiano cpu_freq x gpu_freq)
# = 240 corridas, todas cortas (<=90s c/u).
#
# Preambulo (module load, LD_LIBRARY_PATH, HYPERION_GPU_FREQ_WRITE_CAPABLE,
# `set -o pipefail` SIN -e) copiado de
# run_campaign_pacca_gpu_fine_grid_dataset.sbatch, ya validado en produccion
# real con gpu.enabled=true (job 6471+).
set -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
# Reset defensivo ANTES de que el orquestador capture su propio snapshot
# "original" -- ver scripts/pacca/reset_cpu_freq_range.sh: un job previo
# crasheado puede haber dejado cpu0-5/16-21 pinneados en un rango angosto,
# que de otro modo se propagaria en silencio como el "original" de ESTA
# corrida.
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_smoke.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 \
  --campaign-timeout-seconds 8400
'
echo CAMPAIGN_SCRIPT_DONE
