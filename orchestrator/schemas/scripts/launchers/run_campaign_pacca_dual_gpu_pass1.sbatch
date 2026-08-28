#!/bin/bash
#SBATCH --job-name=hyp_dual_gpu_pass1
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_gpu_pass1_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_gpu_pass1_%j.err
#
# Pase 1 del eje GPU. Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_pass1.yaml
#
# 68 config_id x 1 nivel CPU (REF) x 4 niveles GPU x 3 rep = 816 corridas
# (1088 lanzamientos de proceso). A los ~18 s/lanzamiento medidos en el
# pre-vuelo 6681 son ~5.4 h; --time=06:00:00 y el timeout de campana a
# 5h20m dejan margen para que el `finally:` de run_campaign restaure la
# frecuencia por su cuenta en vez de morir por el kill de Slurm.
#
# Si la ventana de nodo se acaba antes, relanzar este mismo script reanuda
# (CAM-11 salta las corridas ya aceptadas en el output_dir).
set -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_pass1.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_gpu_N512 \
  --campaign-timeout-seconds 19200
'
echo CAMPAIGN_SCRIPT_DONE
