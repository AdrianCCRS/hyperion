#!/bin/bash
#SBATCH --job-name=hyp_big_ref_cpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/big_ref_cpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/big_ref_cpu_%j.err
#
# Campana suplementaria "big" REF-only -- eje CPU. Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_big_ref.yaml para la
# justificacion metodologica completa (verificacion 136/136 de que la etiqueta
# de dispositivo con solo REF coincide con la del barrido completo de 40
# acciones, y el caveat de que esa verificacion se hizo sobre N<=4096).
#
# 9 config_id x 1 nivel (REF) x 3 rep = 27 corridas.
#
# PRESUPUESTO DE TIEMPO (calculado, no heredado):
#   suma expected_runtime_seconds de los 9 kernels CPU = 1366 s por barrido
#   x 3 repeticiones                                   = 4098 s (~68 min)
#   + ~30 s de overhead extremo-a-extremo x 27 corridas = ~14 min
#   ---------------------------------------------------------------
#   total esperado ~1.4 h. --time=3h da mas del doble de margen. A REF NO
#   hay ralentizacion por DVFS (no se fija frecuencia), asi que el factor de
#   _frequency_slowdown_ratio() no aplica aqui -- esa funcion dimensiona
#   timeouts de kernels CPU-bound bajo reloj fijo, no este caso.
#
#   --campaign-timeout-seconds=9000 (2.5 h) deja ~30 min de margen sobre el
#   --time de Slurm para que el `finally:` de run_campaign restaure la
#   frecuencia ANTES del kill duro. El proceso siempre debe salir por su
#   cuenta, nunca por SIGKILL de Slurm.
#
# Cabe holgadamente en UNA sesion. Si aun asi se cortara, relanzar este mismo
# script reanuda solo lo pendiente (CAM-11 detecta las corridas aceptadas en
# el output_dir y las salta).
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_big_ref.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 \
  --campaign-timeout-seconds 9000
'
echo CAMPAIGN_SCRIPT_DONE
