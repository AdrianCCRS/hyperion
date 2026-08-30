#!/bin/bash
#SBATCH --job-name=hyp_big_ref_gpu
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/big_ref_gpu_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/big_ref_gpu_%j.err
#
# Campana suplementaria "big" REF-only -- eje GPU. Par exacto de
# run_campaign_pacca_dual_cpu_big_ref.sbatch: juntas producen las 54 corridas
# que cierran la comparacion CPU-REF vs GPU-REF para los 9 tamanos nuevos y
# el mapa de amortizacion K_break_even (cada corrida entrega cold Y warm).
#
# 9 config_id x 1 nivel CPU (REF) x 1 nivel GPU (REF) x 3 rep = 27 corridas.
#
# PRESUPUESTO DE TIEMPO (modelo GPU de dos terminos del pase 6689, NO los
# expected_runtime_seconds heredados que estaban inflados 17x y que se
# corrigieron en el catalogo el 2026-08-30):
#   gemm     N8192/12288/16384 =  2.0 +  6.8 + 16.0 = 24.8 s
#   fft      N8192/12288/16384 =  1.5 +  3.0 +  5.4 =  9.9 s
#   cholesky N8192/12288/16384 =  3.6 + 12.2 + 28.9 = 44.7 s
#   ---------------------------------------------------------
#   79.4 s por barrido x 3 repeticiones = ~4 min de computo medido
#   + ~35 s de overhead extremo-a-extremo x 27 corridas = ~16 min
#   (el doble asentamiento CPU+GPU domina el costo, no el kernel)
#   ---------------------------------------------------------
#   total esperado ~20 min. --time=2h da un margen enorme a proposito: es la
#   PRIMERA vez que estos tamanos corren en GPU, y el modelo de dos terminos
#   se valido sobre N<=4096 -- extrapolarlo a N=16384 es razonable (el modelo
#   es analitico en fn(N), no una tabla) pero no esta medido. Si los tiempos
#   reales salen muy por encima, este job lo revelara sin costar una sesion
#   larga, y ESE es el smoke real que faltaba.
#
#   --campaign-timeout-seconds=5400 (1.5 h) deja ~30 min de margen sobre el
#   --time de Slurm para el `finally:` que restaura frecuencia.
#
# Relanzar este mismo script reanuda solo lo pendiente (CAM-11).
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_big_ref.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_gpu_N512 \
  --campaign-timeout-seconds 5400
'
echo CAMPAIGN_SCRIPT_DONE
