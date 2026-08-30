#!/bin/bash
#SBATCH --job-name=hyp_dual_gpu_full
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/dual_gpu_full_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/dual_gpu_full_%j.err
#
# Campana real del selector CPU/GPU -- eje GPU. Ver
# orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_full.yaml
#
# 68 config_id x 4 niveles CPU x 8 niveles GPU x 3 rep = 6528 corridas.
# Overhead real medido en el pre-vuelo GPU (job 6681, cancelado a 1h12m,
# 2026-08-27): ~30-36 s/corrida extremo-a-extremo (el doble asentamiento
# CPU+GPU domina el costo) -> proyeccion total ~48-55 h. NO cabe en una
# sola sesion sin bloquear el nodo compartido mas de un dia -- 2026-08-27
# se decidio partir en VARIAS SESIONES del mismo alcance (no se redujo la
# rejilla), usando la reanudacion automatica del orquestador (CAM-11):
# `_has_existing_campaign_artifacts(output_dir)` detecta las corridas ya
# aceptadas y las salta, sin flag especial.
#
# COMO RELANZAR: sbatch este mismo script tantas veces como haga falta
# (una por sesion). Cada sesion:
#   1. corre reset_cpu_freq_range.sh antes de que el orquestador tome su
#      propio snapshot (evita la corrupcion de frecuencia entre sesiones,
#      ver docs/general/metodologia_selector_cpu_gpu_20260827.md #6.2),
#   2. mide calibracion solo en la primera sesion; las reanudaciones cargan
#      los mismos artefactos y fallan cerrado si falta alguno (CAM-11),
#   3. corta limpio via --campaign-timeout-seconds (deja margen sobre el
#      --time de Slurm para que el `finally:` de run_campaign restaure la
#      frecuencia ANTES del kill duro de Slurm -- nunca dejar que Slurm
#      mate el proceso, siempre debe salir por su cuenta).
# Verificar con `squeue`/`sacct` + el conteo de corridas aceptadas en
# campaign_metadata.json antes de cada relanzamiento para saber cuantas
# sesiones faltan.
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
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_full.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_gpu_N512 \
  --campaign-timeout-seconds 18000
'
echo CAMPAIGN_SCRIPT_DONE
