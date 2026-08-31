#!/bin/bash
#SBATCH --job-name=hyp_big_ref_cpu_fair_pacca01
#SBATCH --partition=normal
#SBATCH --nodelist=pacca01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/home/latorresn/hyperion-results/campaigns/big_ref_cpu_fair_pacca01_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/campaigns/big_ref_cpu_fair_pacca01_%j.err
#
# NO LANZAR TODAVIA -- ver el bloque de bloqueadores al inicio de
# orchestrator/schemas/campaigns/campaign_pacca_cpu_big_ref_fair_pacca01.yaml
# (permisos pendientes en pacca01, binarios sin compilar/verificar para ese
# node_id, sin preflight corrido).
#
# Correccion del sesgo CPU/GPU (2026-08-31): mismos 9 config_id que la
# campana cancelada (jobs 6763/6769), ahora con 24 nucleos fisicos
# delegados (socket 0 completo menos 2 reservados para telemetria) en vez
# de 6. Ver el manifiesto para el diseno completo de nucleos.
#
# --time=5h es mas holgado que el original (3h para 6 nucleos): con 4x mas
# nucleos delegados el trabajo por corrida no crece (el kernel ya paraleliza
# sobre los nucleos disponibles), pero el presupuesto no esta calculado
# todavia contra una medicion real en este nodo -- placeholder conservador,
# corregir con el tiempo real del primer preflight antes de campanas mas
# largas.
set -e -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
bash ~/hyperion/scripts/pacca/reset_cpu_freq_range.sh
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_cpu_big_ref_fair_pacca01.yaml \
  --node-id pacca01 \
  --hostname "$(hostname)" \
  --reference-kernel-ref dual_gemm_cpu_N512 \
  --campaign-timeout-seconds 17000
'
echo CAMPAIGN_SCRIPT_DONE
