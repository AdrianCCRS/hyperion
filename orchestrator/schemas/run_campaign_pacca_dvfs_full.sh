#!/bin/bash
# Primer intento de dataset final CPU: 9 kernels x 6 niveles x 10
# repeticiones = 540 corridas. No ejecutar hasta que el smoke actualizado
# confirme el helper de Turbo y la traza por ventana bajo F0--F4.
set -e -o pipefail
# ARC-126: sin "-u" a proposito -- rompe "module load" en este cluster
# (Lmod referencia LD_PRELOAD sin definir).

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=05:00:00 \
  ~/hyperion/scripts/pacca/with_cpu_turbo_disabled.sh bash -c '
# ARC-127: "module load openblas/0.3.21" solo, sin el padre gnu12, siempre
# fallo (RC=1, "Or load any one of these options: module load gnu12/12.4.0
# openblas/0.3.21" -- dependencia jerarquica no satisfecha) -- el "2>/dev/null"
# original se tragaba el error en silencio. Confirmado con ldd sobre
# dgemm_bench en una shell limpia: "libopenblas.so.0 => not found" sin este
# fix. dgemm_n2048 (bin/dgemm_bench) lo necesita en tiempo de ejecucion.
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels
python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaign_pacca_dvfs.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 16200
'
echo CAMPAIGN_SCRIPT_DONE
