#!/bin/bash
# Espera a que termine el proceso de la campana final GPU (PID conocido) y
# lanza la campana final CPU automaticamente a continuacion, DENTRO de la
# misma asignacion contenedora (job 7144). Pensado para correr desatado
# (nohup + disown) de cualquier sesion SSH, para que sobreviva a que el
# usuario cierre la conexion o se desconecte durante la noche.
#
# No corre CPU y GPU en paralelo a proposito: comparten los mismos cores
# para RAPL/perf (0-5 delegados, 6 collector, 7 consumer) -- correrlas
# simultaneamente contaminaria la medicion de energia de ambas.
set -uo pipefail

GPU_PID="${1:?uso: wait_gpu_then_cpu.sh <PID_run_campaign_gpu>}"
LOGDIR="$HOME/hyperion-results/final/logs"
LOG="$LOGDIR/manual_cpu_after_gpu_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGDIR"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

log "esperando a que termine GPU (PID $GPU_PID)..."
while ps -p "$GPU_PID" -o cmd= 2>/dev/null | grep -q "run_campaign.py"; do
  sleep 30
done
log "GPU terminado (el PID $GPU_PID ya no corresponde a run_campaign.py)."

# Turbo: 1 desactiva (convencion verificada F1-... set-turbo-state). Se
# reafirma por si algo lo reactivo entre que arranco GPU y termino.
sudo /usr/local/bin/set_turbo_state 1 || log "AVISO: no se pudo reafirmar turbo desactivado"

CPU_MANIFEST="$HOME/hyperion/scripts/pacca/final_campaign/cpu_final.yaml"
OUTPUT_DIR="/home/latorresn/hyperion-results/final/campaigns/cpu"

if [[ -d "$OUTPUT_DIR" ]]; then
  log "AVISO: $OUTPUT_DIR ya existe -- overwrite=false en el manifiesto hara fallar la corrida. No se borra automaticamente aqui (revisar a mano si esto ocurre)."
fi

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 | tee -a "$LOG" || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
source "$HOME/hyperion-venv/bin/activate" || { log "FATAL: no se pudo activar hyperion-venv"; exit 1; }

cd "$HOME/hyperion-kernels"
log "lanzando campana final CPU ($CPU_MANIFEST)..."
if python3 "$HOME/hyperion/fase1_telemetria/run_campaign.py" run-campaign \
    --manifest "$CPU_MANIFEST" --node-id pacca-a100 \
    --reference-kernel-ref npb_mg >>"$LOG" 2>&1; then
  log "campana final CPU: OK"
else
  log "campana final CPU: FALLO (ver $LOG completo) -- la asignacion sigue viva (job 7144), revisar a mano con srun --jobid=7144 --pty bash"
fi
