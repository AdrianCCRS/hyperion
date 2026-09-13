#!/bin/bash
#SBATCH --job-name=hyp_final_holder
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/holder_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/holder_%j.err
#
# Asignacion "contenedor" para las dos campanas finales (CPU + GPU).
#
# Motivo: en esta cuenta compartida la cola es FIFO puro (PriorityWeight de
# Age/FairShare/JobSize/QOS = 0), sin preempcion. Mientras esta asignacion
# este RUNNING, el nodo es nuestro y nadie nos lo puede quitar. El riesgo
# real no es perder el nodo mientras corremos: es que una etapa falle, la
# cadena de --dependency=afterok se rompa (DependencyNeverSatisfied, como
# paso con 7101-7106) y al reencolar quedemos detras de cualquier job ajeno
# que se haya metido mientras tanto. Este script evita ese reencolado por
# completo: las dos campanas corren DENTRO de la misma asignacion, y si una
# falla, la asignacion NO se suelta -- queda viva en espera para diagnosticar
# y reintentar a mano, sin volver a pasar por la cola de Slurm.
#
# Para conectar una shell interactiva a ESTA MISMA asignacion (no gasta
# turno de cola, es un job step):
#   srun --jobid=<este_job_id> --pty bash
#
# Para soltar el nodo cuando ya no se necesite:
#   touch /home/latorresn/hyperion-results/final/HOLDER_STOP
#   (o `scancel <este_job_id>` desde fuera)
#
# SIN --time A PROPOSITO (particion GPU es MaxTime=UNLIMITED). Control
# siempre manual via scancel/HOLDER_STOP, nunca por reloj.
#
# set -u pero NO -e: un fallo dentro de una campana no debe tumbar el script
# completo (eso soltaria la asignacion). Cada paso maneja su propio error.

set -uo pipefail

HOLD_ROOT="$HOME/hyperion-results/final"
LOGDIR="$HOLD_ROOT/logs"
STATUS="$HOLD_ROOT/holder_status_${SLURM_JOB_ID}.txt"
STOP_FILE="$HOLD_ROOT/HOLDER_STOP"
mkdir -p "$LOGDIR"
rm -f "$STOP_FILE"

estado() { printf '[%s][holder] %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"; }

CPU_MANIFEST="$HOME/hyperion/scripts/pacca/final_campaign/cpu_final.yaml"
GPU_MANIFEST="$HOME/hyperion/scripts/pacca/final_campaign/gpu_final.yaml"
NODE_ID="pacca-a100"
REFERENCE_KERNEL="npb_mg"
KERNEL_ROOT="$HOME/hyperion-kernels"
LAUNCHER="$HOME/hyperion/common/telemetry/build/telemetry_kernel_launcher"

fail() { estado "FATAL: $*"; exit 2; }

[[ -f "$CPU_MANIFEST" ]] || fail "no existe $CPU_MANIFEST"
[[ -f "$GPU_MANIFEST" ]] || fail "no existe $GPU_MANIFEST"
[[ -d "$KERNEL_ROOT" ]]  || fail "no existe kernel-root: $KERNEL_ROOT"
[[ -x "$LAUNCHER" ]]     || fail "falta el binario de telemetria: $LAUNCHER"

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate" || fail "no se pudo activar hyperion-venv"

# Turbo: 1 DESACTIVA (escribe no_turbo=1), 0 reactiva. Verificado en
# hardware 2026-09-12. Se desactiva para TODA la vida de la asignacion
# (igual que en los wrappers previos de cada campana por separado) y se
# restaura al salir pase lo que pase.
restore_turbo() { sudo /usr/local/bin/set_turbo_state 0 || true; }
trap restore_turbo EXIT
sudo /usr/local/bin/set_turbo_state 1

estado "ASIGNACION TOMADA -- nodo=$(hostname) job=$SLURM_JOB_ID"
estado "conectar con: srun --jobid=$SLURM_JOB_ID --pty bash"
estado "soltar con  : touch $STOP_FILE   (o scancel $SLURM_JOB_ID)"

cd "$KERNEL_ROOT"

ejecutar_campania() {
  local nombre="$1" manifest="$2"
  local log="$LOGDIR/holder_${nombre}_${SLURM_JOB_ID}.log"
  estado "campana $nombre: iniciando ($manifest) -- log: $log"
  if python3 "$HOME/hyperion/fase1_telemetria/run_campaign.py" run-campaign \
      --manifest "$manifest" --node-id "$NODE_ID" \
      --reference-kernel-ref "$REFERENCE_KERNEL" >"$log" 2>&1; then
    estado "campana $nombre: OK"
  else
    estado "campana $nombre: FALLO (revisar $log) -- la asignacion SIGUE VIVA, conectate con srun --jobid=$SLURM_JOB_ID --pty bash y reintenta a mano (el resume salta las corridas ya aceptadas, CAM-03/CAM-11)"
  fi
}

ejecutar_campania final_cpu "$CPU_MANIFEST"
ejecutar_campania final_gpu "$GPU_MANIFEST"

estado "ambas campanas terminaron su intento. Asignacion sigue viva en espera."
estado "sigue disponible con: srun --jobid=$SLURM_JOB_ID --pty bash"

while [[ ! -f "$STOP_FILE" ]]; do
  sleep 60
done
estado "HOLDER_STOP detectado -- liberando el nodo."
