#!/bin/bash
# Diagnostico de la bimodalidad de myocyte en la aplicacion E-B (familias ineditas): la fase de myocyte dura ~41 s o ~50 s sin
# relacion clara con el brazo: en la campana 7696 (3 reps) 2 de 3 de REF fueron lentas, pero en la repeticion de 8 bloques
# (fase4_EB_confirmatorio) REF fue rapida 8 de 8 (myocyte 40.3-41.1 s) y los brazos con agente mostraron 4 lentas (~50 s) y 6
# intermedias (42-44.5 s) de 16. Hipotesis a contrastar: el hilo del agente de GPU (sin fijar a nucleos) cae en los nucleos 0-5 de la
# aplicacion y retrasa a myocyte, una fase limitada por lanzamientos. Repite REF, sombra y sombra con el daemon fijado al nucleo 6
# (sombra_gpupin, fuera de 0-5; es la prueba directa de la hipotesis y el arreglo candidato). Repite REF y sombra (mismas condiciones, aleatorizado por
# bloques) y registra durante toda la asignacion: reloj/potencia/temperatura/throttling de la GPU (NVML, 250 ms); frecuencia de los
# nucleos 0-5 y 16-21, carga del nodo y los 3 procesos que mas CPU usan (1 s); en que nucleo corre cada hilo con >1% de CPU (1 s, ps -L);
# y la celda en curso (0.5 s). El analisis cruza esos registros con la duracion de cada fase (phases.jsonl) de cada celda.
# No usar localmente. Sin --time por politica operativa de pacca; se corta a mano con scancel si hace falta.
#SBATCH --job-name=hyp_fase4_myocyte_diag
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err

set -uo pipefail

ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
RESULTS_ROOT="$HOME/hyperion-results/final"
BASE="${OUT_BASE:-$RESULTS_ROOT/fase4_myocyte_diag_${SLURM_JOB_ID}}"
REPS="${REPS:-10}"
mkdir -p "$BASE/telemetria"
T="$BASE/telemetria"
state() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$BASE/campaign_status.txt"; }

[[ -f "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" ]] || { state "FALLO no existe launcher de matriz en $ROOT"; exit 1; }
state "ASIGNACION TOMADA nodo=$(hostname) job=$SLURM_JOB_ID root=$ROOT base=$BASE reps=$REPS"
git -C "$ROOT" log --oneline -1 | tee -a "$BASE/campaign_status.txt"

# muestreadores en segundo plano: viven toda la asignacion y se detienen al salir
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,clocks_throttle_reasons.active,pstate \
    --format=csv,noheader,nounits -lms 250 > "$T/gpu_nvml.csv" 2>"$T/gpu_nvml.err" &
NV_PID=$!
(
  CPUS="0 1 2 3 4 5 16 17 18 19 20 21"
  echo "epoch,no_turbo,loadavg1,$(for c in $CPUS; do printf 'f%s,' "$c"; done)top3" > "$T/cpu_nodo.csv"
  i=0
  while true; do
    ep=$(date +%s.%N)
    f=""; for c in $CPUS; do f+="$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null),"; done
    top3=$(ps -eo user:12,pcpu,comm --sort=-pcpu --no-headers | head -3 | tr -s ' ' | tr '\n' ';')
    echo "$ep,$(cat /sys/devices/system/cpu/intel_pstate/no_turbo),$(cut -d' ' -f1 /proc/loadavg),${f}${top3}" >> "$T/cpu_nodo.csv"
    sleep 1
  done
) &
CPU_PID=$!
(
  echo "epoch,celda_en_curso" > "$T/celda.csv"
  while true; do
    echo "$(date +%s.%N),$(ls -t "$BASE/matriz/cells" 2>/dev/null | head -1)" >> "$T/celda.csv"
    sleep 0.5
  done
) &
CELL_PID=$!
(
  echo "epoch psr pcpu pid tid comm" > "$T/hilos_por_nucleo.txt"
  while true; do
    ep=$(date +%s.%N)
    ps -eLo psr,pcpu,pid,tid,comm --no-headers | awk -v ep="$ep" '$2 > 1 { print ep, $1, $2, $3, $4, $5 }' >> "$T/hilos_por_nucleo.txt"
    sleep 1
  done
) &
THR_PID=$!
trap 'kill $NV_PID $CPU_PID $CELL_PID $THR_PID 2>/dev/null' EXIT

state "INICIO matriz REF, sombra y sombra_gpupin, solo E-B"
if env OUT="$BASE/matriz" HYPERION_ROOT="$ROOT" \
    CYCLES=2 GAP_S=1 CYCLES_GPUMEM=1 REPS_GENERAL="$REPS" RANDOM_SEED=20260926 ORDER_MODE=balanced_blocks \
    ONLY_SCOPES="gpumem" ONLY_SETS="unseen" ONLY_ARMS="base sombra sombra_gpupin" \
    bash "$ROOT/scripts/pacca/hyp_fase4_matrix.sbatch" > "$BASE/matriz.log" 2>&1; then
    state "OK matriz"
else
    state "FALLO matriz; ver $BASE/matriz.log"
    exit 1
fi
state "TERMINO; datos en $BASE"
