#!/bin/bash
#SBATCH --job-name=hyp_uncore_probe
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/latorresn/hyperion-results/analysis/uncore_probe_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analysis/uncore_probe_%j.err
#
# ¿Se pueden LEER los contadores de uncore DESDE DENTRO de un job de Slurm,
# con el entorno de módulos que usan las campañas?
#
# POR QUÉ ESTA SONDA EXISTE. El 2026-08-24 se comprobó por ssh directo al
# nodo que `uncore_imc/cas_count_read` devuelve conteos reales otra vez.
# Pero ese chequeo NO es suficiente para soltar 6431/6412, por una razón
# concreta: `perf_event_paranoid` sigue en 2, y lo que habilita la lectura
# es una CAPACIDAD DE ARCHIVO sobre el binario --
# `/usr/bin/perf cap_sys_ptrace,cap_sys_admin,cap_perfmon=ep`.
#
# Eso hace que el permiso dependa de QUÉ BINARIO se ejecute, no solo de un
# sysctl global. Y el lector de uncore no llama `perf_event_open` directo:
# hace `execvp("perf", argv)` (uncore_reader.cpp, ARC-118), que resuelve
# por $PATH. Si el entorno de módulos de la campaña antepusiera otro `perf`
# sin capacidades, E13 pasaría por ssh y la campaña fallaría igual --
# exactamente el modo de fallo silencioso del 2026-08-22 (27 corridas, 0
# aceptadas, todo `intensity_undefined`).
#
# Por eso la sonda replica el entorno del lanzador real
# (run_campaign_pacca_fine_grid_preflight.sbatch): mismos `module load`,
# mismo venv, mismo PYTHONPATH.
#
# NO se envuelve en with_cpu_turbo_disabled.sh a propósito: esa regla
# aplica a toda corrida con un nivel de frecuencia FIJO, porque el turbo
# invalida P_pico y el ridge. Esta sonda no fija ningún nivel ni mide
# tiempo ni energía -- solo pregunta si un contador se puede abrir.
#
# NO pide --gres=gpu ni --exclusive a propósito: con OverSubscribe=YES:4
# en la partición GPU, así puede entrar junto al job que ocupa el nodo en
# vez de hacer fila detrás de la batería de GPU ya encolada. La exclusividad
# que exige E11 es para la VALIDEZ de la medida de ancho de banda, no para
# la disponibilidad del contador, que es lo único que aquí se pregunta.
#
# Uso: sbatch probe_uncore_in_job.sbatch
set -e -o pipefail
# ARC-126: sin "-u" a propósito -- rompe "module load" en este clúster.

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"

echo "===== 0. CONTEXTO ====="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
date -Is

bash -c '
module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate

echo
echo "===== 1. QUE perf RESUELVE LA RUTA DE BUSQUEDA (lo que execvp vera) ====="
which -a perf || echo "SIN perf EN PATH"
for p in $(which -a perf); do
  printf "%s -> " "$p"
  getcap "$p" 2>/dev/null || echo "SIN CAPACIDADES"
done
echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"

echo
echo "===== 2. LECTURA CRUDA DEL IMC (todas las unidades) ====="
perf stat -a -x , -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ sleep 0.2 2>&1

echo
echo "===== 3. E13, EL CHEQUEO REAL DEL ORQUESTADOR ====="
python3 -c "
from orchestrator.preflight import check_uncore_readable
r = check_uncore_readable(True)
print(\"E13_PASSED=\", r.passed)
print(\"E13_RESULT=\", r)
"
'

echo
echo "UNCORE_PROBE_DONE"
