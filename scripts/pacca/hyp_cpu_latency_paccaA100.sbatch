#!/bin/bash
# Mide la latencia de inferencia del clasificador CPU en paccaA100 (nodo de destino).
# Fija el proceso a los cores 0-3 (rango de frecuencia sano) y registra el reloj antes y despues.
# Solo correr con el nodo libre: mide CPU y no debe coexistir con una campana activa. Sin --time.
#SBATCH --job-name=hyp_cpu_latency_a100
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/cpu_latency_a100_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/cpu_latency_a100_%j.err

set -euo pipefail
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/hyperion-venv/bin/activate"

SRC="$HOME/hyperion-results/final/training/cpu_quality_30fam_20260920/source30.csv"
OUT="$HOME/hyperion-results/final/training/cpu_quality_30fam_base6_20260921"

freq_state() {
  echo "[$1] no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  for c in 0 1 2 3; do
    d=/sys/devices/system/cpu/cpu$c/cpufreq
    echo "[$1] cpu$c min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq) cur=$(cat $d/scaling_cur_freq) gov=$(cat $d/scaling_governor)"
  done
}

hostname
freq_state antes
taskset -c 0-3 python3 -m fase2_clasificador.analysis.cpu_quality_report \
  --source "$SRC" --out "$OUT" --stages latency --cap 1000
freq_state despues
