#!/bin/bash
# Medicion previa al escenario F de la Fase 4: efecto de fijar el reloj de GPU en F1 (1260 MHz) frente a REF (reloj nativo) sobre
# fases de COMPUTO candidatas, incluido el DGEMM de CUTLASS de E-A, para el que la Fase 2 no tiene medicion en F1.
# Por kernel: NREP repeticiones intercaladas (el orden REF/F1 se alterna por repeticion). Registra duracion, energia de CPU (RAPL) y
# de GPU (NVML). Toca el reloj de GPU (nvidia-smi -lgc/-rgc) y lo libera al final. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_fase4_F_f1compute
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
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"; KERNEL_ROOT="$HOME/hyperion-kernels"; OUT="${OUT:-$HOME/hyperion-results/final/fase4_F_f1compute}"; mkdir -p "$OUT"
NREP="${NREP:-5}"; GPU=0; CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
GER="$HOME/hyperion-results/final/fase4_E/build/gpu_energy_read"; [ -x "$GER" ] || { echo "sin gpu_energy_read" >&2; exit 1; }
R0=/sys/class/powercap/intel-rapl:0; R1=/sys/class/powercap/intel-rapl:1; RMAX=$(cat $R0/max_energy_range_uj)
hostname; git -C "$ROOT" log --oneline -1
cleanup() { sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true; }; trap cleanup EXIT
sudo -n nvidia-smi -i $GPU -rgc || { echo "sudo -n nvidia-smi -rgc NO permitido" >&2; exit 1; }
CSV="$OUT/results.csv"; [ -f "$CSV" ] || echo "kernel,level,rep,wall_s,e_cpu_j,e_gpu_j,sm_mhz_med,rc" > "$CSV"
# candidatos de computo: el de E-A (sin medicion de Fase 2 en F1), conv2d (F1 mejora en Fase 2), LavaMD escalado (compute de E-B)
KERNELS=("gpu_cutlass_simt_dgemm_n4096" "gpu_cutlass_simt_conv2d" "rodinia_lavamd::-boxes1d 100")
one() {  # $1 kernel $2 nivel (REF|F1) $3 rep
  local k=$1 lv=$2 rep=$3
  if [ "$lv" = F1 ]; then sudo -n nvidia-smi -i $GPU -lgc 1260,1260 >/dev/null || return 1; else sudo -n nvidia-smi -i $GPU -rgc >/dev/null; fi
  sleep 5
  nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader,nounits -lms 500 > "$OUT/clk.tmp" & local S=$!
  local e0 f0 g0 t0; e0=$(cat $R0/energy_uj); f0=$(cat $R1/energy_uj); g0=$("$GER" $GPU); t0=$(date +%s.%N)
  python3 "$ROOT/fase3_daemon/composite_apps/survey_kernels.py" --node-id pacca-a100 --kernels-root "$KERNEL_ROOT" "$k" > "$OUT/last.out" 2>&1; local rc=$?
  local t1 e1 f1 g1; t1=$(date +%s.%N); e1=$(cat $R0/energy_uj); f1=$(cat $R1/energy_uj); g1=$("$GER" $GPU); kill $S 2>/dev/null
  local w ec eg mhz
  w=$(python3 -c "print(round($t1-$t0,3))"); ec=$(python3 -c "m=$RMAX; d=lambda a,b:(b-a)%m; print(round((d($e0,$e1)+d($f0,$f1))/1e6,2))"); eg=$(python3 -c "print(round(($g1-$g0)/1e3,2))")
  mhz=$(python3 -c "import statistics as s; v=[float(x) for x in open('$OUT/clk.tmp') if x.strip() and float(x)>300]; print(s.median(v) if v else 0)")
  echo "${k//,/ },$lv,$rep,$w,$ec,$eg,$mhz,$rc" | tee -a "$CSV"
}
for rep in $(seq 1 "$NREP"); do
  for k in "${KERNELS[@]}"; do
    if [ $((rep % 2)) = 1 ]; then order="REF F1"; else order="F1 REF"; fi
    for lv in $order; do grep -q "^${k//,/ },$lv,$rep," "$CSV" || one "$k" "$lv" "$rep"; done
  done
done
echo "=== FIN medicion F1 en computo ==="
