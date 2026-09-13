#!/bin/bash
#SBATCH --job-name=hyp_probe_transition_smoke
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/latorresn/hyperion-results/analisis/probe_smoke_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/analisis/probe_smoke_%j.err
#
# Sondeo minimo de la etapa `transition` antes de comprometer la cadena
# completa otra vez. Ejecuta UNA sola invocacion del probe, con la misma
# carga y los mismos parametros que usara la etapa real, y reporta el
# `result` que salga.
#
# Motivo: la etapa transition ya fallo dos veces (7055 por tolerance-mhz
# incompatible con la rejilla de 15 MHz de la A100; 7082 por carga fasica
# que no mantiene la GPU activa). Cada fallo rompe la cadena y nos cuesta el
# puesto en la cola, asi que sale mas barato gastar 30 s en comprobarlo que
# reencolar seis jobs a ciegas.
#
# Sin --time, se detiene a mano si hiciera falta.

set -eo pipefail

mkdir -p /home/latorresn/hyperion-results/analisis
OUT=/home/latorresn/hyperion-results/analisis/probe_smoke_$SLURM_JOB_ID

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
# libcublas: los module load no lo dejan en el path (job 7090, exit 127).
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/hyperion:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate"

PROBE="$HOME/hyperion/common/telemetry/build/gpu_clock_transition_probe"
WORKLOAD="$HOME/hyperion-kernels/bin/cublas_dgemm_bench --size 4096 --iterations 2000"

echo "nodo    : $(hostname)"
echo "probe   : $PROBE"
echo "workload: $WORKLOAD"
echo

# Mismos parametros que el barrido de cadencia de run_screening_to_report.sh,
# con la tolerancia y el max-wait ya corregidos.
"$PROBE" --workload-cmd "$WORKLOAD" --gpu "${CUDA_VISIBLE_DEVICES%%,*}" \
  --from-clock REF --to-clock 810 --tolerance-mhz 7 \
  --probe-interval-ns 5000000 \
  --warmup-ns 2000000000 --workload-min-active-ns 6000000000 \
  --max-wait-ns 3000000000 --out-dir "$OUT" || true

echo
echo "=== RESULTADO ==="
if [[ -f "$OUT/gpu_clock_transition_matrix.csv" ]]; then
  python3 - "$OUT/gpu_clock_transition_matrix.csv" <<'PY'
import csv, sys
with open(sys.argv[1]) as fh:
    for fila in csv.DictReader(fh):
        print("result        :", fila.get("result"))
        print("failure_reason:", fila.get("failure_reason") or "(ninguno)")
        print("n_readings    :", fila.get("n_readings"))
        print("metrics_valid :", fila.get("metrics_valid"))
PY
else
  echo "el probe no escribio matriz: revisar stderr"
fi
echo PROBE_SMOKE_DONE
