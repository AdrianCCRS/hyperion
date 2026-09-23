#!/bin/bash
# Prueba corta de validacion del candado de reloj de GPU (H1, reparado por
# el administrador el 2026-09-23 -- reset via nvidia-smi -rgc +
# nvidia-bug-report). Pedido explicito del admin: validar con NUESTRO
# propio flujo (common/hpc/gpu_freqctl.py::apply_gpu_frequency, la funcion
# real que usan las campanas) y la cuenta delegada, antes de relanzar nada
# largo. Aplica 1410/810/210 MHz con carga GPU real sostenida
# (dual_cholesky_gpu_N32768, checksum verificado) y lee clocks.sm +
# power.draw despues de cada uno. apply_gpu_frequency YA aborta
# (GpuFrequencyControlError) si la relectura bajo carga real no coincide
# con el techo pedido (ARC-94/ARC-112) -- este script solo lo ejercita de
# verdad, no reimplementa el chequeo.
#SBATCH --job-name=hyp_gpu_freq_validate_h1
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/gpu_freq_validate_h1_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/gpu_freq_validate_h1_%j.err

set -uo pipefail
ROOT="$HOME/hyperion"
KERNELS_ROOT="$HOME/hyperion-kernels"

module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HYPERION_GPU_FREQ_WRITE_CAPABLE=1
source "$HOME/hyperion-venv/bin/activate"

hostname
nvidia-smi --query-gpu=name,driver_version,clocks.sm,power.draw --format=csv,noheader

echo "=== carga GPU sostenida real en background: dual_cholesky_gpu_N32768 (~60s) ==="
python3 - "$KERNELS_ROOT" <<'PYEOF' &
import sys
sys.path.insert(0, "/home/latorresn/hyperion")
import subprocess
from dataclasses import replace
from pathlib import Path
from common.hpc.catalog import load_catalog, verify_binary
import shlex

kernels_root = Path(sys.argv[1])
catalog = load_catalog("/home/latorresn/hyperion/fase1_telemetria/catalog/catalog.yaml")
entry = catalog["dual_cholesky_gpu_N32768"]
resolved = str(kernels_root / entry.exec_path)
ok = verify_binary(replace(entry, exec_path=resolved), node_id="pacca-a100")
print(f"[carga] checksum verificado: {ok}", flush=True)
if not ok:
    raise SystemExit("checksum invalido, abortando carga")
argv = [resolved, *shlex.split(entry.exec_args)]
proc = subprocess.run(argv, cwd=str(kernels_root), capture_output=True, text=True)
print(f"[carga] exit_code={proc.returncode}", flush=True)
print(proc.stdout[-500:], flush=True)
PYEOF
LOAD_PID=$!

sleep 3  # dejar que la carga suba de verdad antes de tocar el reloj

echo "=== ciclo de validacion: apply_gpu_frequency real (gpu_freqctl.py) ==="
python3 <<'PYEOF'
import sys, time
sys.path.insert(0, "/home/latorresn/hyperion")
from types import SimpleNamespace
from common.hpc import environment as environment_module
from common.hpc import gpu_freqctl

env = environment_module.detect_environment()
print(f"gpu_frequency_write_capable={env.gpu_frequency_write_capable} "
      f"gpu_available_clocks_mhz={env.gpu_available_clocks_mhz}")
available = env.gpu_available_clocks_mhz
if not available:
    raise SystemExit("sin gpu_available_clocks_mhz -- no se puede validar")
lo, hi = min(available), max(available)

failure = None
try:
    for mhz in (1410, 810, 210):
        fraction = 0.0 if hi == lo else max(0.0, min(1.0, (mhz - lo) / (hi - lo)))
        level = SimpleNamespace(id=f"validate_{mhz}mhz", mode="fixed", fraction=fraction)
        try:
            applied = gpu_freqctl.apply_gpu_frequency(level, env, gpu_index=None)
            print(f"[{mhz}MHz] OK: {applied}")
        except gpu_freqctl.GpuFrequencyControlError as exc:
            print(f"[{mhz}MHz] FALLO: {exc}")
            failure = exc
            break
        time.sleep(2)
finally:
    # SIEMPRE restaurar, incluso si un nivel intermedio fallo -- dejar la
    # GPU bloqueada a un reloj bajo seria peor que el problema que se
    # estaba diagnosticando.
    print("=== restaurando (-rgc), pase lo que pase ===")
    try:
        restored = gpu_freqctl.apply_gpu_frequency(
            SimpleNamespace(id="restore", mode="native_governor"), env, gpu_index=None,
        )
        print(f"restaurado: {restored}")
    except gpu_freqctl.GpuFrequencyControlError as exc:
        print(f"RESTAURACION TAMBIEN FALLO: {exc} -- avisar al admin de inmediato")
        raise

if failure is not None:
    raise failure
PYEOF
VALIDATE_STATUS=$?

echo "=== esperando a que termine la carga de fondo ==="
wait "$LOAD_PID"
LOAD_STATUS=$?

echo "validacion: status=$VALIDATE_STATUS carga: status=$LOAD_STATUS"
if [ "$VALIDATE_STATUS" -ne 0 ]; then
  echo "VALIDACION H1 FALLO -- el candado de reloj NO responde de forma confiable, avisar al admin" >&2
  exit 1
fi
echo "VALIDACION H1: OK"
