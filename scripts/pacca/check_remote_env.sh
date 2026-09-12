#!/bin/bash
# Preflight de entorno para pacca/paccaA100: corre en el nodo de LOGIN, sin
# encolar nada en Slurm. Detecta deriva de modulos/toolchain (p.ej. un
# `module load cmake/3.24.2` en ~/.bashrc que dejo de existir tras una
# actualizacion del cluster) ANTES de someter una cadena sbatch larga.
#
# Uso: bash scripts/pacca/check_remote_env.sh
#
# Contexto: ARC screening 2026-09-10 -- job 6811 (f1_screening_preflight)
# fallo en 3s con exit 127 (`cmake: command not found`) porque
# ~/.bashrc:33 en pacca carga cmake/3.24.2, modulo retirado (solo queda
# cmake/4.3.4 en /opt/ohpc/pub/modulefiles). El check costo <5s por SSH
# plano y habria evitado encolar/cancelar 6818-6823 en cascada.
#
# Ver memoria: feedback-preflight-before-long-campaigns,
# feedback-verify-remote-build-not-just-source-sync,
# screening-chain-pacca-launch.

set -uo pipefail

JUMP="hpc-unicartagena"
NODE="pacca"
FAIL=0

run_remote() {
  # $1: descripcion, $2: comando remoto (login shell -> dispara .bashrc)
  local desc="$1" cmd="$2"
  local out
  out=$(ssh "$JUMP" "ssh $NODE $(printf '%q' "$cmd")" 2>&1)
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $desc"
    echo "$out" | sed 's/^/       /'
    FAIL=1
  else
    echo "[ OK ] $desc"
  fi
}

echo "=== Preflight de entorno remoto: $NODE (via $JUMP) ==="
echo

# 1. Login limpio: cualquier "Lmod has detected an error" al hacer login
#    (dispara .bashrc) es deriva de modulos que rompera CUALQUIER job.
out=$(ssh "$JUMP" "ssh $NODE 'echo LOGIN_OK'" 2>&1)
if echo "$out" | grep -q "Lmod has detected"; then
  echo "[FAIL] Login limpio (sin errores de Lmod en .bashrc)"
  echo "$out" | grep -A2 "Lmod has detected" | sed 's/^/       /'
  FAIL=1
else
  echo "[ OK ] Login limpio (sin errores de Lmod en .bashrc)"
fi

# 2. Cadena de modulos documentada para el screening (gnu12 SIEMPRE primero)
run_remote "module load gnu12/12.4.0 nvhpc/23.1 openblas/0.3.21" \
  "module load gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 openblas/0.3.21 && echo MODULES_OK"

# 3. Toolchain que invoca run_screening_to_report.sh (stage validate: cmake)
run_remote "cmake en PATH y version" "cmake --version | head -1"

# 4. venv de Python del proyecto
run_remote "hyperion-venv activable" \
  "source ~/hyperion-venv/bin/activate && python3 -c 'import sys; print(sys.executable)'"

# 5. GPU visible desde el nodo de login (no siempre, pero si falla el
#    binario CUDA/nvcc esto ya lo adelanta)
run_remote "nvcc en PATH" "nvcc --version | tail -1" || true

echo
if [[ $FAIL -ne 0 ]]; then
  echo "=== RESULTADO: al menos un chequeo fallo. NO someter la cadena sbatch todavia. ==="
  exit 1
else
  echo "=== RESULTADO: entorno limpio, seguro someter la cadena. ==="
  exit 0
fi
