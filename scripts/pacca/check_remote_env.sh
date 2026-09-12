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
COMPUTE_NODE="paccaA100"
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

# 6. Chequeos que SOLO valen en el nodo de computo. El nodo de login y
#    paccaA100 divergen en paquetes de sistema (p.ej. libjsoncpp existe en
#    login pero no en paccaA100, lo que rompio cmake 2026-09-12) y el
#    estado de cpufreq vive en el nodo real. Un srun corto y no exclusivo
#    basta; sigue siendo mucho mas barato que descubrirlo en la cola.
echo
echo "--- Chequeos en el nodo de computo ($COMPUTE_NODE) ---"

compute_out=$(ssh "$JUMP" "ssh $NODE $(printf '%q' "srun -p GPU -w $COMPUTE_NODE -N1 -n1 --time=00:03:00 -J hyp_preflight_env bash -c 'source ~/hyperion-venv/bin/activate 2>/dev/null; echo CMAKE=\$(cmake --version 2>&1 | head -1); for c in 0 1 2 3 4 5; do echo FREQ \$c \$(cat /sys/devices/system/cpu/cpu\$c/cpufreq/scaling_min_freq) \$(cat /sys/devices/system/cpu/cpu\$c/cpufreq/scaling_max_freq); done; echo NOTURBO=\$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)'")" 2>&1)

# 6a. cmake realmente EJECUTABLE en el nodo de computo, no solo en login.
if echo "$compute_out" | grep -q "CMAKE=cmake version"; then
  echo "[ OK ] cmake ejecutable en $COMPUTE_NODE: $(echo "$compute_out" | grep -o 'CMAKE=.*' | cut -d= -f2-)"
else
  echo "[FAIL] cmake NO corre en $COMPUTE_NODE (libreria de sistema ausente?)"
  echo "$compute_out" | grep -i "CMAKE=" | sed 's/^/       /'
  FAIL=1
fi

# 6b. Contaminacion de cpufreq: una campana muerta por SIGKILL (scancel,
#     timeout de Slurm, OOM) no ejecuta el `finally` que restaura la
#     frecuencia y deja los cores clavados en min==max. Toda calibracion
#     posterior falla D03 con un P_pico absurdamente bajo. Ojo:
#     max<cpuinfo_max con turbo desactivado NO es contaminacion, el kernel
#     recorta al reloj base; la firma real es min == max.
pinned=$(echo "$compute_out" | awk '/^FREQ /{if ($3 == $4) print $2}')
if [[ -n "$pinned" ]]; then
  echo "[FAIL] cpufreq CONTAMINADO: cores con min==max (clavados): $(echo $pinned | tr '\n' ' ')"
  echo "$compute_out" | grep '^FREQ ' | sed 's/^/       /'
  echo "       Reparar: escribir cpuinfo_min_freq/cpuinfo_max_freq en esos cores."
  FAIL=1
else
  echo "[ OK ] cpufreq sin contaminacion (ningun core con min==max)"
fi

echo
if [[ $FAIL -ne 0 ]]; then
  echo "=== RESULTADO: al menos un chequeo fallo. NO someter la cadena sbatch todavia. ==="
  exit 1
else
  echo "=== RESULTADO: entorno limpio, seguro someter la cadena. ==="
  exit 0
fi
