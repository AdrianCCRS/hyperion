#!/bin/bash
# Lee el estado de frecuencia/turbo de paccaA100 (todos los CPU lógicos) y, si
# alguno se sale del estado nativo conocido, lo repara y lo verifica. Nativo:
# no_turbo=0; CPUs delegados (0-5 y sus hermanos 16-21): min=800000
# max=3200000 (estado observado tras las campañas); el resto: min=800000
# max=cpuinfo_max_freq. (La v1 de este script asumio 3200000 para TODOS y
# bajo el max de los no delegados; se corrigio.) Sin --time. Sirve tras un corte
# anormal de un job que actuaba sobre frecuencia (preflight C8, campañas).
#SBATCH --job-name=hyp_node_state_check
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/node_state_check_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/node_state_check_%j.err
set -uo pipefail
C=/sys/devices/system/cpu
bad=0
DELEG=" 0 1 2 3 4 5 16 17 18 19 20 21 "
expected_max() {  # $1 = cpu, $2 = ruta cpufreq
  case "$DELEG" in *" $1 "*) echo 3200000 ;; *) cat $2/cpuinfo_max_freq ;; esac
}
report() {
  echo "no_turbo=$(cat $C/intel_pstate/no_turbo)"
  for d in $C/cpu[0-9]*/cpufreq; do
    c=${d#$C/cpu}; c=${c%/cpufreq}
    mn=$(cat $d/scaling_min_freq); mx=$(cat $d/scaling_max_freq)
    [ "$mn" = 800000 ] && [ "$mx" = "$(expected_max $c $d)" ] || { echo "cpu$c min=$mn max=$mx  <-- FUERA DE LO NATIVO"; bad=1; }
  done
  [ "$(cat $C/intel_pstate/no_turbo)" = 0 ] || bad=1
}
echo "=== ANTES ==="; report
ps -eo pid,args | grep -E "[c]pu_loop_main|[c]pu_freq_actuator" || echo "sin procesos del actuador"
if [ "$bad" = 1 ]; then
  echo "=== reparando ==="
  sudo -n /usr/local/bin/set_turbo_state 0
  for d in $C/cpu[0-9]*/cpufreq; do
    c=${d#$C/cpu}; c=${c%/cpufreq}
    echo "$(expected_max $c $d)" > $d/scaling_max_freq; echo 800000 > $d/scaling_min_freq
  done
  bad=0; echo "=== DESPUES ==="; report
fi
[ "$bad" = 0 ] && echo "ESTADO NATIVO OK" || { echo "ESTADO NO REPARADO" >&2; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv
