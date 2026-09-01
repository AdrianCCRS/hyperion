#!/bin/bash
# Auditoria de solo lectura de felix: topologia, cpufreq, RAPL, cache,
# espacio en disco, estado del repo, modulos y GPU. No escribe nada en
# sysfs. Usado para la Parte 0 del plan y las ampliaciones de ARC-29/ARC-35.
set -euo pipefail

echo "=== CPUSET EFECTIVO ==="
cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpuset.cpus.effective

echo "=== CPUFREQ: governors disponibles (cpu0) ==="
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null || echo NOFILE

echo "=== CPUFREQ: dominio de frecuencia por CPU ==="
for c in 0 1 8 16 24 32; do
  d=/sys/devices/system/cpu/cpu$c/cpufreq
  if [ -d "$d" ]; then
    echo "--cpu$c--"
    for f in related_cpus affected_cpus freqdomain_cpus scaling_governor scaling_cur_freq; do
      [ -f "$d/$f" ] && echo "  $f: $(cat $d/$f)"
    done
  fi
done

echo "=== scaling_setspeed existe hoy ==="
ls /sys/devices/system/cpu/cpu0/cpufreq/ | grep setspeed || echo NOFILE

echo "=== cpuinfo: modelo y conteo physical id / core id ==="
grep -m1 'model name' /proc/cpuinfo
echo "logical_processors: $(grep -c '^processor' /proc/cpuinfo)"
echo "sockets: $(grep 'physical id' /proc/cpuinfo | sort -u | wc -l)"
echo "physical_cores: $(grep -E 'physical id|core id' /proc/cpuinfo | paste - - | sort -u | wc -l)"

echo "=== cache index cpu0 ==="
for i in /sys/devices/system/cpu/cpu0/cache/index*; do
  echo "--$(basename $i)--"
  for f in level type size coherency_line_size shared_cpu_list; do
    [ -f "$i/$f" ] && echo "  $f: $(cat $i/$f)"
  done
done

echo "=== /scratch ==="
df -h /scratch 2>&1 | tail -n +1
touch /scratch/$USER-write-test 2>&1 && echo WRITABLE && rm -f /scratch/$USER-write-test || echo NOT-WRITABLE-OR-NOT-FOUND

echo "=== HOME quota ==="
df -h $HOME 2>&1 | tail -n +1

echo "=== ~/hyperion en el cluster ==="
if [ -d ~/hyperion ]; then
  cd ~/hyperion && git log --oneline -3 2>&1
  echo "--status--"
  git status --short 2>&1 | head -20
else
  echo NO-EXISTE
fi

echo "=== modulos relevantes ==="
module avail 2>&1 | grep -iE 'gnu|gcc|mpi|cuda|conda|cmake|openblas'

echo "=== gfortran/gcc/cmake por defecto ==="
gcc --version | head -1
gfortran --version | head -1
cmake --version | head -1

echo "=== perf CLI y contadores de uncore (ver ARC-35) ==="
perf --version 2>&1
ls /sys/bus/event_source/devices/ 2>&1 | grep -i uncore

echo "=== GPU ==="
nvidia-smi -L 2>&1 || echo "nvidia-smi no disponible (¿--gres=gpu:1?)"

echo "DONE"
