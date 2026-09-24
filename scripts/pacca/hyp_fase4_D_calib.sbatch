#!/bin/bash
# Calibracion del escenario D de la Fase 4: tiempo por paso de cada benchmark de LAMMPS (paquete GPU, 1 proceso MPI) para fijar el
# numero de pasos de la matriz en ~60 s. Sin agentes, sin energia, sin tocar frecuencias. NUNCA local. Sin --time.
#SBATCH --job-name=hyp_fase4_D_calib
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
OUT="${OUT:-$HOME/hyperion-results/final/fase4_D_calib}"; mkdir -p "$OUT"
BENCH=/usr/local/src/lammps/lammps-23Jun2022/bench; NV=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1
export LD_LIBRARY_PATH="/opt/ohpc/pub/lammps/2023_gpu/lib64:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib:$NV/comm_libs/openmpi/openmpi-3.1.5/lib:$NV/math_libs/12.0/lib64:$NV/cuda/12.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LAMMPS_POTENTIALS=/opt/ohpc/pub/lammps/2023_gpu/share/lammps/potentials OPAL_PREFIX="$NV/comm_libs/openmpi/openmpi-3.1.5"
export OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1 OMPI_MCA_mtl=^ofi OMPI_MCA_btl_vader_single_copy_mechanism=none
LMPCMD="$OPAL_PREFIX/bin/mpirun -np 1 --oversubscribe /opt/ohpc/pub/lammps/2023_gpu/bin/lmp"
hostname
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
# (benchmark, variables de tamano): lj y eam a 256k atomos (con 2M ni 100 pasos terminaron en 75 s); chain y rhodo, 32000 atomos fijos
for spec in "lj|-var x 2 -var y 2 -var z 2" "eam|-var x 2 -var y 2 -var z 2" "chain|" "rhodo|"; do
  b=${spec%%|*}; VARS=${spec#*|}; W="$OUT/$b"; rm -rf "$W"; mkdir -p "$W"; cp -r "$BENCH"/* "$W"/; cd "$W"
  T0=$(date +%s.%N)
  taskset -c 0-3 timeout 300 $LMPCMD -sf gpu -pk gpu 1 -in "in.$b" $VARS -log "$W/log.lammps" > "$W/stdout.txt" 2> "$W/stderr.txt"; RC=$?
  T1=$(date +%s.%N)
  echo "$b rc=$RC total=$(python3 -c "print(round($T1-$T0,1))")s $(grep -m1 'Loop time of' "$W/log.lammps" 2>/dev/null)"
done
echo "=== FIN calibracion D ==="
