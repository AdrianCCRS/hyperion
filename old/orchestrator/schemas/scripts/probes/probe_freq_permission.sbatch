#!/bin/bash
#SBATCH --job-name=probe_freq_perm
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:03:00
#SBATCH --output=/home/latorresn/probe_freq_perm_%j.out
#SBATCH --error=/home/latorresn/probe_freq_perm_%j.err

echo "whoami: $(whoami)"
echo "Cpus_allowed_list: $(grep Cpus_allowed_list /proc/self/status)"
echo "--- perm cpu0 ---"
ls -la /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo "--- test write ---"
CUR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
echo 2000000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo "write exit=$?"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo $CUR > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo "DONE"
