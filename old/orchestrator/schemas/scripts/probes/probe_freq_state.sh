#!/bin/bash
for c in 0 1 2 3 4 5; do
  echo "cpu$c: gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor) min=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq) max=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) cur=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq)"
done
echo "---cpuinfo---"
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
echo "---turbo---"
cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null
cat /sys/devices/system/cpu/intel_pstate/max_perf_pct 2>/dev/null
