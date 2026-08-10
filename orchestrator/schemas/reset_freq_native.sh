#!/bin/bash
MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)
MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
for c in 0 1 2 3 4 5 6 7; do
  echo $MAX > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq 2>&1
  echo $MIN > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq 2>&1
done
for c in 0 1 2 3 4 5 6 7; do
  echo "cpu$c: min=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq) max=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq)"
done
