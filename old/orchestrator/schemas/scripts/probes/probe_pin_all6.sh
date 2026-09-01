#!/bin/bash
for c in 0 1 2 3 4 5; do
  P=/sys/devices/system/cpu/cpu$c/cpufreq
  echo "cpu$c before: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
done
echo "--- pin all 6 to 3600000 (max first, then min, per cpu) ---"
for c in 0 1 2 3 4 5; do
  P=/sys/devices/system/cpu/cpu$c/cpufreq
  echo 3600000 > $P/scaling_max_freq
  echo 3600000 > $P/scaling_min_freq
done
for c in 0 1 2 3 4 5; do
  P=/sys/devices/system/cpu/cpu$c/cpufreq
  echo "cpu$c after: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
done
echo "--- restore ---"
for c in 0 1 2 3 4 5; do
  P=/sys/devices/system/cpu/cpu$c/cpufreq
  echo 3600000 > $P/scaling_max_freq
  echo 800000 > $P/scaling_min_freq
done
for c in 0 1 2 3 4 5; do
  P=/sys/devices/system/cpu/cpu$c/cpufreq
  echo "cpu$c final: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
done
