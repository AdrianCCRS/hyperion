#!/bin/bash
C=0
P=/sys/devices/system/cpu/cpu$C/cpufreq
echo "before: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
echo "write max=3600000"
echo 3600000 > $P/scaling_max_freq
echo "after max write: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
echo "write min=3600000"
echo 3600000 > $P/scaling_min_freq
echo "exit=$?"
echo "after min write: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
sleep 0.2
echo "after sleep: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
echo "--- restore ---"
echo 3600000 > $P/scaling_max_freq
echo 800000 > $P/scaling_min_freq
echo "final: min=$(cat $P/scaling_min_freq) max=$(cat $P/scaling_max_freq)"
