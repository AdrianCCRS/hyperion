#!/bin/bash
cd /home/latorresn/hyperion-kernels
./bin/ert_probe_gpu fp64 &
PID=$!
sleep 0.3
sudo -n /home/latorresn/yacacerest/probe_nvml_lock_direct
wait $PID
