#!/bin/bash
echo "hostname: $(hostname)"
echo "whoami: $(whoami)"
cd /home/latorresn/hyperion-kernels
./bin/ert_probe_gpu fp64 &
PID=$!
sleep 0.5
echo "=== lgc 1200,1200 via SSH directo (no srun bash -c) ==="
sudo -n nvidia-smi -i 0 -lgc 1200,1200
for i in 1 2 3 4; do
  nvidia-smi -i 0 --query-gpu=clocks.sm,utilization.gpu,power.draw --format=csv,noheader
  sleep 0.5
done
wait $PID
sudo -n nvidia-smi -i 0 -rgc
