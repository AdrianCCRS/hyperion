#!/bin/bash
echo "=== candado ANTES de lanzar el kernel (no despues) ==="
sudo -n nvidia-smi -i 0 -lgc 1200,1200
echo "LGC_EXIT=$?"
nvidia-smi -i 0 --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader
cd /home/latorresn/hyperion-kernels
./bin/ert_probe_gpu fp64 &
PID=$!
for i in 1 2 3 4 5 6; do
  sleep 0.3
  nvidia-smi -i 0 --query-gpu=clocks.sm,utilization.gpu,power.draw --format=csv,noheader
done
wait $PID
sudo -n nvidia-smi -i 0 -rgc
