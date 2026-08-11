#!/bin/bash
# Corre gaussian en background para generar carga real, y mide cuanto
# tarda el reloj SM en converger tras -lgc mientras hay trabajo activo.
cd /home/latorresn/yacacerest
./gaussian_build_test -s 8192 &
PID=$!
sleep 0.3
sudo -n nvidia-smi -i 0 -lgc 210,210
echo "LGC_EXIT=$?"
for i in 1 2 3 4 5 6 7 8; do
  nvidia-smi -i 0 --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader
  sleep 0.3
done
wait $PID
sudo -n nvidia-smi -i 0 -rgc
