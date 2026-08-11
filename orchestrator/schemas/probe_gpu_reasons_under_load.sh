#!/bin/bash
echo "=== LIMITES DE POTENCIA (nunca revisado antes) ==="
nvidia-smi -i 0 -q -d POWER
echo ""
echo "=== ENLACE PCIe ==="
nvidia-smi -i 0 -q | grep -A12 "GPU Link Info"
echo ""
echo "=== ahora con CARGA REAL: razones de evento de reloj ==="
cd /home/latorresn/hyperion-kernels
./bin/ert_probe_gpu fp64 &
PID=$!
sleep 1.0
sudo -n nvidia-smi -i $CUDA_VISIBLE_DEVICES -lgc 1005
echo "LGC_EXIT=$?"
for i in 1 2 3; do
  sleep 0.6
  echo "--- muestra $i (bajo carga) ---"
  nvidia-smi -i 0 --query-gpu=clocks.sm,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv,noheader
  nvidia-smi -i 0 -q -d PERFORMANCE | grep -A12 "Clocks Event Reasons"
done
wait $PID
sudo -n nvidia-smi -i $CUDA_VISIBLE_DEVICES -rgc
