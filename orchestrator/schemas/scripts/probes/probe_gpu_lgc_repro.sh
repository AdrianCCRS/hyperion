#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "--- before ---"
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=clocks.sm,clocks.max.sm,clocks.min.gpc --format=csv
echo "--- lgc 210,210 ---"
sudo -n nvidia-smi -i $CUDA_VISIBLE_DEVICES -lgc 210,210
echo "LGC_EXIT=$?"
echo "--- immediately after ---"
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=clocks.sm,clocks.max.sm --format=csv
sleep 1
echo "--- after 1s ---"
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=clocks.sm,clocks.max.sm --format=csv
echo "--- supported clocks ---"
nvidia-smi -i $CUDA_VISIBLE_DEVICES -q -d SUPPORTED_CLOCKS | head -20
echo "--- rgc ---"
sudo -n nvidia-smi -i $CUDA_VISIBLE_DEVICES -rgc
echo "RGC_EXIT=$?"
