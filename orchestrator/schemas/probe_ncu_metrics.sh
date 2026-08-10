#!/bin/bash
ncu --query-metrics > /tmp/ncu_all_metrics.txt 2>&1
echo "TOTAL_METRICS: $(wc -l < /tmp/ncu_all_metrics.txt)"
echo "--- tensor ---"
grep -iE "tensor" /tmp/ncu_all_metrics.txt
echo "--- dram bytes ---"
grep -iE "^dram__bytes" /tmp/ncu_all_metrics.txt
echo "--- dfma/ffma/dadd/fadd/dmul/fmul ---"
grep -iE "sass_thread_inst_executed_op_(d|f)(fma|add|mul)_pred_on" /tmp/ncu_all_metrics.txt
cp /tmp/ncu_all_metrics.txt /home/latorresn/yacacerest/ncu_all_metrics_2026.1.1.0.txt
