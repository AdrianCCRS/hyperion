#!/bin/bash
set -uo pipefail
srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:40:00 bash -c '
source ~/hyperion-venv/bin/activate
python3 /home/latorresn/hyperion/docs/justifications/scripts/sweep_ncu_launch_count.py
'
echo SWEEP_SCRIPT_DONE
