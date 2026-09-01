#!/bin/bash
set -uo pipefail
srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --time=00:20:00 bash -c '
source ~/hyperion-venv/bin/activate
python3 /home/latorresn/hyperion/docs/justifications/scripts/extend_ncu_lud_convergence.py
'
echo EXTEND_SCRIPT_DONE
