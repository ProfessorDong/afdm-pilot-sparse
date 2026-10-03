#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f '^python3 theory_check.py' >/dev/null; do sleep 20; done
python3 run_sweep.py specs/tune_sp_ext.json > runs/tune_sp_ext.log 2>&1
python3 pick_sp.py > runs/pick_sp.log 2>&1
python3 run_sweep.py specs/e4_main.json > runs/e4_main.log 2>&1
python3 run_sweep.py specs/e5_frame.json > runs/e5_frame.log 2>&1
python3 run_sweep.py specs/e6_robust.json > runs/e6_robust.log 2>&1
echo ALLDONE > runs/queue.done
