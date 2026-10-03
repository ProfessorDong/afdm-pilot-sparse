#!/bin/bash
# Runs the experiment queue sequentially after the SP tuning sweep finishes.
cd "$(dirname "$0")"
python3 run_sweep.py specs/tune_sp.json > runs/tune_sp.log 2>&1
python3 pick_sp.py > runs/pick_sp.log 2>&1
python3 theory_check.py e2 e3 > runs/theory_check.log 2>&1
python3 run_sweep.py specs/e4_main.json > runs/e4_main.log 2>&1
python3 run_sweep.py specs/e5_frame.json > runs/e5_frame.log 2>&1
python3 run_sweep.py specs/e6_robust.json > runs/e6_robust.log 2>&1
echo ALLDONE > runs/queue.done
