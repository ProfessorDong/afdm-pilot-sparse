#!/bin/bash
# Final evaluation with the final receiver (retries, pilot-block polishing,
# aperture-aware resolution, residual-level re-acquisition, direct genie).
cd "$(dirname "$0")"
python3 run_sweep.py specs/e4_main.json > runs/e4_main.log 2>&1
python3 run_sweep.py specs/e5_frame.json > runs/e5_frame.log 2>&1
python3 run_sweep.py specs/e6_robust.json > runs/e6_robust.log 2>&1
echo ALLDONE > runs/queue3.done
