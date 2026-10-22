#!/bin/bash
# follow every result line of the battery (grid logs + all queue job logs, including ones that appear later)
cd "$(dirname "$0")"
files="logs/t1_band_0.txt logs/t1_band_1.txt"
for job in "../../.venv-torch/bin/python t1b_band.py 0 3" "../../.venv-torch/bin/python t1b_band.py 1 3" \
  "../../.venv-torch/bin/python t1b_band.py 2 3" \
  "../../.venv-torch/bin/python t3_battery.py e3 m225" "../../.venv-torch/bin/python t3_battery.py e2 m225" \
  "../../.venv-torch/bin/python t3_battery.py e4 m225" "../../.venv-torch/bin/python t3_battery.py e3 k3m" \
  "../../.venv-torch/bin/python t3_battery.py e2 k3m" "../../.venv-torch/bin/python t3_battery.py e4 k3m" \
  "../../.venv-torch/bin/python t4_stack.py t4_step1_n10.json -2.25 band_6_448_pre" \
  "../../.venv-torch/bin/python t4_stack.py t4_step1b_n10.json -2.25 band_5_448_pre" \
  "../../.venv-torch/bin/python t4_stack.py t4_step1c_n10.json -2.25 band_6_384_pre" $EXTRA_JOBS; do
  files="$files logs/q_$(echo "$job" | sed 's/[^A-Za-z0-9_.-]/_/g' | cut -c1-80).txt"
done
tail -n 0 -F $files jobs1.txt.done 2>/dev/null | grep --line-buffered -E "raw |FAILED|Traceback|Error|KILLED|survives|->|finished|empty"
