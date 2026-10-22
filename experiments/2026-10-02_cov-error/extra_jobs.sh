#!/bin/bash
# Extra arms (after the main drivers): B on s4 and the reset noise control R (+R0 self-check) for both MLPs.
cd "$(dirname "$0")"
PY=../../.venv-torch/bin/python
export PB_THREADS=1
for m in 0 1; do
  until [ -f res2/arm_${m}_all_all_B.json ]; do sleep 30; done
  for part in off all; do $PY t2.py arm $m s4 $part B; done
  for v in R0 R; do for set in s4 all; do $PY t2.py arm $m $set off $v; done; done
done
echo "EXTRA DONE"
