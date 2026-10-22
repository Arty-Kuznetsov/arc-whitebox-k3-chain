#!/bin/bash
# Task 3 driver, per MLP: waits for both oracles and for task 2's arms of that MLP, then spectrum, the gain model, who,
# and the rank-r / sparse / gain payoff arms at the brief's layers (s4 = 5, 8, 11, 14). PB_THREADS=1. Idempotent.
cd "$(dirname "$0")"
PY=../../.venv-torch/bin/python
export PB_THREADS="${PB_THREADS:-1}"
for m in ${MLPS:-0 1}; do
  until [ -f res2/arm_${m}_all_all_B.json ]; do sleep 30; done
  [ -f res3/spec_$m.json ] || $PY t3.py spec $m
  [ -f res3/gain_$m.json ] || $PY t3.py gain $m
  [ -f res3/who_$m.json ] || $PY t3.py who $m
  $PY t3.py gainarm $m s4 P
  for r in 1 4 16 64 256; do $PY t3.py rank $m s4 $r P; done
  for q in 0.01 0.05 0.2; do for sel in A B; do $PY t3.py sparse $m s4 $q $sel; done; done
done
echo "T3 DONE"
