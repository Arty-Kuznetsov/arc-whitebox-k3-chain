#!/bin/bash
# Task 2 driver: per MLP (0 first), wait for both covariance oracles, then rec -> split -> the payoff arms.
# PB_THREADS=1 (one core; the oracle worker has the other three). Idempotent: every step skips what exists.
cd "$(dirname "$0")"
PY=../../.venv-torch/bin/python
export PB_THREADS="${PB_THREADS:-1}"
for m in 0 1; do
  until [ -f oracles/cov_${m}_s2101_C.npy ] && [ -f oracles/cov_${m}_s2102_C.npy ]; do sleep 30; done
  $PY t2.py rec $m
  [ -f res2/split_$m.json ] || $PY t2.py split $m
  for set in s4 all; do
    for part in off diag all; do
      for var in P N; do $PY t2.py arm $m $set $part $var; done
    done
  done
  for part in off diag all; do for var in A B; do $PY t2.py arm $m all $part $var; done; done
done
echo "T2 DONE"
