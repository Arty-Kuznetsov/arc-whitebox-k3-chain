#!/bin/bash
# Task 3, second driver: the born-vs-carried decomposition per MLP (after that MLP's task-3 arms), then the entrywise
# models (feat, both MLPs). PB_THREADS=1. Idempotent.
cd "$(dirname "$0")"
PY=../../.venv-torch/bin/python
export PB_THREADS="${PB_THREADS:-1}"
for m in 0 1; do
  until [ -f res3/sparse_${m}_s4_0.2_B.json ]; do sleep 30; done
  $PY t3.py born $m
done
[ -f res3/feat.json ] || $PY t3.py feat
echo "T3B DONE"
