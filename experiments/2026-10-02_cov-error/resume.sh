#!/bin/bash
# Idempotent restart after a container restart: the oracle worker (resumes from checkpoints) and the task-2 driver.
cd "$(dirname "$0")"
bash oracle_jobs.sh 3
if ! pgrep -f "[t]2_jobs.sh" > /dev/null && ! grep -q "T2 DONE" logs/t2_jobs.txt 2>/dev/null; then
  nohup bash t2_jobs.sh >> logs/t2_jobs.txt 2>&1 &
  echo "task-2 driver (re)started"
fi
if ! pgrep -f "[t]3_jobs.sh" > /dev/null && ! grep -q "T3 DONE" logs/t3_jobs.txt 2>/dev/null; then
  nohup bash t3_jobs.sh >> logs/t3_jobs.txt 2>&1 &
  echo "task-3 driver (re)started"
fi
if ! pgrep -f "[t]3b_jobs.sh" > /dev/null && ! grep -q "T3B DONE" logs/t3b_jobs.txt 2>/dev/null; then
  nohup bash ./t3b_jobs.sh >> logs/t3b_jobs.txt 2>&1 &
  echo "task-3b driver (re)started"
fi
if ! pgrep -f "[e]xtra_jobs.sh" > /dev/null && ! grep -q "EXTRA DONE" logs/extra_jobs.txt 2>/dev/null; then
  nohup bash ./extra_jobs.sh >> logs/extra_jobs.txt 2>&1 &
fi
if [ ! -f res3/featborn_0.json ] && ! pgrep -f "[f]eatborn" > /dev/null; then
  PB_THREADS=1 nohup ../../.venv-torch/bin/python t3.py featborn 0 > logs/featborn_0.txt 2>&1 &
fi
true
