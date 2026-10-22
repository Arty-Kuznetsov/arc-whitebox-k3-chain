#!/bin/bash
# runs after screen.sh: the honest L5 test (10 MLPs through the grader runner, then RSS on one 1024x16)
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-01_k3r/runs
until grep -q ALLDONE $R/screen.log; do sleep 15; done
echo "=== L5real 10 MLPs $(date +%H:%M:%S)"
nice -n 10 uv run --project external/whest-starterkit whest run --estimator $R/v_k3r_L5real.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --n-mlps 10 --format json > $R/k3r_L5real_n10.json 2> $R/k3r_L5real_n10.err
echo "    exit=$? $(date +%H:%M:%S)"
echo "=== L5 RSS + residual on one 1024x16 $(date +%H:%M:%S)"
nice -n 10 /usr/bin/time -l uv run --project external/whest-starterkit python experiments/2026-10-01_k3r/k3r_check.py l5 \
  > $R/l5_rss.txt 2> $R/l5_rss.err
echo "    exit=$? $(date +%H:%M:%S)"
echo QUEUE2DONE
