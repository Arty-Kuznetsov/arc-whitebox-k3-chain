#!/bin/bash
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-01_l5t/runs
for arm in "$@"; do
  echo "=== $arm $(date +%H:%M:%S)"
  nice -n 10 uv run --project external/whest-starterkit whest run --estimator $R/v_$arm.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
    --runner local --max-threads 4 --n-mlps 10 --format json > $R/${arm}_n10.json 2> $R/${arm}_n10.err
  echo "    exit=$? $(date +%H:%M:%S)"
done
echo ALLDONE
