#!/bin/bash
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-01_k4corr/runs
nice -n 10 uv run --project external/whest-starterkit whest run --estimator src/estimator_k3v.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --format json > $R/k3v_n100.json 2> $R/k3v_n100.err
echo "run100 exit=$? $(date +%H:%M:%S)"
