#!/bin/bash
# authoritative 100-MLP local score of one k3o variant:  run100.sh 25|225
cd "$(dirname "$0")/../../.."
t=$1
nice -n 10 uv run --project external/whest-starterkit whest run --estimator experiments/2026-09-30_k3o/runs/v_k3o_m${t}.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --format json > experiments/2026-09-30_k3o/runs/k3o_m${t}_n100.json 2> experiments/2026-09-30_k3o/runs/k3o_m${t}_n100.err
echo done $t
