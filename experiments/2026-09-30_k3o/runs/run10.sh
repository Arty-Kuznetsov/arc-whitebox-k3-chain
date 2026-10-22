#!/bin/bash
# 10-MLP flopscope screen of the two matched thresholds (util is what the port measures; raw is noisy on 10)
cd "$(dirname "$0")/../../.."
for t in 25 225; do
  nice -n 10 uv run --project external/whest-starterkit whest run --estimator experiments/2026-09-30_k3o/runs/v_k3o_m${t}.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --n-mlps 10 \
    --runner local --max-threads 4 --format json > experiments/2026-09-30_k3o/runs/k3o_m${t}_n10.json 2> experiments/2026-09-30_k3o/runs/k3o_m${t}_n10.err
done
echo done
