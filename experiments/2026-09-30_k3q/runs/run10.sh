#!/bin/bash
# 10-MLP flopscope screen of k3q saturation thresholds:  run10.sh s30 s25 s20
cd "$(dirname "$0")/../../.."
for c in "$@"; do
  nice -n 10 uv run --project external/whest-starterkit whest run --estimator experiments/2026-09-30_k3q/runs/v_k3q_${c}.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --n-mlps 10 \
    --runner local --max-threads 4 --format json > experiments/2026-09-30_k3q/runs/k3q_${c}_n10.json 2> experiments/2026-09-30_k3q/runs/k3q_${c}_n10.err
done
echo done
