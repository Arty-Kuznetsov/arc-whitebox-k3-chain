#!/bin/bash
# 10-MLP flopscope screen of k3p band configurations:  run10.sh 6_448_rr 5_448_rr ...
cd "$(dirname "$0")/../../.."
for c in "$@"; do
  nice -n 10 uv run --project external/whest-starterkit whest run --estimator experiments/2026-09-30_k3p/runs/v_k3p_${c}.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --n-mlps 10 \
    --runner local --max-threads 4 --format json > experiments/2026-09-30_k3p/runs/k3p_${c}_n10.json 2> experiments/2026-09-30_k3p/runs/k3p_${c}_n10.err
done
echo done
