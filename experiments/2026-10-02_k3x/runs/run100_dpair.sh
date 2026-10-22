#!/bin/bash
# 100-MLP local score of k3x WITH "dpair" (wrapper), residual cap NOT enforced locally (shared Mac); the grader decides
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-02_k3x/runs
nice -n 10 uv run --project external/whest-starterkit whest run --estimator $R/v_k3x_dpair.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --no-residual-wall-time-limit --format json > $R/k3x_dpair_n100.json 2> $R/k3x_dpair_n100.err
echo "run100 exit=$? $(date +%H:%M:%S)"
