#!/bin/bash
# 100-MLP local score of src/estimator_k3y.py with its defaults, residual cap NOT enforced locally (shared Mac)
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-02_csym/runs
nice -n 10 uv run --project external/whest-starterkit whest run --estimator src/estimator_k3y.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --no-residual-wall-time-limit --format json > $R/k3y_n100.json 2> $R/k3y_n100.err
echo "run100 exit=$? $(date +%H:%M:%S)"
