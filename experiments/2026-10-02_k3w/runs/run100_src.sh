#!/bin/bash
# 100-MLP local score of src/estimator_k3w.py with its defaults, residual cap NOT enforced locally (the shared Mac's load
# would zero predictions and corrupt the score; residual is still measured, and the grader's telemetry decides)
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-02_k3w/runs
nice -n 10 uv run --project external/whest-starterkit whest run --estimator src/estimator_k3w.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --no-residual-wall-time-limit --format json > $R/k3w_n100.json 2> $R/k3w_n100.err
echo "run100 exit=$? $(date +%H:%M:%S)"
