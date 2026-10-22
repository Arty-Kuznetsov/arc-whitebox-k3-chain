#!/bin/bash
# authoritative 100-MLP local score of src/estimator_k3t.py with its defaults (the file that gets packaged)
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-01_l5t/runs
nice -n 10 uv run --project external/whest-starterkit whest run --estimator src/estimator_k3t.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --format json > $R/k3t_n100.json 2> $R/k3t_n100.err
echo "run100 exit=$? $(date +%H:%M:%S)"
