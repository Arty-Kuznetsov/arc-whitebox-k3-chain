#!/bin/bash
# authoritative 100-MLP local score of src/estimator_k3q.py with its defaults (the file that gets packaged)
cd "$(dirname "$0")/../../.."
nice -n 10 uv run --project external/whest-starterkit whest run --estimator src/estimator_k3q.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
  --runner local --max-threads 4 --format json > experiments/2026-09-30_k3q/runs/k3q_n100.json 2> experiments/2026-09-30_k3q/runs/k3q_n100.err
echo done
