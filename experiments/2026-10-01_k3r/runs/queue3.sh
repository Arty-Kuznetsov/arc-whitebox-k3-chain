#!/bin/bash
# after the 100-MLP run: bake the two MLP-1 MC oracles (1e7, two seeds) so cloud #10 has a second MLP
cd "$(dirname "$0")/../../.."
R=experiments/2026-10-01_k3r/runs
until [ -s $R/k3r_n100.json ]; do sleep 20; done
sleep 20
cd experiments/2026-09-24_k3-chain
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
for seed in "" "777"; do
  echo "=== mc_oracle 1 1e7 ${seed:-default} $(date +%H:%M:%S)"
  env PYTHONPATH="$ARC/src:." nice -n 10 uv run --project "$ARC" python mc_oracle.py 1 1e7 $seed \
    > logs/mc_oracle_1${seed:+_s$seed}.txt 2>&1
  echo "    exit=$? $(date +%H:%M:%S)"
done
echo QUEUE3DONE
