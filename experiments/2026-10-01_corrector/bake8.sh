#!/bin/bash
# Oracles for MLPs 2-9 (one seed each, 2e6 samples): the cross-MLP test of the kappa4 pattern needs more than
# two MLPs. For TRAINING a corrector the per-neuron oracle noise averages over 1024 neurons per layer; at 2e6 the
# late-layer kappa4 noise ceiling is still ~0.95 (it is 0.99 at 1e7). ~3.2 min each.
cd "$(dirname "$0")/../2026-09-24_k3-chain"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
for i in 2 3 4 5 6 7 8 9; do
  echo "=== oracle $i $(date +%H:%M:%S)"
  env PYTHONPATH="$ARC/src:." nice -n 15 uv run --project "$ARC" python mc_oracle.py $i 2e6 > logs/mc_oracle_${i}_2e6.txt 2>&1
  echo "    exit=$? $(date +%H:%M:%S)"
done
echo BAKE8DONE
