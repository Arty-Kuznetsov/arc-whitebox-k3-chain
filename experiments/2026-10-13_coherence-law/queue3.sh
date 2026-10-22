#!/bin/zsh
# after queue2: the error anatomy of two production estimators on the ten public networks with 1e9 truth
cd "$(dirname "$0")/../.."
until grep -q "QUEUE2 DONE" experiments/2026-10-13_coherence-law/res/queue2.log 2>/dev/null; do sleep 30; done
for est in external/whest-starterkit/examples/03_covariance_propagation.py src/estimator_k3z.py; do
  date; echo "== anatomy $est"
  nice -n 10 uv run --project external/whest-starterkit python scripts/error_anatomy.py $est 0-9 2>&1 | grep -v "Warning\|^  "
done
date; echo QUEUE3 DONE
