#!/bin/bash
# MLP 0 again, now with K22 (same seeds: the D and C entries are identical to the first bake, K22 is added)
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
for s in 3101 3102; do
  env PYTHONPATH="$ARC/src" nice -n 10 uv run --project "$ARC" python d_oracle.py 0 1e7 $s 3 > logs_d_0_s${s}_k22.txt 2>&1
  echo "oracle m=0 s=$s exit=$? $(date +%H:%M:%S)"
done
