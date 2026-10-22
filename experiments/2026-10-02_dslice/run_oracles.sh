#!/bin/bash
# four D-slice oracles, 1e7 samples each (MLPs 0 and 1, seeds 3101 and 3102), sequential, 3 torch threads, nice
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
for m in 0 1; do for s in 3101 3102; do
  env PYTHONPATH="$ARC/src" nice -n 10 uv run --project "$ARC" python d_oracle.py $m 1e7 $s 3 > logs_d_${m}_s${s}.txt 2>&1
  echo "oracle m=$m s=$s exit=$? $(date +%H:%M:%S)"
done; done
