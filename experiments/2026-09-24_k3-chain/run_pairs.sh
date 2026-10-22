#!/bin/zsh
# run ablate.py configs two at a time (10 MLPs each), nice 15, 3 torch threads each
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
cfgs=("$@")
for ((i=1; i<=${#cfgs}; i+=2)); do
  pids=()
  for c in ${cfgs[i]} ${cfgs[i+1]}; do
    [[ -z "$c" ]] && continue
    env PYTHONPATH="$ARC/src:." nice -n 15 uv run --project "$ARC" python ablate.py $c 10 > logs/ablate_${c}_n10.txt 2>&1 &
    pids+=($!)
  done
  wait $pids
done
for c in "$@"; do grep -h "RESULT" logs/ablate_${c}_n10.txt; done | sed 's/(reference.*//'
