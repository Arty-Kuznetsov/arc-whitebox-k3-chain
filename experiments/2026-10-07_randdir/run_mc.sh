#!/bin/zsh
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
for s in 1001 1002; do echo "== seed $s =="; date; nice -n 10 uv run --project "$ARC" python randdir_mc.py 0 1e7 $s 4,8,11,14 2>&1 | grep -v Warning; done; date
