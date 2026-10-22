#!/bin/zsh
# after queue3: the age-truncated k = 4 in ARC's representation at four shapes
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
until grep -q "QUEUE3 DONE" ../2026-10-13_coherence-law/res/queue3.log 2>/dev/null; do sleep 30; done
for cfg in "128 8 0 8" "256 8 0 3" "128 16 0 4" "256 16 0 3"; do
  date; echo "== k4age $cfg"
  env PYTHONPATH="$ARC/src:../2026-10-04_r1" nice -n 10 uv run --project "$ARC" python k4age.py ${=cfg} 1,2,3 2>&1 | grep -v Warning
done
date; echo QUEUE4 DONE
