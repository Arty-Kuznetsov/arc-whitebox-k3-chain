#!/bin/zsh
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
while pgrep -f "ablate.py J3" >/dev/null || ! grep -q RESULT logs/ablate_J3.txt 2>/dev/null; do sleep 10; done
for c in tk_y1_r128 tk_y2_r128 tk_y1_r64 tk_y2_r256 tk_y2_r64 tk_y3_r128 tk_y1_r32; do
  env PYTHONPATH="$ARC/src:." nice -n 10 uv run --project "$ARC" python ablate.py $c 3 2>&1 | grep -v "Warning\|warn" > logs/ablate_$c.txt
done
echo ALLDONE > logs/tucker_done.txt
