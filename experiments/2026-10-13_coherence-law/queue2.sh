#!/bin/zsh
# after run_all.sh: add the k = 2 estimator to every file, then the Proposition 3 test at three widths
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
until grep -q "ALL DONE" res/run_all.log 2>/dev/null; do sleep 20; done
date; echo "== k2 for all files"
for t in n1024_L16_baked n256_L8_net n256_L16_net n256_L32_net n512_L16_net; do
  env PYTHONPATH="$ARC/src:../2026-09-24_k3-chain" nice -n 10 uv run --project "$ARC" python coh_k2.py $t 2>&1 | grep -v Warning
done
date; echo "== prop3"
cd ../2026-10-13_prop3
for cfg in "64 16 3" "128 16 3" "256 16 2"; do
  env PYTHONPATH="$ARC/src:../2026-09-24_k3-chain" nice -n 10 uv run --project "$ARC" python prop3.py ${=cfg} 2>&1 | grep -v Warning | grep -E "net [0-9]+ done|layer (8|10|12|14) "
done
date; echo QUEUE2 DONE
