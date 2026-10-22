#!/bin/zsh
# Re-run the measurement scripts with logs kept (fast -> slow), so the write-up figures
# can be generated from logged outputs. Run from experiments/2026-09-16_kprop-arc/.
set -u
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
run() {  # run <script.py> [args...]
  local s="$1"; shift
  local name="${s%.py}"
  echo "=== $(date '+%F %T') START $s $*" | tee -a logs/_runner.txt
  env PYTHONPATH="$ARC/src" uv run --project "$ARC" python "$s" "$@" 2>&1 \
    | grep -vi "SyntaxWarning\|escape sequence\|it/s\]" | tee "logs/${name}.txt"
  echo "=== $(date '+%F %T') END $s (exit ${pipestatus[1]})" | tee -a logs/_runner.txt
}
run kink_diag.py
run kprop_cost.py
run flop_breakdown.py
run rank_diag.py
run kprop_1024.py 5 1,2,3
run rcap_test.py
run layer_adaptive.py
run reverse_adaptive.py
echo "=== $(date '+%F %T') ALL DONE" | tee -a logs/_runner.txt
