#!/bin/zsh
# Queued after run.sh: gain-only k=2 on 100 MLPs (the kit's covariance propagation, same pipeline).
set -u
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
while pgrep -f "whest run --estimator v_notetra.py" > /dev/null; do sleep 20; done
for v in k2gain; do
  echo "=== $(date '+%F %T') START $v"
  nice -n 10 uv run --project "$ROOT/external/whest-starterkit" whest run --estimator "v_$v.py" \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
    --runner local --max-threads 4 --format json > "report_$v.json" 2> "run_$v.log"
  echo "=== $(date '+%F %T') END $v (exit $?)"
done
