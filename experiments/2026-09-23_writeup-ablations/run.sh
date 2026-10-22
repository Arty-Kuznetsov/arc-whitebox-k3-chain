#!/bin/zsh
# Two 100-MLP ablations for the write-up, run sequentially and niced (shared machine).
set -u
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
for v in k2 notetra; do
  echo "=== $(date '+%F %T') START $v"
  nice -n 10 uv run --project "$ROOT/external/whest-starterkit" whest run --estimator "v_$v.py" \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
    --runner local --max-threads 4 --format json > "report_$v.json" 2> "run_$v.log"
  echo "=== $(date '+%F %T') END $v (exit $?)"
done
