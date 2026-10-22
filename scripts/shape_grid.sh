#!/bin/bash
# Shape grid before every package: the grader's smoke test runs a shape other than 1024 x 16 (probe 334476 died there,
# 2026-10-07). Runs the estimator in-process over widths 16..MAXW x depths 1..16 x budgets {2^41, scaled} and lists the
# failures. Expected: only k3z's own tiny-shape set (widths 16-64 at depth >= 12); anything else is a shipping bug.
# usage: bash scripts/shape_grid.sh src/estimator_X.py [MAXW=768]   (~4 min at 768 under the machine policy)
set -u
EST="$1"; MAXW="${2:-768}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$(mktemp -t shapegrid)"
nice -n 10 uv run --project "$ROOT/external/whest-starterkit" python "$ROOT/experiments/2026-10-06_k4a/shape_grid.py" "$EST" "$MAXW" 2>&1 | grep -v "Warning\|^  " | tee "$LOG" | grep -E "^(FAIL)" 
echo "---- $(grep -c '^OK' "$LOG") OK, $(grep -c '^FAIL' "$LOG") FAIL (full log: $LOG)"
echo "k3z's own tiny-shape failures (acceptable): w=16 d=12/16, w=32 d=16, w=64 d=16 (both budgets)."
