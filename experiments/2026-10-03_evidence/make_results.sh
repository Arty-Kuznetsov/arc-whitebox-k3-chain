#!/bin/bash
# regenerate RESULTS.md (head + tables + reading) and push; called after every finished arm
cd "$(dirname "$0")"
T=$(../../.venv-torch/bin/python tables.py)
{ sed -n '1,/^## B\./p' results_head.md | sed '$d'
  echo "$T" | sed -n '1,/^$/p'
  sed -n '/^## B\./,$p' results_head.md; echo
  echo "$T" | sed -n '/torch, 10 baked/,$p'; echo
  cat results_tail.md; } > RESULTS.md
cd ../.. && git add -A experiments/2026-10-03_evidence && git commit -q -m "cloud #12: results update ($(ls experiments/2026-10-03_evidence/runs/*_n32.json | wc -l) arms done)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XAoUrsDXhs8XU5CFf4yoTh" && git push -q origin claude/laughing-hopper-61l0bn
