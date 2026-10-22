#!/bin/zsh
# chain side + comparison; needs res/mc_0_s1001.npz and res/mc_0_s1002.npz
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
env PYTHONPATH="$ARC/src:../2026-09-24_k3-chain" nice -n 10 uv run --project "$ARC" python randdir.py "${1:-0}" 2>&1 | grep -v Warning
