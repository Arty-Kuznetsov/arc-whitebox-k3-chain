#!/bin/zsh
# full data collection for the coherence law (one process at a time, nice, 3 threads)
cd "$(dirname "$0")"
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
R() { env PYTHONPATH="$ARC/src:../2026-09-24_k3-chain" nice -n 10 uv run --project "$ARC" python coh.py "$@" 2>&1 | grep -v Warning; }
date; R baked 0 10
date; R synth 256 8 0 10 2e6
date; R synth 256 16 0 10 2e6
date; R synth 256 32 0 10 2e6
date; R synth 512 16 0 6 4e6
date; echo ALL DONE
