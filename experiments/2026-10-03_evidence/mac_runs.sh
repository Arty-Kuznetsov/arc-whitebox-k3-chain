#!/bin/bash
# The ablation arms of k3y on the Mac (the cloud's container stopped after 4 arms): mini MLPs 0-31, flopscope, timing
# gates off, results in runs_mac/ (Mac-vs-Mac ratios; the cloud's runs/ stay cloud-vs-cloud). Skips complete JSONs.
#   bash mac_runs.sh <arm> [<arm> ...]
cd "$(dirname "$0")"
mkdir -p runs_mac logs_mac
for name in "$@"; do
  out=runs_mac/${name}_n32.json
  python3 -c "import json,sys;d=json.load(open('$out'));sys.exit(0 if len(d['results']['per_mlp'])==32 else 1)" 2>/dev/null && continue
  nice -n 10 uv run --project ../../external/whest-starterkit whest run --estimator arms/v_$name.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --n-mlps 32 --runner local --max-threads 4 \
    --format json --no-residual-wall-time-limit --wall-time-limit 600 > $out.tmp 2> logs_mac/$name.err \
    && mv $out.tmp $out && echo "done $name $(date +%H:%M)" || echo "FAILED $name $(date +%H:%M)"
done
