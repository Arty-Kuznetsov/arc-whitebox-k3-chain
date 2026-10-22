#!/bin/bash
# run one ablation arm on mini MLPs 0..N-1 (flopscope, timing gates off); skip if a complete JSON exists
cd "$(dirname "$0")"
name=$1; n=${2:-32}; th=${3:-2}; out=runs/${name}_n$n.json
../../.venv-torch/bin/python -c "import json,sys;d=json.load(open('$out'));sys.exit(0 if len(d['results']['per_mlp'])==$n else 1)" 2>/dev/null && exit 0
../../.venv-whest/bin/whest run --estimator arms/v_$name.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
  --split mini --n-mlps $n --runner local --max-threads $th --format json --no-residual-wall-time-limit \
  --wall-time-limit 600 > $out.tmp 2> logs/$name.err && mv $out.tmp $out || echo "FAILED $name" >> logs/failed.txt
