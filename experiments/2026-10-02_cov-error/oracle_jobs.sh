#!/bin/bash
# The covariance oracles of task 1, one worker, in order: MLP 0 seeds 2101, 2102, then MLP 1 seeds 2101, 2102
# (3 threads; one core stays free for chain runs at PB_THREADS=1). Idempotent: a finished oracle (its _C.npy exists)
# is skipped; an interrupted one resumes from its checkpoint on the identical random stream.
cd "$(dirname "$0")"
PY=../../.venv-torch/bin/python
TH="${1:-3}"
if pgrep -f "[o]racle_worker" > /dev/null; then echo "oracle worker alive"; exit 0; fi
nohup bash -c "exec -a oracle_worker bash -c 'for j in \"0 2101\" \"0 2102\" \"1 2101\" \"1 2102\"; do set -- \$j; [ -f oracles/cov_\$1_s\$2_C.npy ] || $PY cov_oracle.py \$1 1e7 \$2 $TH >> logs/oracle_\$1_s\$2.txt 2>&1; done'" > /dev/null 2>&1 &
echo "started oracle worker ($TH threads)"
