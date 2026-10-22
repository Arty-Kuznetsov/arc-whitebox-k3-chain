#!/bin/zsh
# One worker of the R1 factorial: runs r1fact.py for each "n:seed" job given as arguments, in order (resumable).
#   zsh run_fact.sh 256:0 256:3 128:0 ...   (log: logs/fact_worker_<first job>.txt)
cd "$(dirname "$0")"
ARC=$(cd ../../external/mlp_cumulant_propagation && pwd)
mkdir -p logs
for job in "$@"; do
  n=${job%%:*}; s=${job##*:}
  env PYTHONPATH="$ARC/src" nice -n 2 uv run --project "$ARC" python r1fact.py "$n" 8 "$s" 1 all \
    > "logs/fact_n${n}_s${s}.txt" 2>&1
  echo "$(date +%H:%M:%S) done n=$n seed=$s" >> logs/fact_progress.txt
done
