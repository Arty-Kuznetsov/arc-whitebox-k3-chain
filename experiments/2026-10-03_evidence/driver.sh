#!/bin/bash
# cloud #12 driver (idempotent; rerun after a container restart): gate -> B (bg) + heavy A arms solo -> rest of A x2
cd "$(dirname "$0")"
bash run_arm.sh base 3 3                                   # gate (and the download, once)
grep -q "B DONE" logs/B.txt 2>/dev/null || (PB_THREADS=1 nohup ../../.venv-torch/bin/python b_ceiling.py >> logs/B.txt 2>&1 &)
for a in base band_off cuts_off; do bash run_arm.sh $a; done   # one at a time while B runs (RSS)
until grep -q "B DONE" logs/B.txt; do sleep 60; done
Q=logs/queue.txt; [ -f $Q ] || tr ' ' '\n' < arms/ORDER.txt | grep -v "^base$\|^band_off$\|^cuts_off$" > $Q
worker() { while true; do a=$(flock $Q.lock bash -c "l=\$(head -n1 $Q); [ -n \"\$l\" ] && sed -i 1d $Q; echo \$l"); [ -z "$a" ] && break; bash run_arm.sh $a; done; }
worker & worker & wait
echo "DRIVER DONE" >> logs/driver.txt
