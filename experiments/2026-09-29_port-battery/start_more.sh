#!/bin/bash
# wait until the task-1 grid has finished, then add 2 workers to the jobs1 queue
cd "$(dirname "$0")"
while pgrep -f "[t]1_band.py" > /dev/null; do sleep 20; done
echo "[$(date -u +%H:%M:%S)] grid finished, adding 2 workers" >> jobs1.txt.done
bash queue_more.sh jobs1.txt 2
