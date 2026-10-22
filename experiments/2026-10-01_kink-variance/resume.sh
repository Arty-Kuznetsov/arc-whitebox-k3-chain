#!/bin/bash
# Idempotent restart after a container restart: re-queue the jobs that were running (started, not done), restart the
# queue workers, resume the MC oracle chains from their checkpoints. Does nothing for what is still alive.
cd "$(dirname "$0")"
PY=../../.venv-torch/bin/python
NW="${1:-3}"
if ! pgrep -f "[q]ueue.sh jobs1.txt" > /dev/null; then
  if [ -f jobs1.txt.done ]; then
    python3 - << 'PYEOF'
import re
started, done = [], set()
for ln in open("jobs1.txt.done"):
    m = re.match(r"\[.*\] worker \d+ (start|done \(\d+\)): (.*)$", ln.rstrip("\n"))
    if m:
        (started.append(m.group(2)) if m.group(1) == "start" else done.add(m.group(2)))
lost = [j for j in started if j not in done]
rest = open("jobs1.txt").read().splitlines()
head = [j for j in lost if j not in rest]
open("jobs1.txt", "w").write("\n".join(head + rest) + ("\n" if head + rest else ""))
print("re-queued:", head)
PYEOF
    mv jobs1.txt.done "logs/jobs1_$(date -u +%H%M%S).done"
  fi
  [ -s jobs1.txt ] && { nohup bash queue.sh jobs1.txt "$NW" > logs/queue_$(date -u +%H%M%S).out 2>&1 & echo "queue restarted ($NW workers)"; }
fi
