#!/bin/bash
# Minimal job queue for the port battery: N workers, each pops the next line of a job file and runs it
# (PB_THREADS=1 per job, so N workers on N cores). Usage: bash queue.sh <jobfile> <nworkers>
# Each job line is a shell command run from this directory; its output goes to logs/<sanitised line>.txt.
cd "$(dirname "$0")"
JOBS="$1"; NW="${2:-4}"
LOCK="$JOBS.lock"
worker() {
  while true; do
    job=$(flock "$LOCK" bash -c 'line=$(head -n 1 "'"$JOBS"'"); [ -n "$line" ] && sed -i 1d "'"$JOBS"'"; echo "$line"')
    [ -z "$job" ] && break
    name=$(echo "$job" | sed "s/[^A-Za-z0-9_.-]/_/g" | rev | cut -c1-60 | rev)_$(echo "$job" | md5sum | cut -c1-6)
    echo "[$(date -u +%H:%M:%S)] worker $1 start: $job" >> "$JOBS.done"
    PB_THREADS=1 bash -c "$job" > "logs/q_$name.txt" 2>&1
    echo "[$(date -u +%H:%M:%S)] worker $1 done ($?): $job" >> "$JOBS.done"
  done
}
for w in $(seq 1 "$NW"); do worker "$w" & done
wait
echo "[$(date -u +%H:%M:%S)] queue empty" >> "$JOBS.done"
