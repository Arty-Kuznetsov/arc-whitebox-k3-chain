#!/bin/bash
# follow every result line of every queue log (current and future): poll the logs directory
cd "$(dirname "$0")"
declare -A seen
while true; do
  for f in logs/q_*.txt jobs1.txt.done; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    o=${seen[$f]:-0}
    if [ "$n" -gt "$o" ]; then
      [ "$o" -gt 0 ] && tail -n $((n - o)) "$f" | grep -E "raw |FAILED|Traceback|Error|KILLED|survives|projected|empty"
      seen[$f]=$n
    fi
  done
  sleep 20
done
