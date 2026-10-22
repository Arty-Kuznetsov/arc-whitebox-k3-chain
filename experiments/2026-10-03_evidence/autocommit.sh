#!/bin/bash
cd "$(dirname "$0")"; last=-1
while true; do
  k=$(ls runs/*_n32.json 2>/dev/null | wc -l)
  [ "$k" != "$last" ] && bash make_results.sh && last=$k
  grep -q "DRIVER2 DONE" logs/driver.txt && { bash make_results.sh; break; }
  sleep 120
done
