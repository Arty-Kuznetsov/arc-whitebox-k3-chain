#!/bin/bash
# sequential: one whest process at a time (each holds ~8.3 GB RSS incl. the dataset; two OOM the 15.7 GB box)
cd "$(dirname "$0")"
for a in $(cat arms/ORDER.txt); do bash run_arm.sh $a 32 4; done
echo "DRIVER2 DONE" >> logs/driver.txt
