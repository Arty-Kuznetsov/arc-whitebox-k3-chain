#!/bin/bash
# after a container restart: resume the sequential arms and the auto-commit
cd "$(dirname "$0")"
pgrep -f "[d]river2.sh" >/dev/null || nohup bash driver2.sh >> logs/driver.txt 2>&1 &
pgrep -f "[a]utocommit.sh" >/dev/null || nohup bash autocommit.sh > logs/autocommit.txt 2>&1 &
