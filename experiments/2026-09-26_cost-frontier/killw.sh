#!/bin/bash
pgrep -f "python sweep.py" | xargs -r kill
