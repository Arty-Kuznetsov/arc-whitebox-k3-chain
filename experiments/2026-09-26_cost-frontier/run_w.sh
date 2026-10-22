#!/bin/bash
cd "$(dirname "$0")"
PYTHONPATH=. exec ../../.venv-torch/bin/python sweep.py "$@"
