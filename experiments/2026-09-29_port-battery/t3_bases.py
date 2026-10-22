"""The two bases of the battery, computed once: k3m (gate: must be 2.3462e-08) and k3m + matched cut -2.25
(experiments/2026-09-29_colcut/pairs.py: 2.4151e-08).   PB_THREADS=1 ../../.venv-torch/bin/python t3_bases.py <k3m|m225>"""
import sys
from pathlib import Path

import pb

HERE = Path(__file__).resolve().parent
which = sys.argv[1]
cfg = pb.K3M if which == "k3m" else pb.matched(-2.25)
pb.run_arms([(f"{which} base", dict(cfg))], HERE / f"t3_base_{which}_n10.json")
