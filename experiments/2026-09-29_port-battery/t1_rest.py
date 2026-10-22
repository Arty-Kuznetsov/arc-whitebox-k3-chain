"""The rest of the task-1 grid, reduced after (6,448) and (6,384) showed k = 1, 2, 3, 5 all equal to E1, and
(5,448) k = 1 = E1 (x1.0215, t1_band_n10_1.json): for (5,448) only k = 5 (the answer to "largest k within +1%"),
k = inf (the failure mode) and the QR projection. 10 MLPs.
    PB_THREADS=1 ../../.venv-torch/bin/python t1_rest.py <k: 5|inf|qr>"""
import json
import sys
from pathlib import Path

import torch

import pb

HERE = Path(__file__).resolve().parent
which = sys.argv[1]
E1 = json.loads((HERE.parent / "2026-09-29_band" / "e1_n10.json").read_text())
k, proj = {"5": (5, "ne"), "inf": (None, "ne"), "qr": (None, "qr")}[which]
tag = f"f32 (5,448) k={'inf' if k is None else k} {proj}"
cfg = dict(pb.K3M, bandz=dict(a0=5, r=448, k=k, dtype=torch.float32, proj=proj, shadow=True))
pb.run_arms([(tag, cfg)], HERE / f"t1_band_n10_rest_{which}.json", ref_per=E1["k3m base (2.3462e-08)"]["per_mlp"],
            keep_logs=True)
