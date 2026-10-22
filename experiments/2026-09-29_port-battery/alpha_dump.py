"""Per-layer alpha = mu_pre / sigma_pre of the chain on the 10 baked MLPs, for the cost model (cost_pb.py).

Writes alpha_k3m.npy (10 x 16 x 1024, gitignored) and alpha_fractions.json (committed): for every layer l the
fraction of neurons with alpha_l < t (dead side, t = -4.00 .. -1.00 step 0.25) and alpha_l > th (saturated side,
th = 1.50 .. 4.00 step 0.25), per MLP and pooled. The column cut uses the dead fraction at the BIRTH layer.

    PB_THREADS=3 ../../.venv-torch/bin/python alpha_dump.py [config: k3m | m225]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import pb

HERE = Path(__file__).resolve().parent
which = sys.argv[1] if len(sys.argv) > 1 else "k3m"
cfg = pb.K3M if which == "k3m" else pb.matched(-2.25)
A = []
for Ws, _ in pb.data(10):
    _, info = pb.run_chain(Ws, record=True, **cfg)
    A.append(np.stack([(r["mu_pre"] / r["var"].clamp_min(1e-30).sqrt()).numpy() for r in info]))
A = np.stack(A)                      # (10, 16, 1024)
np.save(HERE / f"alpha_{which}.npy", A)
dead_t = [round(-4.0 + 0.25 * i, 2) for i in range(13)]
sat_t = [round(1.5 + 0.25 * i, 2) for i in range(11)]
out = {"config": which, "shape": list(A.shape),
       "dead": {f"{t:.2f}": (A < t).mean(axis=(0, 2)).tolist() for t in dead_t},
       "sat": {f"{t:.2f}": (A > t).mean(axis=(0, 2)).tolist() for t in sat_t},
       "dead_per_mlp": {f"{t:.2f}": (A < t).mean(axis=2).tolist() for t in dead_t},
       "sat_per_mlp": {f"{t:.2f}": (A > t).mean(axis=2).tolist() for t in sat_t}}
(HERE / f"alpha_fractions_{which}.json").write_text(json.dumps(out, indent=1))
for t in (-3.5, -3.0, -2.5, -2.25, -2.0, -1.5):
    print(f"dead < {t:5.2f}:", " ".join(f"{x:.3f}" for x in out["dead"][f"{t:.2f}"]))
for t in (3.5, 3.0, 2.5):
    print(f"sat  > {t:5.2f}:", " ".join(f"{x:.3f}" for x in out["sat"][f"{t:.2f}"]))
