"""Task 1 of the port battery: the transported band in FLOAT32 with a LAZY QR (chain.py bandz), 10 baked MLPs.

Production mirror (chain.py _band_step): old sources (age >= a0) are legs = P Z with one shared basis P transported
like a leg, fixed coefficients Z; P re-orthonormalised only every k layers (then Z <- R Z); the entrant projected once
by least squares on the current P -- proj="ne": Z = (P^T P)^-1 P^T X (the brief's normal equations), proj="qr":
Z = R^-1 Q^T X from a QR of P taken for the projection only (members never rotated). Band arithmetic in float32;
a float64 shadow of the same band gives the per-layer relative leg error (diagnostic, does not feed the chain).

Anchor: dtype=float64, k=1 must reproduce E1's transported (6,448) = 2.3624e-08 (per MLP, to 1e-4 relative).
Question: the largest k whose float32 raw stays within +1% of E1's float64 number (E1 per-band).

    cd experiments/2026-09-29_port-battery
    PB_THREADS=2 ../../.venv-torch/bin/python t1_band.py <part: 0|1|all>
"""
import json
import sys
from pathlib import Path

import torch

import pb

HERE = Path(__file__).resolve().parent
part = sys.argv[1] if len(sys.argv) > 1 else "all"
E1 = json.loads((HERE.parent / "2026-09-29_band" / "e1_n10.json").read_text())
E1_TAG = {(6, 384): "transported 6:384", (6, 448): "transported 6:448", (5, 448): "transported 5:448"}


def band(a0, r, k, dtype=torch.float32, proj="ne"):
    return dict(pb.K3M, bandz=dict(a0=a0, r=r, k=k, dtype=dtype, proj=proj, shadow=True))


arms = [("anchor f64 k=1 (6,448) ne", band(6, 448, 1, torch.float64))]
for (a0, r) in [(6, 448), (6, 384), (5, 448)]:
    for k in (1, 2, 3, 5, None):
        arms.append((f"f32 ({a0},{r}) k={'inf' if k is None else k} ne", band(a0, r, k)))
    arms.append((f"f32 ({a0},{r}) k=inf qr", band(a0, r, None, proj="qr")))
if part in ("0", "1"):
    arms = arms[int(part)::2]
ref_per = E1["k3m base (2.3462e-08)"]["per_mlp"]   # x and better-count against the k3m base, as in E1
pb.run_arms(arms, HERE / f"t1_band_n10_{part}.json", ref_per=ref_per, keep_logs=True)
