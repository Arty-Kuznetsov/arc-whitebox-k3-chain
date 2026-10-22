"""Task 1b: the band variants that the pricing says matter (10 MLPs, float32 band, QR projection, no rotation).

Task 1 found (t1_band.py): with the normal equations the band needs a re-orthonormalisation every <= 5 layers
(k = inf: x1.12, cond(P) ~ 1e4-1e5 and the r x r solve squares it); with the entrant projected through a QR of P
(proj="qr") no re-orthonormalisation and no coefficient rotation is needed at all. The pricing (cost_pb.py) then
says the next largest construction cost is the entrant's own dense transport at age a0 (E1 semantics, "post"):
entry="pre" projects it at the end of the previous layer onto the gated basis and saves that transport
(x0.930 -> x0.887 util at (6,448)). This measures what "pre" costs in accuracy, and the r = 512 points.

    PB_THREADS=1 ../../.venv-torch/bin/python t1b_band.py <part> <nparts>
"""
import json
import sys
from pathlib import Path

import torch

import pb

HERE = Path(__file__).resolve().parent
part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
E1 = json.loads((HERE.parent / "2026-09-29_band" / "e1_n10.json").read_text())


def band(a0, r, entry, base=pb.K3M, k=None, proj="qr", init="cols"):
    return dict(base, bandz=dict(a0=a0, r=r, k=k, dtype=torch.float32, proj=proj, entry=entry, shadow=True,
                                 init=init))


arms = []  # task 1c: the range-finder init (init="rr") on the frontier
for (a0, r) in [(5, 384), (4, 448), (4, 512), (5, 480)]:
    arms.append((f"f32 ({a0},{r}) k=inf qr pre init=rr", band(a0, r, "pre", init="rr")))
arms = arms[part::nparts]
ref_per = E1["k3m base (2.3462e-08)"]["per_mlp"]
pb.run_arms(arms, HERE / f"t1c_band_n10_{part}.json", ref_per=ref_per, keep_logs=True)
