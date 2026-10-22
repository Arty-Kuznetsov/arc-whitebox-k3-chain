"""R3 step 1 (notes/PLAN-2026-10-04.md): does the young-kappa4 carrier (R2: x0.62-0.67 on 3 of 4 nets at n=256, L=8)
pay at the challenge's 1024 x 16?  Float64 research chain, research base B9 + marg2 (as R1/R2 and the 10-03 last-layer
runs), against the baked ground truth (experiments/2026-09-16_kprop-arc/baked). Carrier: chain.py k4young impl="legs"
(young4.py; gated against ARC's factored k=4 objects to 1e-14). Final-layer MSE per MLP; resumable.
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r3full.py <mlps comma> [arms comma]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("R3_THREADS", "3")))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402

BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
Y = dict(impl="legs", conv="arc", ages=2, types=("path", "star", "32"), resid=True, read=("K4d", "K31"))
ARMS = {"base": {}, "y2": dict(k4young=Y), "y2K4": dict(k4young=dict(Y, read=("K4d",))),
        "y1": dict(k4young=dict(Y, ages=1)), "y1K4": dict(k4young=dict(Y, ages=1, read=("K4d",))),
        "y3K4": dict(k4young=dict(Y, ages=3, read=("K4d",))),
        "y2c31": dict(k4young=dict(Y, k31_32="cheap")),     # K31: paths, stars + the cheap part of the 3+2
        "y2n31": dict(k4young=dict(Y, k31_32="none")),      # K31: paths and stars only
        "y1c31": dict(k4young=dict(Y, ages=1, k31_32="cheap"))}
ARMS.update({
    "y2K4pd": dict(k4young=dict(Y, read=("K4d",), types=("pathdiag", "star", "32"))),   # K4d only, the path's
    "y1K4pd": dict(k4young=dict(Y, ages=1, read=("K4d",), types=("pathdiag", "star", "32"))),   # n^2 piece
    "y2K4-32": dict(k4young=dict(Y, read=("K4d",), types=("32",))),
})
# the production accuracy configuration adds the transported band (k3p: a0 5, rank 448, rr init, float32, lazy QR)
BAND = dict(bandz=dict(a0=5, r=448, k=None, dtype=torch.float32, proj="qr", entry="pre", init="rr"))
for k_ in list(ARMS):
    ARMS["b" + k_] = dict(ARMS[k_], **BAND)
mlps = [int(x) for x in sys.argv[1].split(",")]
sel = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "y2K4", "y2", "y1K4", "y1"]
gt = np.load(pb.BAKED / "gt.npy")
ARMS.update({   # the production form (young4.Young4Chain), with the band
    "bCy2K4pd": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32")), **BAND),
    "bCy1K4pd": dict(k4young=dict(Y, impl="chain", ages=1, types=("pathdiag", "star", "32")), **BAND),
    "bCy2K4": dict(k4young=dict(Y, impl="chain", types=("path", "star", "32")), **BAND),
    "bCy2K4pd32a1": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), ages32=1), **BAND),
    "bCy2K4pd32a1nl": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), ages32=1, last32=False),
                           **BAND),
    "bCy2K4pdnl": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), last32=False), **BAND),
    "bCy2K4pdL": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), only_last32=True), **BAND),
    "bCy2K4pdL1": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), only_last32=True, ages32=1),
                       **BAND),
    "bCy2K4ps": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star")), **BAND),
    # 2026-10-06 (k4a port check): Dx as the production partners give it (no thin remainder at the snapshot, no q m^T)
    "bCy2K4pdnlP": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), last32=False, dx="prod"), **BAND),
    "bCy2K4pdnlPc": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), last32=False, dx="prod", rowcut=(-2.5, 2.5)), **BAND),
    "bCy2K4pdnlPc2": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), last32=False, dx="prod", rowcut=(-2.5, 2.5), cproj="mask"), **BAND),
    "bCy2K4pdnlPcA": dict(k4young=dict(Y, impl="chain", types=("pathdiag", "star", "32"), last32=False, dx="prod", rowcut=(-2.5, 2.5), rowcut_after=True), **BAND),
})
(HERE / "res" / "arm").mkdir(exist_ok=True)
for m in mlps:   # one file per (MLP, arm): parallel workers never overwrite each other
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{m}.npy")]
    for a in sel:
        f = HERE / "res" / "arm" / f"m{m}_{a}.json"
        if f.exists():
            continue
        t0 = time.time()
        mu, _ = run_chain([W.clone() for W in Ws], **dict(BASE, **ARMS[a]))
        e = float(((np.asarray(mu[-1]) - gt[m][-1]) ** 2).mean())
        f.write_text(json.dumps({"mlp": m, "arm": a, "mse": e, "secs": time.time() - t0}))
        print(f"MLP {m}  {a:6s} {e:.4e}  [{time.time() - t0:.0f} s]", flush=True)
