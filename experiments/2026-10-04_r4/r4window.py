"""R4 (2026-10-04, the old tier): the leaders' old kappa3 tier costs ~nothing (504aldo, forum 18218, section 6), ours
(the band members: materialisation, compressed pairing, entry) ~37% of the FLOPs. Old content cannot be closed
wholesale (NEGATIVE 16, 39, 46, 62), but the amplitude-mode closure (positive homogeneity: t += alpha mu var,
D += alpha/3 (2 mu C0 + mu var)) was only ever tested with a SHORT window (3: the tail ~70% of |t|). Here: the
oldest sources only (window W = 8, 10, 12 sources kept), with and without the amplitude tail, on the production
accuracy configuration (research base B9 + marg2 + band 5/448 rr), 10 baked MLPs at 1024 x 16. One file per
(MLP, arm). Member-layers saved: W=12 -> 6/55, W=10 -> 15/55, W=8 -> 28/55 of the band's members.
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r4window.py <mlps comma> [arms comma]
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("R4_THREADS", "3")))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402

BAND = dict(bandz=dict(a0=5, r=448, k=None, dtype=torch.float32, proj="qr", entry="pre", init="rr"))
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),), **BAND)
ARMS = {"bbase": {}}
for W in (12, 10, 8, 6):
    ARMS[f"W{W}amp"] = dict(window=W, tail="amp")
    ARMS[f"W{W}"] = dict(window=W)
for y in (5, 7, 9, 11):   # H2: the old tier feeds t (the readout) but the (2,1) slice D only from the y newest sources
    ARMS[f"D{y}"] = dict(d21_ages=y)
for r_ in (384, 320, 256, 192, 128, 96, 64):   # H3: the members read through their top-r' SVD subspace (per-layer re-fit)
    ARMS[f"R{r_}"] = dict(bandz=dict(BAND["bandz"], readrank=r_))
mlps = [int(x) for x in sys.argv[1].split(",")]
sel = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bbase", "W12amp", "W12", "W10amp", "W10", "W8amp", "W8"]
gt = np.load(pb.BAKED / "gt.npy")
(HERE / "res").mkdir(exist_ok=True)
for m in mlps:
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{m}.npy")]
    for a in sel:
        f = HERE / "res" / f"m{m}_{a}.json"
        if f.exists():
            continue
        t0 = time.time()
        log = []
        cfg = copy.deepcopy(dict(BASE, **ARMS[a]))   # bandz carries a per-run log
        mu, _ = run_chain([W.clone() for W in Ws], tail_log=log, **cfg)
        e = float(((np.asarray(mu[-1]) - gt[m][-1]) ** 2).mean())
        bl = cfg.get("bandz", {}).get("log", [])
        en = [r_.get("readrank_energy") for r_ in bl if r_.get("readrank_energy") is not None]
        f.write_text(json.dumps({"mlp": m, "arm": a, "mse": e, "secs": time.time() - t0, "tail_log": log, "energy": en}))
        print(f"MLP {m}  {a:8s} {e:.4e}  [{time.time() - t0:.0f} s]", flush=True)
