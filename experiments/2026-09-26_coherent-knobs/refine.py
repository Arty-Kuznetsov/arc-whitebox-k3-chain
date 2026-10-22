"""Refine the two knobs that moved in scan.py, then test the joint setting. 10 baked MLPs.

scan.py (10 MLPs, base 2.3135e-08): feedmult 0.8 -> x1.0058 (better on 7/10), k22c 0.6 -> x1.0014 (8/10),
sdmult 0.75 -> x1.0011 (6/10), k4scale 0.85 -> x1.0010 (6/10). All four are scalar multipliers on terms the
chain already computes, so they cost ZERO extra FLOPs -- any gain is free on the score. This finer grid finds
each optimum and then checks whether they compose.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-26_coherent-knobs/refine.py

Self-check: the base arm must print 2.3135e-08.
"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)

gt_all = np.load(BAKED / "gt.npy")
data = []
for m in range(len(gt_all)):
    W = np.load(BAKED / f"weights_{m}.npy")
    data.append(([torch.tensor(w, dtype=torch.float64) for w in W], torch.tensor(gt_all[m][-1])))


def arm(**kw):
    cfg = dict(K3L); cfg.update(kw)
    per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    return float(np.mean(per)), per


t0 = time.time()
base, bper = arm()
print(f"base: {base:.4e}  (must be 2.3135e-08)", flush=True)
out = {"base": base, "arms": {}}


def show(tag, **kw):
    r, per = arm(**kw)
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out["arms"][tag] = {"kw": {k: v for k, v in kw.items()}, "raw": r, "x": base / r, "better": b, "per_mlp": per}
    print(f"  {tag:42s} {r:.4e}   x{base/r:.4f}   better on {b}/10", flush=True)
    (OUT / "refine_n10.json").write_text(json.dumps(out, indent=1))


print("\nfeedmult (the kappa4 -> kappa3 P2 feed level):", flush=True)
for v in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
    show(f"feedmult={v}", feedmult=v)
print("\nk22c (the 2*C0^2 part of the (2,2) closure pattern):", flush=True)
for v in (0.0, 0.2, 0.4, 0.6):
    show(f"k22c={v}", k22c=v)
print("\njoint settings:", flush=True)
BEST_F = 0.8
show("feed+k22c", feedmult=BEST_F, k22c=0.6)
show("feed+k22c+sd", feedmult=BEST_F, k22c=0.6, sdmult=0.75)
show("feed+k22c+sd+k4s", feedmult=BEST_F, k22c=0.6, sdmult=0.75, k4scale=0.85)
show("feed+sd", feedmult=BEST_F, sdmult=0.75)
show("feed+k4s", feedmult=BEST_F, k4scale=0.85)
print(f"\ntotal {time.time()-t0:.0f}s")
