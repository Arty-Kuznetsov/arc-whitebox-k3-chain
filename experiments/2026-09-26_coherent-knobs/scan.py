"""Coherent-knob re-scan on the current base (k3l/k3m), 10 baked MLPs.

Rationale (NEGATIVE §57): the readout sums COHERENTLY over neurons, so a correction that is smooth across
neurons is expensive when it is wrong and valuable when it is right. The variance direction is stiff
(MSE = 2.4e-8 + 5c², half-width ~1e-5) and therefore already optimal. The kappa4 / feed directions are SOFT
-- `k31mult` moved 1.0 -> 0.65 for x1.058 -- so their optima can have MOVED when k3i / k3j / k3l landed, and
each of them is a single number, free in production. This scans them one at a time around their current value.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-26_coherent-knobs/scan.py [n_mlps]

Self-check: the base arm must print 2.3135e-08 on 10 MLPs (2.4074e-08 on MLP 0 alone).
Every knob is a scalar multiplying an existing term, so nothing here changes the FLOP count: any gain is
free on the score. Tuned on 10 of OUR 100 baked MLPs -- the grader's public 50 are a disjoint set -- and a
winner must then be confirmed on all 100 through the flopscope path.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
NM = int(sys.argv[1]) if len(sys.argv) > 1 else 10

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)

GRID = {
    "k31mult":  [0.45, 0.55, 0.65, 0.75, 0.85],     # the (3,1) closure slice; 0.65 today
    "k22c":     [0.6, 0.8, 1.0, 1.2, 1.4],          # the 2*C0^2 part of the (2,2) closure pattern
    "sdmult":   [0.5, 0.75, 1.0, 1.25, 1.5],        # the star-diagonal kappa4 birth
    "feedmult": [0.6, 0.8, 1.0, 1.2, 1.4],          # the kappa4 -> kappa3 feed (P2) hub weight
    "k4scale":  [0.85, 0.925, 1.0, 1.075, 1.15],    # the level of the kappa4 births added each layer
}

gt_all = np.load(BAKED / "gt.npy")
Wss = []
for m in range(NM):
    W = np.load(BAKED / f"weights_{m}.npy")
    Wss.append(([torch.tensor(w, dtype=torch.float64) for w in W], torch.tensor(gt_all[m][-1])))


def arm(**kw):
    cfg = dict(K3L); cfg.update(kw)
    per = []
    for Ws, g in Wss:
        pred, _ = run_chain(Ws, **cfg)
        per.append(float(((pred[-1] - g) ** 2).mean()))
    return float(np.mean(per)), per


t0 = time.time()
base, base_per = arm()
print(f"base ({NM} MLPs): {base:.4e}   (10 MLPs must be 2.3135e-08)   [{time.time()-t0:.0f}s]", flush=True)
out = {"base": base, "base_per": base_per, "grid": {}}
for knob, vals in GRID.items():
    print(f"\n{knob}:", flush=True)
    out["grid"][knob] = {}
    for v in vals:
        if v == K3L.get(knob, 1.0):
            r, per = base, base_per
        else:
            r, per = arm(**{knob: v})
        better = sum(1 for a, b in zip(per, base_per) if a < b)
        out["grid"][knob][str(v)] = {"raw": r, "per_mlp": per, "better": better}
        print(f"   {v:<7} {r:.4e}   x{base/r:.4f} of base   better on {better}/{NM} MLPs"
              f"{'   <- current' if v == K3L.get(knob, 1.0) else ''}", flush=True)
    (OUT / f"scan_n{NM}.json").write_text(json.dumps(out, indent=1))
print(f"\ntotal {time.time()-t0:.0f}s -> {OUT / f'scan_n{NM}.json'}")
