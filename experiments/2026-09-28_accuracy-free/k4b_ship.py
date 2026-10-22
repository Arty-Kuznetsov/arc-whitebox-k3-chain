"""Is the kappa4 birth DEPTH free on the shippable base? 10 MLPs.

`k4b` loops over the last k4b sources and adds their all-distinct kappa4 births. With our
`k4bparts=("sd",)` the whole "pd"/"p22" branch is skipped (`if "pd" in parts or "p22" in parts`) and so is
"s22", leaving only
    Y = A*B*d3 ;  BB = B*B ;  addd += 4*sdmult*(Y*BB).sum(1)
which is O(n^2) per source. So raising k4b from 1 to k costs k*O(n^2) -- about 0.03% of budget at k=5,
i.e. FREE, unlike every other lever in 2026-09-28_ceiling. If the gain survives on the base we can actually
ship, it is available NOW and does not wait for the cost breakthrough.

Base here = the k3n configuration (k3m + feedmult 0.8 + k22c 0.0), deadcut -2.5, nothing from the
cost-free stack. Self-check: k4b=1 must reproduce the k3n torch number.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-28_accuracy-free/k4b_ship.py
"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
SHIP = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
            marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
            k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-2.5, k31mult=0.65,
            feedmult=0.8, k22c=0.0)

gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
print("10 MLPs, SHIPPABLE base (k3n config). k4bparts=('sd',) -> every extra source is O(n^2).\n")
for k in (1, 2, 3, 5, 8, 15):
    cfg = dict(SHIP); cfg["k4b"] = k
    t0 = time.time()
    per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[str(k)] = {"raw": r, "x": base / r, "better": b, "per_mlp": per}
    print(f"  k4b = {k:<3d} raw {r:.4e}   x{base/r:.4f}   better {b}/10   [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "k4b_ship_n10.json").write_text(json.dumps(out, indent=1))
