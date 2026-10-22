"""Does the 'cost is free' accuracy stack multiply? 10 baked MLPs, base k3l_t = 2.3135e-08.

From ceiling_n3: thin rank 192 x1.0231, paths at all ages x1.0261, knobs (feedmult 0.8 + k22c 0.0) x1.0089,
deadcut off x1.0020. NOTE: the corr="full" arms in ceiling_n3 are NOT comparable -- chain.py takes a separate
branch there in which our path/age switches are inactive (identical numbers for two different configs and a
shorter runtime give it away), so the exact (2,1) correction is NOT measured; thin 192 is the honest proxy
for it (the return is saturating: 24 -> 48 -> 96 -> 192 gives 1.0135, 1.0183, 1.0231).

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-28_ceiling/combo.py
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
KN = dict(feedmult=0.8, k22c=0.0)

ARMS = [
    ("base",                           {}),
    ("knobs",                          dict(KN)),
    ("allages",                        dict(e3age=None)),
    ("thin192",                        dict(corr="split192rr")),
    ("knobs+allages",                  dict(KN, e3age=None)),
    ("knobs+allages+thin192",          dict(KN, e3age=None, corr="split192rr")),
    ("knobs+allages+thin192+cutoff",   dict(KN, e3age=None, corr="split192rr", deadcut=None)),
]

gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
print("10 MLPs. 'floor' = raw x 0.10 = the adjusted score at the multiplier floor.\n")
for tag, kw in ARMS:
    cfg = dict(K3L); cfg.update(kw)
    t0 = time.time()
    per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": base / r, "better": b, "per_mlp": per}
    print(f"  {tag:32s} raw {r:.4e}  x{base/r:.4f}  better {b}/10  floor {r*0.1:.3e}  [{time.time()-t0:.0f}s]",
          flush=True)
    (OUT / "combo_n10.json").write_text(json.dumps(out, indent=1))
