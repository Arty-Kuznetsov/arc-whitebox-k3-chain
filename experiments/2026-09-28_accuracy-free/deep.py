"""Push the two levers that were still improving: k4b (kappa4 birth depth) and k31mult. 10 MLPs.

combine_n10 (on the x1.11 stack, base 2.0844e-08): pathD=full x1.0252, +k4b 2 x1.0465 (9/10),
+k4b 3 x1.0637 (10/10), +k4k3 1.0+k4b 2+thin384 x1.0655 (10/10), +k31mult 0.55 x1.0812 (10/10).
k4k3 has an interior optimum near 0.5; k4b and k31mult had not turned over yet.

STRUCTURAL vs FITTED. pathD, k4b and thin384 are structural (terms added / rank raised) and should transfer
to the 100-MLP flopscope set. k4k3, k31mult, feedmult, k22c are SCALARS fitted on these same 10 MLPs, and
k3n measured the shrinkage of exactly that kind of gain: torch x1.0089 on 10 -> flopscope x1.0030 on 100,
a factor of 3. So this run reports the structural-only combination separately, and that is the number to
trust.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-28_accuracy-free/deep.py
"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
STACK = dict(k4="memo22", corr="split192rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
             marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=None, pathD="merge", k4b=1,
             k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=None, k31mult=0.65,
             feedmult=0.8, k22c=0.0)
S = dict(pathD="full", corr="split384rr")          # structural base

ARMS = [
    ("stack (2.0844e-08)",                 {}),
    ("STRUCTURAL: pathD+thin384+k4b3",     dict(S, k4b=3)),
    ("STRUCTURAL: ... k4b4",               dict(S, k4b=4)),
    ("STRUCTURAL: ... k4b5",               dict(S, k4b=5)),
    ("struct k4b3 + k4k3 0.5",             dict(S, k4b=3, k4k3=0.5)),
    ("struct k4b3 + k31mult 0.55",         dict(S, k4b=3, k31mult=0.55)),
    ("struct k4b3 + k31mult 0.45",         dict(S, k4b=3, k31mult=0.45)),
    ("struct k4b3 + both",                 dict(S, k4b=3, k4k3=0.5, k31mult=0.55)),
    ("struct k4b4 + both",                 dict(S, k4b=4, k4k3=0.5, k31mult=0.55)),
]

gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
print("10 MLPs. floor = torch raw x 0.920 x 0.10.  Board: #1 1.70e-9, #2 1.90e-9, #5 2.20e-9.\n")
for tag, kw in ARMS:
    cfg = dict(STACK); cfg.update(kw)
    t0 = time.time()
    try:
        per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    except Exception as e:
        print(f"  {tag:34s} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": base / r, "better": b, "per_mlp": per}
    print(f"  {tag:34s} raw {r:.4e}  x{base/r:.4f}  better {b}/10  floor {r*0.0920:.3e}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    (OUT / "deep_n10.json").write_text(json.dumps(out, indent=1))
