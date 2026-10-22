"""Do the three new cost-free-regime levers compose, and where are their optima? 10 MLPs.

From sweep_n10 (on top of the x1.11 stack, base 2.0844e-08):
  pathD="full" x1.0252 (7/10), k4k3=1.0 x1.0154 (9/10), k4b=2 x1.0136 (7/10), thin384 x1.0079 (9/10).
Dead on accuracy, not on cost (this updates NEGATIVE 45 and the AUGMENT notes):
  aug x0.463, e3 x0.944, edge2 x0.981, tet=4 x0.999, q22x ~1.000.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-28_accuracy-free/combine.py
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
PF = dict(pathD="full")

ARMS = [
    ("stack (2.0844e-08)",              {}),
    ("pathD=full",                      dict(PF)),
    ("pathD + k4k3 0.5",                dict(PF, k4k3=0.5)),
    ("pathD + k4k3 1.0",                dict(PF, k4k3=1.0)),
    ("pathD + k4k3 1.5",                dict(PF, k4k3=1.5)),
    ("pathD + k4k3 2.0",                dict(PF, k4k3=2.0)),
    ("pathD + k4b 2",                   dict(PF, k4b=2)),
    ("pathD + k4b 3",                   dict(PF, k4b=3)),
    ("pathD + k4k3 1.0 + k4b 2",        dict(PF, k4k3=1.0, k4b=2)),
    ("+ thin384",                       dict(PF, k4k3=1.0, k4b=2, corr="split384rr")),
    ("+ thin384, k31mult 0.55",         dict(PF, k4k3=1.0, k4b=2, corr="split384rr", k31mult=0.55)),
    ("+ thin384, k31mult 0.75",         dict(PF, k4k3=1.0, k4b=2, corr="split384rr", k31mult=0.75)),
]

gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
print("10 MLPs. grader raw = torch x 0.920; at the 10% floor the score is that x 0.10.")
print("Board: #1 1.70e-9 (raw 1.62e-8), #2 1.90e-9, #5 2.20e-9.\n")
for tag, kw in ARMS:
    cfg = dict(STACK); cfg.update(kw)
    t0 = time.time()
    try:
        per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    except Exception as e:
        print(f"  {tag:30s} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": base / r, "better": b, "per_mlp": per}
    print(f"  {tag:30s} raw {r:.4e}  x{base/r:.4f}  better {b}/10  "
          f"floor {r*0.920*0.1:.3e}  [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "combine_n10.json").write_text(json.dumps(out, indent=1))
