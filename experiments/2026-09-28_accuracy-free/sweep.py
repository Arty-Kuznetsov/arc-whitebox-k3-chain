"""Cost-unconstrained accuracy: what ELSE is there, on top of the measured x1.11 stack?

Rationale. At the 10% multiplier floor, FLOPs are worth nothing and raw is worth 1:1. Our measured ceiling
(2026-09-28_ceiling) is raw 1.92e-8 -> 1.92e-9 at the floor = rank 2 today. **#1 (J2W 1.70e-9, raw 1.62e-8 at
util 10.5%) needs another x1.13 of accuracy.** Many terms in this chain were rejected with the words "does
not pay" -- which always meant a cost/benefit at 25% of budget, never an accuracy verdict. At the floor that
judgement is void, so every one of them is re-opened here, measured ON TOP of the best known configuration.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-28_accuracy-free/sweep.py

Self-check: the `stack` arm must reproduce 2.0844e-08 (combo_n10.json, 10 MLPs).
10 MLPs throughout -- 3 MLPs has mis-ranked a 5-10% effect four times in this project (NEGATIVE 51).
"""
import json, time, sys
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

ARMS = [
    ("stack (self-check 2.0844e-08)", {}),
    # --- the AUGMENT term set: the single biggest untried block
    ("+ aug (AUGMENT dense diagrams)", dict(aug=True)),
    ("+ edge2 (2nd-order Edgeworth)",  dict(edge2=True)),
    ("+ aug + edge2",                  dict(aug=True, edge2=True)),
    # --- rejected ON COST in NEGATIVE-RESULTS, re-opened
    ("+ e3 births (NEG 45)",           dict(e3=True)),
    ("+ full k4 births (NEG 47)",      dict(k4b=1, k4bparts=("pd", "p22", "sd", "s22"))),
    ("+ k4b=2",                        dict(k4b=2)),
    ("+ q22x=1",                       dict(q22x=1)),
    ("+ q22x=2",                       dict(q22x=2)),
    ("+ k4k3=1.0",                     dict(k4k3=1.0)),
    ("+ pathD=full",                   dict(pathD="full")),
    ("+ tet=4",                        dict(tet=4)),
    ("+ tet=2 (control)",              dict(tet=2)),
    ("+ thin 384",                     dict(corr="split384rr")),
    ("+ exact corr (control)",         dict(corr="split1024rr")),
]

gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
print("10 MLPs, cost unconstrained. 'floor' = raw x 0.10. Board: #1 1.70e-9, #2 1.90e-9, #5 2.20e-9.\n")
for tag, kw in ARMS:
    cfg = dict(STACK); cfg.update(kw)
    t0 = time.time()
    try:
        per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    except Exception as e:
        print(f"  {tag:34s} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
        continue
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": base / r, "better": b, "per_mlp": per}
    print(f"  {tag:34s} raw {r:.4e}  x{base/r:.4f}  better {b}/10  floor {r*0.1:.3e}  [{time.time()-t0:.0f}s]",
          flush=True)
    (OUT / "sweep_n10.json").write_text(json.dumps(out, indent=1))
