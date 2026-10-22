"""If COST became free, how accurate can this chain get? (the ceiling of the merge->floor plan)

Why this is the decisive measurement right now. The cloud has shown the old tail's kappa3 tensor compresses
10x (640 -> 64 CP columns at 0.13% relative error, n=64), and Stage A showed the tail's error budget is
percent-level. If a fixed-size kappa3 state lands us at or below the 10% multiplier floor, then every FLOP
below that floor is FREE on the score -- and the right thing to do with it is buy accuracy back. So the
ceiling of the whole plan is  raw_best x 0.10, and this script measures raw_best.

Every arm here costs MORE budget than k3m; that is the point. Arms are things we currently refuse on cost:
the exact (2,1) correction instead of its rank-24 randomized split, the dead rows restored, path births at
all ages, and the knob winners from 2026-09-26_coherent-knobs.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-28_ceiling/ceiling.py [n_mlps]

Self-check: the base arm must print 2.4074e-08 on 1 MLP / 2.3135e-08 on 10.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
NM = int(sys.argv[1]) if len(sys.argv) > 1 else 3

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)
KNOBS = dict(feedmult=0.8, k22c=0.6)

ARMS = [
    ("base (k3l_t)",                    {}),
    ("knobs: feed 0.8 + k22c 0.6",      dict(KNOBS)),
    ("knobs, k22c=0.0",                 dict(feedmult=0.8, k22c=0.0)),
    ("knobs + sdmult 0.75",             dict(KNOBS, sdmult=0.75)),
    ("deadcut off",                     dict(deadcut=None)),
    ("thin rank 48",                    dict(corr="split48rr")),
    ("thin rank 96",                    dict(corr="split96rr")),
    ("thin rank 192",                   dict(corr="split192rr")),
    ("EXACT (2,1) correction",          dict(corr="full")),
    ("paths at all ages",               dict(e3age=None)),
    ("EXACT + deadcut off + knobs",     dict(corr="full", deadcut=None, **KNOBS)),
    ("EXACT + off + knobs + allages",   dict(corr="full", deadcut=None, e3age=None, **KNOBS)),
]

gt_all = np.load(BAKED / "gt.npy")
data = []
for m in range(NM):
    W = np.load(BAKED / f"weights_{m}.npy")
    data.append(([torch.tensor(w, dtype=torch.float64) for w in W], torch.tensor(gt_all[m][-1])))

out, base = {}, None
print(f"{NM} baked MLP(s).  raw = final-layer MSE; x = gain over base; 'floor' = raw x 0.10 "
      f"(the adjusted score if a fixed-size kappa3 state puts us at the multiplier floor)\n")
for tag, kw in ARMS:
    cfg = dict(K3L); cfg.update(kw)
    t0 = time.time()
    try:
        per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    except Exception as e:
        print(f"  {tag:34s} FAILED: {type(e).__name__}: {e}", flush=True)
        continue
    r = float(np.mean(per))
    if base is None:
        base = r
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": base / r, "per_mlp": per,
                "secs": round(time.time() - t0, 1)}
    print(f"  {tag:34s} raw {r:.4e}   x{base/r:.4f}   floor {r*0.10:.3e}   [{time.time()-t0:.0f}s]", flush=True)
    (OUT / f"ceiling_n{NM}.json").write_text(json.dumps(out, indent=1))
print(f"\n-> {OUT / f'ceiling_n{NM}.json'}")
