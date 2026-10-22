"""R2 (notes/PLAN-2026-10-04.md): one joint carrier -- the YOUNG kappa4 trees (path, star, 3+2) born at every ReLU
from the chain's OWN state, carried exactly for `ages` layers and read into K4d and K31 on top of the closure
(chain.py k4young; ARC's factored kappa4 machinery, small widths). R1 (experiments/2026-10-04_r1) found that k=4's own
young trees, handed to the chain this way (r1res.py "add:trees", K4+K31 only), give x0.55 / 0.77 / 0.87 at n=128.
Kill rule R2: x0.85 or better on BOTH nets at n=256.
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r2young.py <n> <depth> <seeds> [arms]
"""
import json
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("R2_THREADS", "3")))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2026-10-04_r1"))
from r1lib import BASE, Timer, final_mse, mc, weights                     # noqa: E402

HERE = Path(__file__).resolve().parent
n, depth = int(sys.argv[1]), int(sys.argv[2])
seeds = [int(x) for x in sys.argv[3].split(",")]
Y = dict(ages=2, types=("path", "star", "32"), resid=True, read=("K4d", "K31"), conv="arc")
ARMS = {
    "base": {},
    "y2": dict(k4young=Y),
    "y2pure": dict(k4young=dict(Y, conv="pure")),
    "y1": dict(k4young=dict(Y, ages=1)),
    "y3": dict(k4young=dict(Y, ages=3)),
    "y2-32": dict(k4young=dict(Y, types=("path", "star"))),
    "y2noresid": dict(k4young=dict(Y, resid=False)),
    "y2K4only": dict(k4young=dict(Y, read=("K4d",))),
    "y2K31only": dict(k4young=dict(Y, read=("K31",))),
    "y2c1": dict(k4young=dict(Y, couple=1)),
    "y2c2": dict(k4young=dict(Y, couple=2)),
    # piece study on legs (young4.py): which parts of the trees carry the gain
    "L:y2": dict(k4young=dict(Y, impl="legs")),
    "L:pathdiag+star+32": dict(k4young=dict(Y, impl="legs", types=("pathdiag", "star", "32"))),
    "L:pathoff+star+32": dict(k4young=dict(Y, impl="legs", types=("pathoff", "star", "32"))),
    "L:path+star": dict(k4young=dict(Y, impl="legs", types=("path", "star"))),
    "L:32": dict(k4young=dict(Y, impl="legs", types=("32",))),
    "L:path+32": dict(k4young=dict(Y, impl="legs", types=("path", "32"))),
    "L:star+32": dict(k4young=dict(Y, impl="legs", types=("star", "32"))),
    "L:pathdiag+32": dict(k4young=dict(Y, impl="legs", types=("pathdiag", "32"))),
    "L:y2c31": dict(k4young=dict(Y, impl="legs", k31_32="cheap")),
    # the production form (young4.Young4Chain): K4d only, from the chain's own legs and its own D pairing terms
    "L:y2K4": dict(k4young=dict(Y, impl="legs", read=("K4d",))),
    "L:y2K4pd": dict(k4young=dict(Y, impl="legs", read=("K4d",), types=("pathdiag", "star", "32"))),
    "C:y2K4": dict(k4young=dict(Y, impl="chain", read=("K4d",), types=("path", "star", "32"))),
    "C:y2K4pd": dict(k4young=dict(Y, impl="chain", read=("K4d",), types=("pathdiag", "star", "32"))),
    "C:y1K4pd": dict(k4young=dict(Y, impl="chain", ages=1, read=("K4d",), types=("pathdiag", "star", "32"))),
}
sel = sys.argv[4].split(",") if len(sys.argv) > 4 else list(ARMS)
for seed in seeds:
    f = HERE / "res" / f"young_n{n}_L{depth}_s{seed}.json"
    res = json.loads(f.read_text()) if f.exists() else {"n": n, "depth": depth, "seed": seed, "arms": {}}
    Ws = weights(n, depth, seed)
    mu_mc, var_mc = mc(Ws, seed, int(1e7))
    noise = float(var_mc.mean() / 1e7)
    res["mc_noise"] = noise
    for a in sel:
        if a in res["arms"]:
            continue
        T = Timer()
        e = final_mse(Ws, dict(BASE, **ARMS[a]), mu_mc)
        res["arms"][a] = e
        b = res["arms"].get("base", e)
        print(f"seed {seed}  {a:12s} {e:.4e}  x{(e - noise) / (b - noise):.3f}  [{T():.0f} s]", flush=True)
        f.write_text(json.dumps(res, indent=1))
