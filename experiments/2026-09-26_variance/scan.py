"""How sensitive is the chain to a UNIFORM scaling of the pre-activation variance, and is there a
beneficial direction? (2026-09-26_variance)

`decompose.py` showed that multiplicative factors measured on the base trajectory are stale after layer 0 --
the chain's own variance moves once you touch it -- so the only valid probes are corrections that are a
function of the CURRENT state. The simplest such family is one scalar:  var <- var (1 + c)  at every layer,
optionally ramped with depth. Both stages are probed:
  late  : only nonlin_slices sees it (the stage at which the exact-variance oracle arm is x1.54)
  early : the legs, D and K4d see it too (what a production correction would do if it changed var itself)

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-26_variance/scan.py [n_mlps] [stage]

Self-check: c = 0 must print MLP 0 = 2.4074e-08 (10 MLPs: 2.3135e-08).
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
NM = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STAGE = sys.argv[2] if len(sys.argv) > 2 else "late"

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)
CS = [-3e-3, -1e-3, -3e-4, -1e-4, -3e-5, 0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

gt_all = np.load(BAKED / "gt.npy")
res = {c: [] for c in CS}
for m in range(NM):
    W = np.load(BAKED / f"weights_{m}.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    g = torch.tensor(gt_all[m][-1])
    for c in CS:
        vf = None if c == 0.0 else (lambda l, v, mu, _c=c: torch.full_like(v, 1.0 + _c))
        kw = {("varfix" if STAGE == "early" else "varfix_late"): vf}
        pred, _ = run_chain(Ws, **kw, **K3L)
        res[c].append(float(((pred[-1] - g) ** 2).mean()))
    print(f"MLP {m}: " + "  ".join(f"{c:+.0e}={res[c][-1]:.4e}" for c in CS), flush=True)

print(f"\nuniform var <- var(1+c), stage={STAGE}, mean over {NM} MLP(s)")
b = float(np.mean(res[0.0]))
for c in CS:
    v = float(np.mean(res[c]))
    print(f"  c = {c:+.1e}   {v:.4e}   x{b / v:.3f} of base")
(OUT / f"scan_{STAGE}_n{NM}.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
