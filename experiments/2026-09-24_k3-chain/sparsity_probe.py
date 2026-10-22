"""How many rows/columns of the per-source legs are negligible because of the ReLU regime (MLP i, k3f config)?
dead: d1 = Phi(a) < eps (row of every leg ~0 after gating); inert hub: d2*sigma, alpha-coef tiny (|a| > a_max).
Prints per layer the dead fraction and the cost-weighted saving if dead rows are dropped from transports/pairings."""
import sys, math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
kw = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True)
mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record=True, **kw)
L = len(info); n = W.shape[1]
alive = []
print(" l | frac a<-3 | a<-2.5 | a<-2 | |a|>3 | |a|<2")
for l, r in enumerate(info):
    a = (r["mu_pre"] / r["var"].sqrt()).numpy()
    alive.append(a)
    print(f"{l:2d} | {np.mean(a < -3):.3f} | {np.mean(a < -2.5):.3f} | {np.mean(a < -2):.3f} | {np.mean(np.abs(a) > 3):.3f} | {np.mean(np.abs(a) < 2):.3f}")
for thr in (-3.0, -2.5):
    # transport at layer l of a leg gated at l-1: rows kept at output (layer l) x contraction rows kept (layer l-1)
    tot = keep = 0.0
    for l in range(1, L):
        S = l  # sources alive at layer l
        fo = np.mean(alive[l] >= thr); fi = np.mean(alive[l - 1] >= thr)
        tot += 4 * S; keep += 2 * S * fo * fi + 2 * S * fo * fo   # 2 transports (out x in), 2 pairings (rows x rows)
    print(f"threshold a<{thr}: cost-weighted kept fraction of per-source products {keep / tot:.3f}")
