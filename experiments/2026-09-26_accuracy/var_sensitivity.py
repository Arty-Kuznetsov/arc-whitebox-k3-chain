"""How steep is the chain's response to the per-neuron variance? (MLP 0, k3l base)

Why this exists: oracle_stage.py's sanity arm failed. Substituting the exact variance from
mc_oracle_0.npz (N=1e7) gives x1.43, but substituting diag(C) from mc_oracle_all_0.npz (N=4e6)
gives x0.96. The two differ by 7e-4 relative, which is exactly the quadrature sum of their MC
noises (4.5e-4 and 7.1e-4) — so they differ by NOISE ONLY, with no systematic part, and the final
MSE moves 48% between them.

This walks the straight line between the two estimates, d(t) = var + t*(diag(C) - var), and past
both ends. A smooth shallow curve would mean the two arms happen to straddle a minimum; a steep or
ragged one means single-channel oracle attribution is not resolvable at this oracle accuracy, and
every "x1.43 / x0.96 / x0.85" in oracle_k3l.py carries an uncertainty as large as the effect.
"""
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

CHAIN = Path(__file__).resolve().parent.parent / "2026-09-24_k3-chain"
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
mc = dict(np.load(CHAIN / "runs" / "mc_oracle_0.npz"))
ma = np.load(CHAIN / "runs" / "mc_oracle_all_0.npz")
W = np.load(BAKED / "weights_0.npy"); gt = np.load(BAKED / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
L = len(Ws); allL = tuple(range(L))
K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True,
           k4feed="P2", marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1,
           pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0,
           k31mult=0.65)
REF = 2.4074e-08

v_hi = [np.asarray(mc["var"][l], dtype=np.float64) for l in range(L)]            # N=1e7
v_lo = [np.diagonal(np.asarray(ma[f"C{l}"], dtype=np.float64)).copy() for l in range(L)]  # N=4e6
step = [lo - hi for hi, lo in zip(v_hi, v_lo)]
rel = np.mean([np.abs(s / h).mean() for s, h in zip(step, v_hi)])
print(f"the two oracle variances differ by {rel:.2e} relative (mean over layers)\n")
print(f"{'t':>6s}  {'variance used':28s} {'final MSE':>12s}  {'x vs k3l':>9s}")
for t in (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    orc = {"var": [h + t * s for h, s in zip(v_hi, step)], "var0_layers": allL}
    mu, _ = run_chain(Ws, oracle=orc, **K3L)
    mse = float(((mu[-1].numpy() - gt[-1]) ** 2).mean())
    tag = {0.0: "= N=1e7 oracle", 1.0: "= N=4e6 oracle"}.get(t, "")
    print(f"{t:6.2f}  {tag:28s} {mse:12.4e}  {REF / mse:8.2f}x", flush=True)
