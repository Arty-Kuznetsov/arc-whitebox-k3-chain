"""Is the variance channel real, or is the exact covariance simply inconsistent with the rest?

oracle_k3l.py found: exact diagonal LATE (var_layers, seen only by nonlin_slices) = x1.43, exact
off-diagonal EARLY = x1.03, exact full C EARLY = x0.96 (worse than doing nothing). But "late" and
"early" are different intervention points, so those three do not decompose. This separates them
with var0_layers (exact diagonal, early — added to chain.py for this).

Sanity arm: var0 + C0 substitutes exactly what C does, so it must reproduce the C arm.
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
base = {k: mc[k] for k in ("k3", "k4", "var")}
base["C"] = [ma[f"C{l}"] for l in range(L)]; base["D"] = [ma[f"D{l}"] for l in range(L)]
REF = 2.4074e-08  # k3l, MLP 0

arms = {
    "var exact, LATE (readout only)":  {"var_layers": allL},
    "var exact, EARLY (whole step)":   {"var0_layers": allL},
    "C0 exact, EARLY":                 {"C0_layers": allL},
    "var0 + C0 EARLY (= C; sanity)":   {"var0_layers": allL, "C0_layers": allL},
    "C exact, EARLY":                  {"C_layers": allL},
}
print(f"{'arm':34s} {'final MSE':>12s}  {'x vs k3l':>9s}")
for name, o in arms.items():
    mu, _ = run_chain(Ws, oracle=dict(o, **base), **K3L)
    mse = float(((mu[-1].numpy() - gt[-1]) ** 2).mean())
    print(f"{name:34s} {mse:12.4e}  {REF / mse:8.2f}x", flush=True)
