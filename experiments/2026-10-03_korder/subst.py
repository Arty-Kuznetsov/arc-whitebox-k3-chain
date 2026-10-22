"""Which channels carry the next order's gain? At small width, take ARC's k_max=4 SIMPLE run (full kappa4, x6-7 better
than our chain at n=256, L=8) as a near-exact source of the pre-activation statistics, and substitute them into our
float64 chain (research base B9 + marg2) at EVERY layer, one channel at a time and jointly:
  var (diag C), C (full pre covariance), t (kappa3 diagonal), D (kappa3 (2,1) slice), K4d (kappa4 diagonal),
  K22 (kappa4 (2,2) slice). Final-layer MSE against the cached Monte-Carlo means (korder.py, 1e7 samples).
    env PYTHONPATH=$ARC/src nice -n 10 uv run --project $ARC python subst.py <n> <depth> <n_nets> <N_mc>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(3)
torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP                                   # noqa: E402
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind            # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402
n, depth, nets, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
SKIP = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # substitute at layers 1 .. depth-1-SKIP (SKIP=1: the chain does the
                                                       # last step itself, so the arm measures UPSTREAM channels only)
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))


def weights(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


def k4_stats(Ws):
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                    up_to_layer=f"act{depth - 1}")
    st = {k: {} for k in ("C", "var", "k3", "D", "k4", "K22")}
    for l in range(depth):
        T = out[f"pre{l}"]
        C = T[2].to_tensor()
        st["C"][l] = C.numpy()
        st["var"][l] = torch.diagonal(C).numpy()
        if 3 in T:
            st["k3"][l] = T[3].get_dslice((3,)).numpy()
            st["D"][l] = T[3].get_dslice((2, 1)).numpy()
        if 4 in T:
            st["k4"][l] = T[4].get_dslice((4,)).numpy()
            st["K22"][l] = T[4].get_dslice((2, 2)).numpy()
    mu = torch.stack([out[f"act{l}"][1].to_tensor() for l in range(depth)]).numpy()
    return st, mu


ARMS = {"base": (), "var": ("var",), "C": ("C",), "t": ("k3",), "D": ("D",), "t+D": ("k3", "D"),
        "K4d": ("k4",), "K22": ("K22",), "K4d+K22": ("k4", "K22"), "t+D+K4d+K22": ("k3", "D", "k4", "K22"),
        "all": ("C", "k3", "D", "k4", "K22")}
res = {"n": n, "depth": depth, "arms": {a: [] for a in ARMS}, "k4S": []}
for s in range(nets):
    Ws = weights(s)
    d = np.load(HERE / "res" / f"mc_n{n}_L{depth}_s{s}_N{N:.0e}.npz")
    mu_mc, var_mc = d["mu"], d["var"]
    st, mu4 = k4_stats(Ws)
    e4 = float(((mu4[-1] - mu_mc[-1]) ** 2).mean())
    res["k4S"].append(e4)
    line = f"net {s}: k4S {e4:.3e} |"
    for arm, keys in ARMS.items():
        orc = {}
        for k in keys:
            orc[k] = st[k]
            orc[f"{k}_layers"] = {l for l in st[k] if 1 <= l <= depth - 1 - SKIP}   # layer 0 is exactly Gaussian
        m_, _ = run_chain([W.clone() for W in Ws], oracle=(orc or None), **BASE)
        pred = np.stack([np.asarray(x, dtype=np.float64) for x in m_])
        e = float(((pred[-1] - mu_mc[-1]) ** 2).mean())
        res["arms"][arm].append(e)
        line += f" {arm} {e:.3e}"
    print(line, flush=True)
    json.dump(res, open(HERE / "res" / f"subst_n{n}_L{depth}_skip{SKIP}.json", "w"), indent=1)
print("\nmean over nets (final-layer MSE vs MC; MC noise ~var/N):")
print(f"  k4S (ARC k_max=4) {np.mean(res['k4S']):.3e}")
b = np.mean(res["arms"]["base"])
for arm in ARMS:
    v = np.mean(res["arms"][arm])
    print(f"  {arm:14s} {v:.3e}   x{v / b:.3f} of our chain")
