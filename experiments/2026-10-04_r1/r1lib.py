"""R1 shared harness (notes/PLAN-2026-10-04.md): random He-init ReLU MLPs at width n, depth L (the challenge's
convention, as experiments/2026-10-03_korder/korder.py), their Monte-Carlo ground truth (cached in korder/res/, same
files and seeds as korder.py), ARC's k_max=4 SIMPLE run stepped layer by layer (so its pre-activation statistics can be
read without keeping every layer's factored kappa4 in memory), and our float64 research chain (B9 + marg2).
Run under ARC's environment:
    ARC=$(cd ../../external/mlp_cumulant_propagation && pwd)
    env PYTHONPATH=$ARC/src nice -n 10 uv run --project $ARC python <script>.py ...
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP                                               # noqa: E402
from mlp_kprop.kprop_harmonic import (Kind, coerce_input, linear_kprop,     # noqa: E402
                                      nonlin_kprop)
from mlp_kprop.wick import WICK_COEF_D                                      # noqa: E402

HERE = Path(__file__).resolve().parent
KORDER = HERE.parent / "2026-10-03_korder"
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                                   # noqa: E402  (chain.py on the path)
from chain import relu_wick, run_chain                                      # noqa: E402

BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))


def weights(n, depth, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


def mc(Ws, seed, N, B=1 << 16):
    """Monte-Carlo post-ReLU means (all layers) and the last layer's per-neuron variance (korder.py's function and
    cache: korder/res/mc_n<n>_L<depth>_s<seed>_N<N>.npz)."""
    n, depth = Ws[0].shape[0], len(Ws)
    f = KORDER / "res" / f"mc_n{n}_L{depth}_s{seed}_N{N:.0e}.npz"
    if f.exists():
        d = np.load(f)
        return d["mu"], d["var"]
    g = torch.Generator().manual_seed(10_000 + seed)
    W32 = [W.to(torch.float32) for W in Ws]
    s1 = torch.zeros(depth, n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    done = 0
    while done < N:
        b = min(B, N - done)
        y = torch.randn(b, n, generator=g, dtype=torch.float32)
        for l in range(depth):
            y = torch.relu(y @ W32[l])
            s1[l] += y.sum(0, dtype=torch.float64)
        s2 += (y.double() ** 2).sum(0)
        done += b
    mu = s1 / N
    var = s2 / N - mu[-1] ** 2
    np.savez(f, mu=mu.numpy(), var=var.numpy())
    return mu.numpy(), var.numpy()


def k4_stats(Ws, s211_layers=(), keep=None):
    """ARC k_max=4 SIMPLE (factored kappa4), stepped as mlp_kprop does (average metric), reading at every layer l the
    PRE-activation statistics: C (full covariance), var, t = k3[iii], D = k3[iij], K4d = k4[iiii], K22 = k4[iijj],
    K31 = k4[iiij] (all float64 numpy), and the (2,1,1) slice S[a,b,c] = k4[aabc] (torch, n^3) at `s211_layers`.
    Returns (stats, mu) with mu the (depth, n) post-activation means (= mlp_kprop's k4S prediction).
    keep: optional callable (l, K_pre) -> None for further reads of the pre-activation tower."""
    n, depth = Ws[0].shape[0], len(Ws)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual",
              w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE)
    st = {k: {} for k in ("C", "var", "k3", "D", "k4", "K22", "K31", "S")}
    mus = []
    for l in range(depth):
        Wm = mlp.Ws[l]
        K = linear_kprop(K, Wm.weight, k_max=4, set_metric=mlp.init_scale[l], bias=Wm.bias)
        C = K[2].to_tensor()
        st["C"][l] = C.numpy().copy()
        st["var"][l] = torch.diagonal(C).numpy().copy()
        if 3 in K:
            st["k3"][l] = K[3].get_dslice((3,)).numpy().copy()
            st["D"][l] = K[3].get_dslice((2, 1)).numpy().copy()
        if 4 in K:
            st["k4"][l] = K[4].get_dslice((4,)).numpy().copy()
            st["K22"][l] = K[4].get_dslice((2, 2)).numpy().copy()
            st["K31"][l] = K[4].get_dslice((3, 1)).numpy().copy()
            if l in s211_layers:
                st["S"][l] = K[4].get_dslice((2, 1, 1)).clone()
        if keep is not None:
            keep(l, K)
        if l == depth - 1:   # the last mean in closed form (the step's (1,) slice: w01 + w31 t/6 + w41 K4d/24 --
            # identical to k=4's own, checked: V+t+K4 substituted at every layer reproduces k4S to the last digit);
            # skipping the last factored step halves the peak memory
            w, _, _ = relu_wick(K[1].to_tensor(), torch.tensor(st["var"][l]))
            mus.append(w[(0, 1)] + w[(3, 1)] * torch.tensor(st["k3"][l]) / 6 + w[(4, 1)] * torch.tensor(st["k4"][l]) / 24)
            break
        K = nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D[mlp.nonlin_names[l]], k_max=4, kind=Kind.SIMPLE,
                         use_pK=True, factor=True)
        mus.append(K[1].to_tensor().clone())
    return st, torch.stack(mus).numpy()


# the eight channels our chain reads at a layer, and how each is substituted
CHANNELS = ("V", "C0", "t", "D", "K4", "K22", "K31", "S")
_ORC = {"V": ("var", "var0_layers"), "C0": ("C", "C0_layers"), "t": ("k3", "k3_layers"), "D": ("D", "D_layers"),
        "K4": ("k4", "k4_layers"), "K22": ("K22", "K22_layers"), "K31": ("K31", "K31_layers")}


def chain_kwargs(subset, st, layers, base=None, feedS=None):
    """run_chain kwargs substituting the channels in `subset` at `layers` (a set of layer indices) from k4_stats.
    feedS: optional {l: tensor} overriding st["S"] for the S channel (e.g. a class-restricted slice)."""
    kw = dict(BASE if base is None else base)
    orc = {}
    for ch in subset:
        if ch == "S":
            S = feedS if feedS is not None else st["S"]
            kw["dense3"] = {"feedS": {l: S[l] for l in layers if l in S}, "mode": "replace"}
            continue
        key, lay = _ORC[ch]
        orc[key] = st[key]
        orc[lay] = {l for l in layers if l in st[key]}
    if orc:
        kw["oracle"] = orc
    return kw


def final_mse(Ws, kw, mu_mc):
    m_, _ = run_chain([W.clone() for W in Ws], **kw)
    return float(((np.asarray(m_[-1], dtype=np.float64) - mu_mc[-1]) ** 2).mean())


def name(subset):
    return "+".join(ch for ch in CHANNELS if ch in subset) or "base"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
