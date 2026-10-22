"""Task 2 side check: is a tail-safe READOUT worth an arm? One step, per regime, no chain: with the MC oracle's own
pre-activation (mu, var, kappa3, kappa4) of every neuron (oracles/mc_M_sS.npz), the post mean from
    gauss   the Gaussian value w(0,1)
    edge3   + w(3,1) k3/6                     (first-order Edgeworth, kappa3 only)
    edge    + w(3,1) k3/6 + w(4,1) k4/24      (the chain's readout form, P1 of nonlin_slices)
    edge2   edge + w(6,1) k3^2/72             (second order, the edge2 switch)
    sln     shifted lognormal matched to (mu, var, kappa3): exact E[relu] of that law (tail-safe: a proper density)
against (i) the MC post mean of the SAME samples (the sampling noise of the moments largely cancels) and (ii) the baked
GT (1e9 samples, independent). MSE per regime (alpha = mu/sigma: dead < -2.5 <= kink <= 2.5 < sat), layers 1..15."""
import json
import sys

import numpy as np
import torch
ndtr = lambda x: torch.special.ndtr(torch.as_tensor(x)).numpy()  # noqa: E731

import ta
from chain import relu_wick

M, S = int(sys.argv[1]), sys.argv[2]
z = np.load(ta.HERE / "oracles" / f"mc_{M}_s{S}.npz")
gt = ta._GT.get(ta.pb.BAKED)
gt = np.load(ta.pb.BAKED / "gt.npy")[M] if gt is None else gt[M]


def sln(mu, var, k3):
    s_ = np.sqrt(var)
    gam = np.abs(k3) / s_ ** 3
    out = np.maximum(mu, 0.0) + 0.0
    ok = gam > 1e-6
    # w = exp(s^2) solves (w + 2) sqrt(w - 1) = gamma (monotone): bisection on w - 1 in (0, big)
    lo, hi = np.zeros_like(gam), np.full_like(gam, 50.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = (mid + 3.0) * np.sqrt(mid) - gam
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    w = 1.0 + 0.5 * (lo + hi)
    sig2 = np.log(w)
    em = np.sqrt(var / ((w - 1.0) * w))      # exp(m)
    mL = em * np.sqrt(w)                     # E[L] = exp(m + s^2/2)
    sg = np.sqrt(sig2)
    lm = np.log(em)
    sign = np.sign(k3)
    c = mu - sign * mL
    res = np.empty_like(mu)
    pos = sign > 0  # X = c + L
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = (np.log(np.where(c < 0, -c, 1.0)) - lm) / sg
        rp = np.where(c >= 0, mu, c * ndtr(-z0) + mL * ndtr(sg - z0))
        z1 = (np.log(np.where(c > 0, c, 1.0)) - lm) / sg
        rn = np.where(c <= 0, 0.0, c * ndtr(z1) - mL * ndtr(z1 - sg))
    res = np.where(pos, rp, rn)
    return np.where(ok, res, out * 0 + np.nan)


out = {}
for l in range(1, z["var"].shape[0]):
    mu, var, k3, k4 = (torch.tensor(z[k][l]) for k in ("mean", "var", "k3", "k4"))
    w, _, _ = relu_wick(mu, var)
    g0 = w[(0, 1)]
    forms = {"gauss": g0, "edge3": g0 + w[(3, 1)] * k3 / 6, "edge": g0 + w[(3, 1)] * k3 / 6 + w[(4, 1)] * k4 / 24}
    forms["edge2"] = forms["edge"] + w[(6, 1)] * k3 * k3 / 72
    fs = {k: v.numpy() for k, v in forms.items()}
    s_ = sln(mu.numpy(), var.numpy(), k3.numpy())
    fs["sln"] = np.where(np.isnan(s_), fs["gauss"], s_)
    a = (mu / var.sqrt()).numpy()
    regs = {"dead": a < -2.5, "kink": np.abs(a) <= 2.5, "sat": a > 2.5, "all": np.ones_like(a, bool)}
    mcp, g = z["post_mean"][l], gt[l]
    out[l] = {R: {"n": int(k.sum()), **{f: [float(((v - mcp)[k] ** 2).sum() / a.size), float(((v - g)[k] ** 2).sum() / a.size)]
                                         for f, v in fs.items()}, "mc_vs_gt": float(((mcp - g)[k] ** 2).sum() / a.size)}
              for R, k in regs.items()}
(ta.HERE / "res1" / f"readout_{M}_s{S}.json").write_text(json.dumps(out, indent=1))
for l in (5, 10, 15):
    for R in ("dead", "kink", "sat"):
        o = out[l][R]
        print(f"L{l:2d} {R:4s} n={o['n']:4d}  vsMC: " + " ".join(f"{f} {o[f][0]:.1e}" for f in ("gauss", "edge3", "edge", "edge2", "sln"))
              + f" | vsGT: " + " ".join(f"{f} {o[f][1]:.1e}" for f in ("gauss", "edge", "sln")) + f" | MC-GT {o['mc_vs_gt']:.1e}")
