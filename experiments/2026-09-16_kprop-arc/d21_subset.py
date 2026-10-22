"""Cheapest route under the 10% floor: apply the D21 covariance term to only the newest sources.

At J=4 the estimator runs ~11 matmuls per layer: 2 for the newest source (its propagator is free),
3 for each older one (transport A, transport B, one D21 product). The D21 products are 4 of the 11.
If the covariance correction is dominated by the NEWEST sources, restricting it to the newest k saves
4-k matmuls per layer at little accuracy cost -- far simpler than factoring the propagators.

Sweeps k = number of newest sources that get the D21 term, with J = 4 transported throughout.
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
B_BUDGET = 2 ** 41
MM = n * n * (2 * n - 1) * 0.88
K2 = 5.179e10
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)
CLIP = 1.0
J = 4


def cost(k):
    mm = 0
    for l in range(1, depth):
        live = min(l, J)
        mm += 2 * live - 1              # transport: A free for the newest
        mm += min(live, k)              # one D21 product per corrected source
    return K2 + mm * MM


def chain(Wms, k):
    mu, C = torch.zeros(n), torch.eye(n)
    src = []
    for l in range(depth):
        Wm = Wms[l]
        for s in src:
            s[0] = Wm if s[0] is None else Wm @ s[0]
            s[1] = Wm @ s[1]
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt(); al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        d1, d2 = Ph, ph / sig
        k3 = torch.zeros(n)
        D21 = torch.zeros(n, n) if (k and src) else None
        for idx, (A, Bm, d2m) in enumerate(src):
            AD = A * d2m
            k3 = k3 + 3.0 * (AD * Bm * Bm).sum(dim=1)
            if k and idx >= len(src) - k:            # newest k sources only
                D21 = D21 + 2.0 * ((AD * Bm) @ Bm.T)
        mu_g = mu_pre * Ph + sig * ph
        mu = mu_g
        if src:
            mu = mu - (k3 / (var * sig)).clamp(-CLIP, CLIP) * sig * al * ph / 6.0
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        var_post = (ez2 - mu_g * mu_g).clamp_min(0.0)
        C = torch.outer(Ph, Ph) * C_pre
        if D21 is not None:
            C = C + 0.5 * (d2[:, None] * d1[None, :] * D21 + d1[:, None] * d2[None, :] * D21.T)
        C.diagonal().copy_(var_post)
        for s in src:
            s[0] = s[0] * d1[:, None]
            s[1] = s[1] * d1[:, None]
        if l < depth - 1:
            src.append([None, C_pre * d1[:, None], d2.clone()])
            src[:] = src[-J:]
    return mu


gt = np.load(BAKED / "gt.npy")
N = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
KS = [0, 1, 2, 3, 4]
res = {k: [] for k in KS}
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for k in KS:
        res[k].append(float(((chain(Wms, k) - g) ** 2).mean()))
    print(f"  mlp {i} done", flush=True)
    del Wms, mlp

print(f"\n=== D21 applied to the newest k of J=4 sources ({N} MLPs) ===\n")
print(f"{'k':>3} | {'final_layer_mse':>16} | {'% of B':>7} {'mult':>6} {'adjusted':>10}")
best = None
for k in KS:
    v = float(np.mean(res[k])); f = cost(k)
    mult = max(0.1, f / B_BUDGET); adj = v * mult
    print(f"{k:>3} | {v:>16.4e} | {100*f/B_BUDGET:>6.2f}% {mult:>6.3f} {adj:>10.3e}")
    if best is None or adj < best[1]:
        best = (k, adj)
print(f"\nbest: k={best[0]} at {best[1]:.4e}   (shipped k=4 scores 7.1794e-08 on the full split)")
