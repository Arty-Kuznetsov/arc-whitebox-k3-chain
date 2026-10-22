"""Break the mean-only ceiling: correct the COVARIANCE with the same cumulant sources.

The map (state_split3) says a mean-only method caps at 23.2x, but means + covariance + our readout
reaches 80x (k=3 quantities) and 175x (ground-truth means). So the covariance correction is the next
ingredient. It follows from the same rule already verified on the mean.

The cumulant correction rule is  Delta(cumulant) = sum_k (1/k!) E[d^k f] * kappa_k.
  1-point:  Delta mu_i = (1/6) E[d^3 f_i] kappa3[i,i,i]
            -- and E[d^3 ReLU] = -alpha*phi/sigma^2, which reproduces the Edgeworth term already
               verified against Monte Carlo. Same rule, so the 2-point case is:
  2-point:  Delta cov[i,j] = (1/2) d2_i d1_j D21[i,j] + (1/2) d1_i d2_j D21[j,i],  D21[i,j]=kappa3[i,i,j]
            (i=j gives Delta var_i = d1_i d2_i kappa3[i,i,i])

From the source decomposition kappa3_h = 3*Sym(sum_p d2_p e_p (x) u^p (x) u^p), transported by T:
  D21[i,j] = sum_p d2_p [ 2 A[i,p] B[i,p] B[j,p] + B[i,p]^2 A[j,p] ]
           = 2 * (M1 @ B^T) + (M2 @ A^T),   M1[i,p]=d2_p A[i,p]B[i,p],  M2[i,p]=d2_p B[i,p]^2
so it costs 2 extra matmuls per source per layer on top of the 2 that transport A and B.

Sweeps J with the covariance correction on and off. 'mean' arm reproduces multisource.py.
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
MATMUL = n * n * (2 * n - 1)
K2_FLOPS = 5.179e10
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)
CLIP = 1.0


def cost(J, cov_on):
    per_source = 4 if cov_on else 2
    mm = sum(per_source * min(l, J) for l in range(1, depth))
    mm -= sum(1 for l in range(1, depth) if J > 0)      # newest source: A starts as the identity
    return K2_FLOPS + mm * MATMUL


def chain(Wms, J, cov_on):
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
        sig = var.sqrt()
        al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        d1, d2 = Ph, ph / sig

        k3 = torch.zeros(n)
        D21 = torch.zeros(n, n) if (cov_on and src) else None
        for A, Bm, d2m in src:
            k3 = k3 + 3.0 * ((A * d2m) * Bm * Bm).sum(dim=1)
            if cov_on:
                M1 = (A * Bm) * d2m
                M2 = (Bm * Bm) * d2m
                D21 = D21 + 2.0 * (M1 @ Bm.T) + (M2 @ A.T)

        mu_g = mu_pre * Ph + sig * ph
        if src and J > 0:
            skew = (k3 / (var * sig)).clamp(-CLIP, CLIP)
            mu = mu_g - skew * sig * al * ph / 6.0
        else:
            mu = mu_g
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        var_post = (ez2 - mu_g * mu_g).clamp_min(0.0)
        C = torch.outer(Ph, Ph) * C_pre
        if D21 is not None:
            dc = 0.5 * (d2[:, None] * d1[None, :] * D21 + d1[:, None] * d2[None, :] * D21.T)
            C = C + dc
            var_post = (var_post + d1 * d2 * k3).clamp_min(1e-12)
        C.diagonal().copy_(var_post)

        for s in src:
            s[0] = s[0] * d1[:, None]
            s[1] = s[1] * d1[:, None]
        if J > 0 and l < depth - 1:
            src.append([None, C_pre * d1[:, None], d2.clone()])
            src[:] = src[-J:]
    return mu


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])
CFGS = [(0, False), (1, False), (3, False), (1, True), (2, True), (3, True), (4, True)]
res = {c: [] for c in CFGS}
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for c in CFGS:
        res[c].append(float(((chain(Wms, *c) - g) ** 2).mean()))
    print(f"  mlp {i}: " + " ".join(f"J{c[0]}{'C' if c[1] else 'M'}={res[c][-1]:.2e}" for c in CFGS),
          flush=True)
    del Wms, mlp

b = float(np.mean(res[(0, False)]))
print(f"\n=== Covariance correction from the same sources ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK J=0 = {b:.4e} (reference k=2 4.50e-06 on MLP 0)")
print("ceilings from the map: mean-only 23.2x ; means+cov+our readout 80x (k=3 quantities)\n")
print(f"{'J':>3} {'cov':>5} | {'final_layer_mse':>16} {'vs k=2':>8} | {'% of B':>7} {'mult':>6} {'adjusted':>10}")
for c in CFGS:
    v = float(np.mean(res[c])); f = cost(*c); pct = 100 * f / B_BUDGET
    mult = max(0.1, f / B_BUDGET)
    print(f"{c[0]:>3} {str(c[1]):>5} | {v:>16.4e} {b/v:>7.2f}x | {pct:>6.2f}% {mult:>6.3f} "
          f"{v*mult:>10.3e}{' OVER' if f > B_BUDGET else ''}")
