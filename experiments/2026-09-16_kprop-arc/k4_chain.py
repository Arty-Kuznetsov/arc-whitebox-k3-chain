"""Does a 4th-cumulant term help the chain? (memoryless path diagram, ~1 matmul/source/layer)

Same cumulant rule, next order:
    Delta mu += (1/24) * E[d^4 ReLU] * kappa4[i,i,i,i],   E[d^4 ReLU] = (alpha^2 - 1) phi / sigma^3
The leading kappa4 diagrams are trees. At a layer where alpha = 0 the STAR term vanishes identically
(its hub carries d3 = -alpha*phi/sigma^2), so the PATH diagram carries it; validated at width 64
against 2e7 Monte-Carlo samples with correlation 0.985 (k4_diagrams.py).

Crucially the path template reuses matrices the kappa3 machinery already carries. With
A[i,a], B[i,a] the transported source legs and d2 the birth coefficients,
    P = (A * d2 * B)^T    (already formed elementwise for kappa3)
    path_i = sum_b P[b,i] * (C P)[b,i]
so it costs exactly ONE extra matmul per source per layer (C @ P).

Sweeps the prefactor to see sign, magnitude and sensitivity before any derivation of the exact
combinatorial constant. c4 = 0 reproduces the shipped multi-source chain (self-check).
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
K2 = 5.179e10
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)
CLIP = 1.0


def cost(J, k4_on):
    per = (3 if k4_on else 2)
    mm = sum(per * min(l, J) for l in range(1, depth)) - (depth - 1 if J else 0)
    return K2 + mm * MATMUL


def chain(Wms, J, c4):
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

        k3 = torch.zeros(n)
        k4 = torch.zeros(n)
        for A, Bm, d2m in src:
            AD = A * d2m
            k3 = k3 + 3.0 * (AD * Bm * Bm).sum(dim=1)
            if c4 != 0.0:
                P = (AD * Bm).T                       # P[b, i]; reuses the kappa3 product
                k4 = k4 + (P * (C_pre @ P)).sum(dim=0)
        mu_g = mu_pre * Ph + sig * ph
        mu = mu_g
        if src:
            skew = (k3 / (var * sig)).clamp(-CLIP, CLIP)
            mu = mu - skew * sig * al * ph / 6.0
            if c4 != 0.0:
                kurt = (c4 * k4 / (var * var)).clamp(-CLIP, CLIP)
                mu = mu + kurt * (al * al - 1.0) * ph * sig / 24.0
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        C = torch.outer(Ph, Ph) * C_pre
        C.diagonal().copy_((ez2 - mu_g * mu_g).clamp_min(0.0))
        for s in src:
            s[0] = s[0] * Ph[:, None]
            s[1] = s[1] * Ph[:, None]
        if J > 0 and l < depth - 1:
            src.append([None, C_pre * Ph[:, None], ph / sig])
            src[:] = src[-J:]
    return mu


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])
CFGS = [(3, 0.0), (3, 2.0), (3, 4.0), (3, 7.45), (3, 12.0), (2, 7.45), (2, 0.0)]
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
    print(f"  mlp {i}: " + " ".join(f"J{c[0]}c{c[1]:g}={res[c][-1]:.3e}" for c in CFGS), flush=True)
    del Wms, mlp

base = float(np.mean(res[(3, 0.0)]))
print(f"\n=== 4th-cumulant path term on top of the 3-source chain ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK J=3, c4=0 = {base:.4e} (the shipped chain)\n")
print(f"{'J':>3} {'c4':>6} | {'final_layer_mse':>16} {'vs c4=0':>9} | {'% of B':>7} {'mult':>6} {'adjusted':>10}")
for c in CFGS:
    v = float(np.mean(res[c])); f = cost(c[0], c[1] != 0.0)
    mult = max(0.1, f / B_BUDGET)
    print(f"{c[0]:>3} {c[1]:>6g} | {v:>16.4e} {base/v:>8.2f}x | {100*f/B_BUDGET:>6.2f}% {mult:>6.3f} "
          f"{v*mult:>10.3e}")
print("\nA prefactor sweep is a diagnostic, not a shipped constant: if the term helps, the exact")
print("combinatorial coefficient gets derived and verified before it goes into the estimator.")
