"""Diagonal-only variance correction: free, and separated from the off-diagonal that hurt.

covcorr.py applied the kappa3 correction to the WHOLE covariance and it hurt at every J. But it
applied two different things at once. The same cumulant rule gives, for i = j,
    Delta var_i = d1_i * d2_i * kappa3[i,i,i]
which needs only the kappa3 diagonal we already compute -- ZERO extra matmuls -- and touches exactly
the quantity the mean readout consumes (sigma_i = sqrt(C_ii)). The off-diagonal part needs D21 and
costs 2 matmuls per source per layer.

Arms at J=4: none / diagonal only / off-diagonal only / both.
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
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)
CLIP = 1.0


def chain(Wms, J, diag_on, offdiag_on):
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
        D21 = torch.zeros(n, n) if (offdiag_on and src) else None
        for A, Bm, d2m in src:
            AD = A * d2m
            k3 = k3 + 3.0 * (AD * Bm * Bm).sum(dim=1)
            if offdiag_on:
                D21 = D21 + 2.0 * ((AD * Bm) @ Bm.T) + ((Bm * Bm) * d2m) @ A.T

        mu_g = mu_pre * Ph + sig * ph
        mu = mu_g
        if src:
            skew = (k3 / (var * sig)).clamp(-CLIP, CLIP)
            mu = mu - skew * sig * al * ph / 6.0
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        var_post = (ez2 - mu_g * mu_g).clamp_min(0.0)
        C = torch.outer(Ph, Ph) * C_pre
        if D21 is not None:
            C = C + 0.5 * (d2[:, None] * d1[None, :] * D21 + d1[:, None] * d2[None, :] * D21.T)
        if diag_on and src:
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
N = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
ARMS = [(4, False, False), (4, True, False), (4, False, True), (4, True, True)]
res = {a: [] for a in ARMS}
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for a in ARMS:
        res[a].append(float(((chain(Wms, *a) - g) ** 2).mean()))
    print(f"  mlp {i}: " + " ".join(f"{'d' if a[1] else '-'}{'o' if a[2] else '-'}={res[a][-1]:.3e}"
                                    for a in ARMS), flush=True)
    del Wms, mlp

b = float(np.mean(res[(4, False, False)]))
print(f"\n=== Splitting the covariance correction at J=4 ({N} MLPs) ===")
print(f"SELF-CHECK no correction = {b:.4e} (the shipped chain)\n")
print(f"{'diag':>6} {'offdiag':>8} | {'final_layer_mse':>16} {'vs none':>9} | extra matmuls/layer")
for a in ARMS:
    v = float(np.mean(res[a]))
    mm = "0" if not a[2] else f"{2*a[0]}"
    print(f"{str(a[1]):>6} {str(a[2]):>8} | {v:>16.4e} {b/v:>8.2f}x | {mm:>18}")
