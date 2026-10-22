"""Sum over ALL cumulant sources, not just the previous layer: accuracy vs cost.

The shipped estimator regenerates kappa3 from the PREVIOUS layer only. But a third-cumulant source
is born at every ReLU layer and transported forward, so a depth-l pre-activation carries l of them.
That is why the one-step regeneration held 99.8% correlation at layer 1 and only 0.64 at depth.

Source born at ReLU layer m, from the leading Wick diagram (pre-activation stats mu^m, C^m):
    kappa3_h[a,b,c] ~= 3 * Sym( d2_a * u_b^(a) * u_c^(a) ),   U = diag(d1) C,  u^(a) = U[:, a]
Transported to the pre-activation of layer l by T = W^l diag(d1^{l-1}) ... diag(d1^{m+1}) W^{m+1}:
    kappa3_z[i,i,i] = 3 * sum_a d2^m_a * A[i,a] * B[i,a]^2,    A = T,  B = T U^m
Both A and B are left-multiplied by the same R^l = W^{l+1} diag(d1^l) each layer, so a source costs
2 matmuls per layer to carry, and 1 at birth. Keeping a WINDOW of the J most recent sources costs
    (16 births + 2 * sum_l min(l, J)) matmuls of n*n*(2n-1) FLOPs
on top of the k=2 chain's 5.18e10, and the score multiplier floors at 10% of B = 2**41.

This sweeps J and reports the resulting adjusted score, so the window can be chosen to sit exactly
at the floor. J=0 is plain k=2 (self-check), J=1 is what is currently submitted.
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
SKEW_CLIP = 1.0


def cost(J):
    updates = sum(min(l, J) for l in range(1, depth))
    births = min(J, depth - 1)
    return K2_FLOPS + (births + 2 * updates) * MATMUL


def chain(Wms, J):
    mu, C = torch.zeros(n), torch.eye(n)
    src = []                      # live sources: [A, B, d2] with A, B of shape (i, a)
    out = []
    for l in range(depth):
        Wm = Wms[l]
        # transport every live source across this linear layer: A <- Wm A, B <- Wm B
        for s in src:
            s[0] = Wm @ s[0]
            s[1] = Wm @ s[1]
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt()
        al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        d1, d2 = Ph, ph / sig

        k3 = torch.zeros(n)
        for A, Bm, d2m in src:
            k3 = k3 + 3.0 * ((A * d2m) * Bm * Bm).sum(dim=1)
        mu_g = mu_pre * Ph + sig * ph
        if J > 0 and src:
            skew = (k3 / (var * sig)).clamp(-SKEW_CLIP, SKEW_CLIP)
            mu = mu_g - skew * sig * al * ph / 6.0
        else:
            mu = mu_g
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        C = torch.outer(Ph, Ph) * C_pre
        C.diagonal().copy_((ez2 - mu_g * mu_g).clamp_min(0.0))

        # apply the ReLU linearisation to the carried sources, then give birth to this layer's own
        for s in src:
            s[0] = s[0] * d1[:, None]
            s[1] = s[1] * d1[:, None]
        if J > 0 and l < depth - 1:
            # The source is born in post-ReLU (h) space: hub leg carries d2, the other two carry
            # u^(a) = diag(d1) C[:, a]. No extra d1 on the legs -- the ReLU linearisation of THIS
            # layer is already inside U, and the loop above has applied it to the older sources only.
            U = C_pre * d1[:, None]                      # U[b, a] = d1_b * C_pre[b, a]
            src.append([torch.eye(n), U.clone(), d2.clone()])
            src[:] = src[-J:]
        out.append(mu.clone())
    return torch.stack(out)


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])
JS = [0, 1, 2, 3, 4, 6, 8, 15]
res = {J: [] for J in JS}
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for J in JS:
        res[J].append(float(((chain(Wms, J)[-1] - g) ** 2).mean()))
    print(f"  mlp {i}: " + " ".join(f"J{J}={res[J][-1]:.2e}" for J in JS), flush=True)
    del Wms, mlp

b = float(np.mean(res[0]))
print(f"\n=== Source-window sweep ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK J=0 = {b:.4e} (reference k=2 4.50e-06 on MLP 0)")
print("mean-only ceiling measured earlier: 2.22e-07 (23.2x)\n")
print(f"{'J':>3} | {'final_layer_mse':>16} {'vs k=2':>8} | {'% of B':>7} {'mult':>6} {'adjusted':>10}")
best = None
for J in JS:
    v = float(np.mean(res[J])); f = cost(J); pct = 100 * f / B_BUDGET
    mult = max(0.1, f / B_BUDGET); adj = v * mult
    flag = " OVER" if f > B_BUDGET else ""
    print(f"{J:>3} | {v:>16.4e} {b/v:>7.2f}x | {pct:>6.2f}% {mult:>6.3f} {adj:>10.3e}{flag}")
    if best is None or adj < best[1]:
        best = (J, adj)
print(f"\nbest adjusted: J={best[0]} at {best[1]:.3e}")
