"""How much of the gap is the BIRTH magnitude, not the transport?

wick_regen_probe measured our regenerated source against truth at layer 1 (where the pre-activations
are exactly Gaussian, so the birth model is the whole story): correlation 0.998 but magnitude 0.918.
The missing 8.2% are the higher-order birth diagrams -- the covariance triangle and kappa3 x edge
products -- which ARC's own factored reference explicitly DROPS because they have no O(n)-rank
factorisation (see factor_k3.py's docstring). So neither we nor the reference can afford them.

But an 8% per-layer magnitude deficit compounds: 0.918^16 = 0.25. This sweeps a multiplicative factor
on the born source to see how much of the gap that explains, and whether the optimum sits near the
measured 1/0.918 = 1.089. A DIAGNOSTIC: if the answer is a large gain at a principled-looking value,
the factor gets derived, not fitted.
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
J = 4


def chain(Wms, scale):
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
        k3 = torch.zeros(n); D21 = torch.zeros(n, n) if src else None
        for A, Bm, d2m in src:
            AD = A * d2m
            k3 = k3 + 3.0 * (AD * Bm * Bm).sum(dim=1)
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
            # scale enters the born source through its hub coefficient d2
            src.append([None, C_pre * d1[:, None], d2 * scale])
            src[:] = src[-J:]
    return mu


gt = np.load(BAKED / "gt.npy")
N = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
SCALES = [0.90, 1.00, 1.05, 1.089, 1.15, 1.25, 1.40]
res = {s: [] for s in SCALES}
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for s in SCALES:
        res[s].append(float(((chain(Wms, s) - g) ** 2).mean()))
    print(f"  mlp {i} done", flush=True)
    del Wms, mlp

b = float(np.mean(res[1.00]))
print(f"\n=== Birth-magnitude sensitivity ({N} MLPs) ===")
print(f"SELF-CHECK scale = 1.00 : {b:.4e} (the shipped chain)\n")
print(f"{'scale':>7} | {'final_layer_mse':>16} {'vs 1.00':>9}")
for s in SCALES:
    v = float(np.mean(res[s]))
    print(f"{s:>7.3f} | {v:>16.4e} {b/v:>8.2f}x")
print("\nA flat curve means the birth magnitude is not the bottleneck and the deficit is being")
print("absorbed elsewhere. A sharp optimum near 1.089 would say the missing higher-order birth")
print("diagrams are exactly the gap, and the factor would then be worth deriving properly.")
